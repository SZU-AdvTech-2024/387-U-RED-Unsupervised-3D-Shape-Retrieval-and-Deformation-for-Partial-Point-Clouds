# experiment version

# only train the deformation and metching network
# the source labels are selected randomly
import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime
import time
import sys
sys.path.append('./')
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import torch.utils.data.dataloader
from train_utils.load_sources import load_sources
from train_utils.optimizer_dm import define_optimizer_dm_re_recon

from tensorboardX import SummaryWriter
from dataset.dataset_utils import get_source_points, get_labels
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape, compute_aabbox
from dataset.dataset_utils import get_symmetric
from dataset.partnet_dataset import partnet_dataset
from network.simple_encoder import TargetEncoder as simple_encoder
from network.deformation_net import DeformNet_MatchingNet as DM_decoder
from network.deformation_net import re_residual_net

from loss.chamfer_loss import compute_cm_loss
from loss.basic_loss import point_loss_matching, residual_retrieval_loss
from loss.basic_consistency_loss import compute_pc_consistency, compute_param_consistency, compute_pc_consistency_weighted
from loss.contrast_loss import compute_contrast_loss_loss
from loss.regularization_loss import regularization_param

            
def get_models(cfg):
    src_encoder_all = simple_encoder(cfg["source_latent_dim"], is_src=True, sem_size=cfg["sem_latent_dim"])
    recon_decoder_src = re_residual_net(cfg['source_latent_dim'] * 2)
    
    target_encoder_full = simple_encoder(cfg['target_latent_dim'], sem_size=cfg["sem_latent_dim"])  #
    recon_decoder_full = re_residual_net(cfg['target_latent_dim'] * 2)
    
    param_decoder_full = DM_decoder(cfg['source_latent_dim'] * 3, graph_dim=cfg['source_latent_dim'], ##
                                    max_num_parts=cfg['MAX_NUM_PARTS'], matching=False)
    embedding_layer = nn.Embedding(42, cfg["sem_latent_dim"])

    if cfg["init_dm"]:
        fname = os.path.join(cfg["dm_model_path"])
        state_dict = torch.load(fname)
        target_encoder_full.load_state_dict(state_dict["target_encoder_full"])

        param_decoder_full.load_state_dict(state_dict["param_decoder_full"])

        recon_decoder_full.load_state_dict(state_dict["recon_decoder_full"])

        src_encoder_all.load_state_dict(state_dict["src_encoder_all"])
        recon_decoder_src.load_state_dict(state_dict["recon_decoder_src"])
        embedding_layer.load_state_dict(state_dict["embedding_layer"])

        print("Initialize the dmnet, done!")

    target_encoder_full.to(cfg['device'], dtype=torch.float)

    param_decoder_full.to(cfg['device'], dtype=torch.float)

    recon_decoder_full.to(cfg['device'], dtype=torch.float)

    src_encoder_all.to(cfg['device'], dtype=torch.float)
    recon_decoder_src.to(cfg['device'], dtype=torch.float)
    embedding_layer.to(cfg['device'], dtype=torch.float)

    # construct retrieval decoder
    re_order_decoder_full = re_residual_net(cfg['target_latent_dim'] * 2)
    if cfg["init_re"]:
        fname_re = os.path.join(cfg["re_model_path"])
        state_dict = torch.load(fname_re)
        re_order_decoder_full.load_state_dict(state_dict["re_residual_net_full"])
    re_order_decoder_full = re_order_decoder_full.to(cfg['device'], dtype=torch.float)

    # define optimizer and scheduler
    optimizer, scheduler = define_optimizer_dm_re_recon(target_encoder_full,
                                                        param_decoder_full,
                                                        recon_decoder_full,
                                                        re_order_decoder_full,
                                                        src_encoder_all, recon_decoder_src,
                                                        embedding_layer,
                                                        cfg)
    
    # start training
    target_encoder_full.train()
    param_decoder_full.train()
    recon_decoder_full.train()
    re_order_decoder_full.train()
    src_encoder_all.train()
    recon_decoder_src.train()
    embedding_layer.train()

    return target_encoder_full, param_decoder_full, recon_decoder_full, re_order_decoder_full, src_encoder_all, recon_decoder_src, embedding_layer, optimizer, scheduler

def get_part(cfg, per_point_full, target_labels, x):
    target_part_f = []
    target_part_per_point = []
    re_input_codes_full = []
    part_x = []
    mask_part = torch.zeros(cfg['batch_size'], cfg['MAX_NUM_PARTS']).to(cfg["device"])
    param_def = torch.zeros(cfg['batch_size'], cfg['MAX_NUM_PARTS'], 6).to(cfg["device"])
    # for each patch
    for w in range(per_point_full.shape[0]):
        part_per_point_f_now = []
        part_f_now = []
        re_input_codes_full_now = []
        part_x_now = []
        unique_sem = torch.unique(target_labels[w])
        for sem in unique_sem:
            sem_idx = target_labels[w,:] == sem
            part_x_now.append(x[w, sem_idx, :])
            param_def[w, int(sem)] = compute_aabbox(part_x_now[-1])
            per_f_now = per_point_full[w, sem_idx, :]
            part_per_point_f_now.append(per_f_now)
            part_f_now.append(torch.mean(per_f_now, dim=0))
            nofp = part_per_point_f_now[-1].shape[0]
            re_input_codes_full_now.append(torch.cat([part_per_point_f_now[-1], part_f_now[-1].unsqueeze(0).repeat(nofp ,1)], dim=-1))
        part_x.append(part_x_now)
        target_part_per_point.append(part_per_point_f_now)
        part_f_now = torch.stack(part_f_now)
        padded_part_f_now = torch.zeros(cfg['MAX_NUM_PARTS'], part_f_now.shape[-1]).to(cfg["device"])
        padded_part_f_now[:part_f_now.shape[0], :] = part_f_now
        mask_part[w, :part_f_now.shape[0]] = 1
        target_part_f.append(padded_part_f_now)
        re_input_codes_full.append(torch.cat(re_input_codes_full_now, dim=0))
    target_part_f = torch.stack(target_part_f)  # bs x MAX_NUM_PARTS x 256 
    re_input_codes_full = torch.stack(re_input_codes_full)  # bs x 2048 x 512
    return target_part_f, target_part_per_point, re_input_codes_full, mask_part, part_x, param_def
     
def save_model(model, start, epoch, cfg):
    now = datetime.datetime.now()
    duration = (now - start).total_seconds()
    log = "> {} | Epoch [{:04d}/{:04d}] | duration: {:.1f}s |"
    log = log.format(now.strftime("%c"), epoch, cfg["epochs"], duration)

    fname = os.path.join(cfg["log_path"], "checkpoint_{:04d}.pth".format(epoch))
    print("> Saving model to {}...".format(fname))
    torch.save(model, fname)

    fname = os.path.join(cfg["log_path"], "train.log")
    with open(fname, "a") as fp:
        fp.write(log + "\n")

    print(log)
    print("--------------------------------------------------------------------------")
    
    
def main(cfg):
    # torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(logdir=cfg["log_path"])
    if cfg["mode"] == 'train':
        DATA_SPLIT = 'train'
        bs = cfg["batch_size"]
    else:
        DATA_SPLIT = 'test'
        bs = 2

    dataset = partnet_dataset(cfg)
    # TODO:debug when drop_last=False
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        shuffle=(True if cfg["mode"] == 'train' else False),
        drop_last=True,
    )
    print("loading sources")
    SOURCE_MODEL_INFO, dist_src = load_sources(cfg)

    print("sources loaded")
    # construct deformation and matching network
    
    target_encoder_full, param_decoder_full, recon_decoder_full, re_order_decoder_full, src_encoder_all, recon_decoder_src, embedding_layer, optimizer, scheduler = get_models(cfg)
    sources_sem = [x['sem_label'] for x in SOURCE_MODEL_INFO]

    for epoch in range(cfg["epochs"]):
        np.random.seed(int(time.time()))
        start = datetime.datetime.now()
        print(str(start), 'training epoch', str(epoch))
        for i, batch in enumerate(loader):
            target_shapes, target_id, target_labels, semantics, point_occ, point_occ_mask, ori_point_occ = batch
            if cfg["complementme"]:
                target_shapes[:, :, 2] = -target_shapes[:, :, 2]
                point_occ[:, :, 2] = -point_occ[:, :, 2]
                
            source_label_shape = np.full([target_shapes.shape[0], cfg['MAX_NUM_PARTS']], -1)  # batchsize
            source_labels, tgt_sem_idx = get_labels(source_label_shape, target_id, target_labels, cfg['filter_threshold'], sources_sem, dist_src, cfg['cl_k'])
            
            # img_list = img
            # gimg=make_grid(img_list,4,normalize=True)
            # writer.add_image('img',gimg, i)
                
            # src_mats: bs x [num_part, 3072, 6], src_default_params: bs x [num_part, 6] ###MAX_PART_NUM x 6:deform lable ,src_connectivity_mat: bs x [96 x 96] TODO:What is this?
            src_mats, src_default_params, src_sem_idx = get_source_info(source_labels, SOURCE_MODEL_INFO)
            src_sem_f = embedding_layer(src_sem_idx.to(cfg["device"]))
            tgt_sem_f = embedding_layer(tgt_sem_idx.to(cfg["device"]))

            # src_points_cloud: [bs x 2048 x 3], point_labels: [bs x 2048], num_parts: [bs]
            src_points_cloud = get_source_points(source_labels, SOURCE_MODEL_INFO,
                                                                            device=cfg["device"]) # 64 x 16 x 1024 x 3
            src_latent_codes, src_per_point_f = src_encoder_all(src_points_cloud, src_sem_f) # codes: [bs*max_part, C], per_point_f: [bs*max_part, C, N]
            
            # for src reconstrcution
            recon_src_latent_codes = src_latent_codes.unsqueeze(2).repeat(1, 1, src_per_point_f.shape[-1])
            recon_src_input_f = torch.cat([recon_src_latent_codes, src_per_point_f], dim=1)
            recon_src_p = recon_decoder_src(recon_src_input_f.permute(0, 2, 1)) # [bs x 2048 x 3]
            recon_src_p = recon_src_p.reshape(bs, cfg['MAX_NUM_PARTS'], -1, 3)

            src_per_point_f = src_per_point_f.permute(0, 2, 1)
            

            # transfer data to gpu
            x = [x.to(cfg["device"], dtype=torch.float) for x in target_shapes] # bs x 2048 x 3 
            semantics = [sem.to(cfg["device"], dtype=torch.int64) for sem in semantics] # bs x [2048]
            target_labels = [label.to(cfg["device"], dtype=torch.float) for label in target_labels] # bs x 1024 x 3
            mat = [mat.to(cfg["device"], dtype=torch.float) for mat in src_mats] # bs x [6144 x 96]
            # def_param = [def_param.to(cfg["device"], dtype=torch.float) for def_param in src_default_params]

            x = torch.stack(x)
            semantics = torch.stack(semantics)
            target_labels = torch.stack(target_labels)
            mat = torch.stack(mat)
            # def_param = torch.stack(def_param)
            # def_param = None

            target_latent_codes_full, per_point_full = target_encoder_full(x, tgt_sem_f)
            per_point_full = per_point_full.permute(0, 2, 1)
            
            ### use src_per_point_f to generate per part features
            target_part_f, target_part_per_point, re_input_codes_full, mask_part, part_x, param_def = get_part(cfg, per_point_full, target_labels, x)

            # target_part_per_point: bs x [num_parts x N x 256], num_parts is not fixed for every batch, N is the number of points of each part, not fixed
            

            # forward pass
            # for target recon
            nofp = per_point_full.shape[-2]
            recon_target_codes_full = target_latent_codes_full.unsqueeze(1).repeat(1, nofp, 1)
            recon_input_codes_full = torch.cat([per_point_full, recon_target_codes_full], dim=-1)
            recon_full_p = recon_decoder_full(recon_input_codes_full)

            # re_residuals  full TODO:design loss for each parts
            # re_source_latent_codes_full = src_latent_codes.unsqueeze(1).repeat(1, nofp, 1) # bs x 2048 x 256
            # re_input_codes_full = torch.cat(target_part_per_point, target_part_f)
            # re_input_codes_full = torch.cat([re_input_codes_full, re_source_latent_codes_full], dim=-1) # bs x 2048 x 768
            # re_input_codes_full = torch.cat([recon_input_codes_full, re_source_latent_codes_full], dim=-1) # bs x 2048 x 768
            re_residuals_full = re_order_decoder_full(re_input_codes_full) # bs x 2048 x 3 ： p2->p1 for each point in p1 with least distance 

            # construct the input of the retrieval residual network
            # src_latent_codes: bs x num_part x c, target_latent_codes_full: bs x 256, per_point_full: bs x 2048 x 256
            src_latent_codes = src_latent_codes.reshape(bs, cfg['MAX_NUM_PARTS'], -1)
            params_full = param_decoder_full(target_latent_codes_full,
                                                src_latent_codes,
                                                per_point_full)

            # TODO:observe the change of match_full and match_partial

            # using params to get deformed shape
            # mat:[64, 6144, 96] params_full: [64, 96, 1] def_param: [64, 1, 96] connectivity_mat: [bs, 96, 96] -> [bs, 2048, 3]
            output_pc_from_full = get_shape(mat, params_full, param_def, cfg["alpha"])
            output_pc_from_full = output_pc_from_full.reshape(bs, -1, 3)
            
            # TODO:regularization loss maybe slow down the training
            source_labels[source_labels>=0] = 1
            
            
            #  compute losses
            loss_all = 0.0

            # TODO: Set suitable num_part and implement mask 
            if cfg["use_param_loss"] > 0.0:
                param_loss = regularization_param(params_full, mask_part)
                loss_all += param_loss * cfg["use_param_loss"]
                writer.add_scalar('param_loss', param_loss.item(), global_step=epoch * len(loader) + i)
            if cfg["use_chamfer_loss"] > 0.0:
                # note that here, the deformed shape, no matter from partial or full, should be align to x
                # deformed src and target
                cd_loss_full, cd_loss_part = compute_cm_loss(output_pc_from_full, x, part_x, mask_part)
                loss_all += cd_loss_full * cfg["use_chamfer_loss"]
                loss_all += cd_loss_part * cfg["use_chamfer_part_loss"]
                writer.add_scalar('cd_loss_full', cd_loss_full.item(), global_step=epoch * len(loader) + i)
                writer.add_scalar('cd_loss_part', cd_loss_part.item(), global_step=epoch * len(loader) + i)
                
            if cfg["use_contrast_loss"] > 0.0:
                contrast_loss = compute_contrast_loss_loss(target_part_f, src_latent_codes, source_labels)
                loss_all += contrast_loss * cfg["use_contrast_loss"]
                writer.add_scalar('contrast_loss', contrast_loss.item(), global_step=epoch * len(loader) + i)
                
            if cfg["use_symmetry_loss"] > 0.0:
                # symmetrized deformed shape and target
                ref_pc_full = get_symmetric(output_pc_from_full)
                ref_cd_loss_full, ref_cd_loss_part = compute_cm_loss(ref_pc_full, x, part_x, mask_part)
                loss_all += ref_cd_loss_full * cfg["use_symmetry_loss"]
                # loss_all += ref_cd_loss_part * cfg["use_chamfer_part_loss"]
                writer.add_scalar('ref_cd_loss_full', ref_cd_loss_full.item(), global_step=epoch * len(loader) + i)
                # writer.add_scalar('ref_cd_loss_part', ref_cd_loss_part.item(), global_step=epoch * len(loader) + i)
                
            if cfg["use_residuals_reg"] > 0.0 and epoch > cfg['init_p_m_loss']:
                # x: target shape, output_pc_from_full: deformed shape from full src, re_residuals_full: [bs x 2048 x 3]
                # residual_retraival_loss: use the target shape and the deformed shape and the residual(p2 for each point in p1 with least distance) to compute the loss
                re_residual_loss_full, reg_residual_full = residual_retrieval_loss(x,
                                                                                    output_pc_from_full.detach(),
                                                                                    re_residuals_full, mask_part)

                loss_all += re_residual_loss_full * cfg["use_residuals_reg"]
                loss_all += reg_residual_full * cfg["use_residuals_reg"] * 0.01

                writer.add_scalar('re_reg_loss_full',
                                    re_residual_loss_full.item(), global_step=epoch * len(loader) + i)
                writer.add_scalar('reg_loss_full', reg_residual_full.item(),
                                    global_step=epoch * len(loader) + i)

            if cfg["use_recon"] > 0.0:
                # recon for src and target
                recon_loss_full = compute_pc_consistency(recon_full_p, x)
                loss_all += recon_loss_full * cfg["use_recon"]
                writer.add_scalar('recon_loss_full',
                                    recon_loss_full.item(), global_step=epoch * len(loader) + i)

                # TODO：implement weight for each part
                recon_loss_src = compute_pc_consistency_weighted(recon_src_p, src_points_cloud, mask_part)
                loss_all += recon_loss_src * cfg["use_recon"]
                writer.add_scalar('recon_loss_src', recon_loss_src.item(),
                                    global_step=epoch * len(loader) + i)
            writer.add_scalar('all_loss', loss_all.item(), global_step=epoch * len(loader) + i)
            
            optimizer.zero_grad()
            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(target_encoder_full.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(param_decoder_full.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(re_order_decoder_full.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(recon_decoder_full.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(recon_decoder_src.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(src_encoder_all.parameters(), 5.0)
            optimizer.step()
        scheduler.step()  # decrease the learning rate by 0.7 every 30 epochs

        # save model
        if ((epoch + 1) % cfg['save_epoch'] == 0):
            # Summary after each epoch
            model = {"target_encoder_full": target_encoder_full.state_dict(),
                        "param_decoder_full": param_decoder_full.state_dict(),
                        "re_residual_net_full": re_order_decoder_full.state_dict(),
                        "recon_decoder_full": recon_decoder_full.state_dict(),
                        "src_encoder_all": src_encoder_all.state_dict(),
                        "recon_decoder_src": recon_decoder_src.state_dict(),
                        "embedding_layer": embedding_layer.state_dict()}
            save_model(model, start, epoch, cfg)


if __name__ == '__main__':
    # config_path = sys.argv[1]
    config_path = "config/config_train_test.json"
    config = json.load(open(config_path, 'rb'))
    # os.system("mkdir " + config["log_path"])
    os.makedirs(config["log_path"], exist_ok=True)
    os.system("cp " + config_path + " " + config["log_path"])
    main(config)
    
    
    pass
