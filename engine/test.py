# test the performance
# only train the deformation and metching network
# the source labels are selected randomly
import os
import json
import torch
import pytorch3d.ops
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime
import time
import torch.utils.data.dataloader
import sys
sys.path.append('./')
from train_utils.load_sources import load_sources

from tensorboardX import SummaryWriter
from dataset.dataset_utils import get_all_selected_models_pickle, get_random_labels, get_labels, get_source_points
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape
from dataset.partnet_dataset import partnet_dataset
from collections import defaultdict
from network.simple_encoder import TargetEncoder as simple_encoder
from network.deformation_net import DeformNet_MatchingNet as DM_decoder
from network.deformation_net import re_residual_net

from loss.chamfer_loss import compute_cm_loss
from tqdm import tqdm


def main(cfg):
    # torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(logdir=cfg["log_path"])
    if cfg["mode"] == 'train':
        DATA_SPLIT = 'train'
        bs = cfg["batch_size"]
    else:
        DATA_SPLIT = 'test'
        bs = cfg["batch_size"]  # must be 2

    dataset = partnet_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        shuffle=(True if cfg["mode"] == 'train' else False),
        drop_last=True,
    )
    print("loading sources")
    SOURCE_MODEL_INFO, _  = load_sources(cfg)
    # TODO:use all sources
    SOURCE_MODEL_INFO = SOURCE_MODEL_INFO#[:100]

    print("sources loaded")
    # construct deformation and matching network
    MAX_NUM_PARTS = 16
    src_encoder_all = simple_encoder(is_src=True)
    recon_decoder_src = re_residual_net(cfg['source_latent_dim'] * 2)

    recon_decoder_full = re_residual_net(cfg['target_latent_dim'] * 2)
    target_encoder_full = simple_encoder()  #
    
    param_decoder_full = DM_decoder(cfg['source_latent_dim'] * 3, graph_dim=cfg['source_latent_dim'],
                                    max_num_parts=MAX_NUM_PARTS, matching=False)

    if cfg["init_dm"]:
        fname = os.path.join(cfg["dm_model_path"])
        state_dict = torch.load(fname)
        target_encoder_full.load_state_dict(state_dict["target_encoder_full"])

        param_decoder_full.load_state_dict(state_dict["param_decoder_full"])

        recon_decoder_full.load_state_dict(state_dict["recon_decoder_full"])

        src_encoder_all.load_state_dict(state_dict["src_encoder_all"])
        recon_decoder_src.load_state_dict(state_dict["recon_decoder_src"])

        print("Initialize the dmnet, done!")

    target_encoder_full.to(cfg['device'], dtype=torch.float)

    param_decoder_full.to(cfg['device'], dtype=torch.float)

    recon_decoder_full.to(cfg['device'], dtype=torch.float)

    src_encoder_all.to(cfg['device'], dtype=torch.float)
    recon_decoder_src.to(cfg['device'], dtype=torch.float)

    # construct retrieval decoder
    re_order_decoder_full = re_residual_net(cfg['target_latent_dim'] * 2)
    if cfg["init_re"]:
        fname_re = os.path.join(cfg["re_model_path"])
        state_dict = torch.load(fname_re)
        re_order_decoder_full.load_state_dict(state_dict["re_residual_net_full"])
    re_order_decoder_full = re_order_decoder_full.to(cfg['device'], dtype=torch.float)

    # turn all branches into test mode
    target_encoder_full.eval()
    param_decoder_full.eval()
    recon_decoder_full.eval()
    re_order_decoder_full.eval()
    src_encoder_all.eval()
    recon_decoder_src.eval()

    np.random.rand(int(time.time()))

    # get src latent codes
    print("calculating source labels for retrieval")
    source_labels_all = np.expand_dims(np.arange(len(SOURCE_MODEL_INFO)), axis=1)
    source_labels_all = np.reshape(source_labels_all, (-1, 1))

    '''
     re_src_latent_codes_template = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES,
                                                                 device=cfg['device'])
     '''
    # src_mats: num x [1, 3072, 6], src_default_params: num x [1, 6]
    src_mats, src_default_params = get_source_info(source_labels_all, SOURCE_MODEL_INFO)
    # src_points_cloud:[6340, 1, 1024, 3]
    src_points_cloud = get_source_points(source_labels_all, SOURCE_MODEL_INFO,
                                                                  device=cfg["device"])
    src_latent_codes_all = []
    src_per_point_f = []
    for i in range(len(src_points_cloud)//512):
        src_points_cloud_now = src_points_cloud[i*512:(i+1)*512]
        with torch.no_grad():
            src_latent_codes_all_now, src_per_point_f_now = src_encoder_all(src_points_cloud_now)
        src_latent_codes_all.append(src_latent_codes_all_now)
        src_per_point_f.append(src_per_point_f_now)
        
    src_points_cloud_now = src_points_cloud[(i+1)*512:]
    with torch.no_grad():
        src_latent_codes_all_now, src_per_point_f_now = src_encoder_all(src_points_cloud_now)
    src_latent_codes_all.append(src_latent_codes_all_now)
    src_per_point_f.append(src_per_point_f_now)
    
    src_latent_codes_all = torch.cat(src_latent_codes_all)
    src_per_point_f = torch.cat(src_per_point_f)
    src_per_point_f = src_per_point_f.permute(0, 2, 1)

    best_cd_loss_full_all = []
    best_re_loss_full_all = []
    best_re_cd_loss_full_all = []

    for i, batch in enumerate(loader):
        print(str(i), '/', str(len(loader)))
        target_shapes, target_id, target_labels, semantics, point_occ, point_occ_mask, ori_point_occ = batch
        if target_shapes.shape[0] <= 1:  # bs should be larger than 1
            break
        # forward pass the deformation and matching network to get the loss

        # x:[bs, 2048, 3]
        x = [x.to(cfg["device"], dtype=torch.float) for x in target_shapes]
        x = torch.stack(x)
        # pdb.set_trace()

        # transfer data to gpu

        mat = [mat.to(cfg["device"], dtype=torch.float) for mat in src_mats]
        def_param = [def_param.to(cfg["device"], dtype=torch.float) for def_param in src_default_params]

        # mat_all:[num, 1, 3072, 6], def_param_all:[num, 1, 6]
        mat_all = torch.stack(mat)
        def_param_all = torch.stack(def_param)

        # forward pass
        # here maybe some problems
        # target_latent_codes_full:2 x 256, per_point_full:2 x 256 x 2048
        with torch.no_grad():
            target_latent_codes_full, per_point_full = target_encoder_full(x)
        per_point_full = per_point_full.permute(0, 2, 1)
        target_part_f = []
        target_part_per_point = []
        re_input_codes_full = []
        mask_part = torch.zeros(bs, MAX_NUM_PARTS).to(cfg["device"])
        for w in range(per_point_full.shape[0]):
            part_per_point_f_now = []
            part_f_now = []
            re_input_codes_full_now = []
            unique_sem = torch.unique(target_labels[w])
            for sem in unique_sem:
                part_per_point_f_now.append(per_point_full[w, target_labels[w,:] == sem, :])
                part_f_now.append(torch.mean(per_point_full[w, target_labels[w,:] == sem, :], dim=0))
                nofp = part_per_point_f_now[-1].shape[0]
                re_input_codes_full_now.append(torch.cat([part_per_point_f_now[-1], part_f_now[-1].unsqueeze(0).repeat(nofp ,1)], dim=-1))
            mask_part[w, :len(part_f_now)] = 1
            target_part_per_point.append(part_per_point_f_now)
            target_part_f.append(torch.stack(part_f_now))
            re_input_codes_full.append(torch.cat(re_input_codes_full_now, dim=0))
        re_input_codes_full = torch.stack(re_input_codes_full)
                
        best_cd_loss_full = []
        best_re_loss_full = []
        best_re_cd_loss_full = []

        # for _ in tqdm(range(4 // cfg["batch_size"])):
        #     cd_loss_full_list = []
        #     re_loss_full_list = []
        #     # bs x 256
        #     for j in range(src_latent_codes_all.shape[0]):
        source_label_shape = torch.zeros(target_shapes.shape[0], MAX_NUM_PARTS)  # batchsize
        source_labels = get_labels(source_label_shape, len(SOURCE_MODEL_INFO), target_id, target_labels)
        # source_labels = get_random_labels(source_label_shape, len(SOURCE_MODEL_INFO), target_id)
        with torch.no_grad():
            src_latent_codes = src_latent_codes_all[source_labels, :]
            mat = mat_all[source_labels, :].squeeze(2)
            def_param = def_param_all[source_labels, :].squeeze(2)
            # construct the input of the retrieval residual network
            # here only consider the partial part

            # select most times

            # for target recon
            # recon_partial_p = recon_decoder_partial(recon_input_codes_partial)

            nofp = per_point_full.shape[-1]
            recon_target_codes_full = target_latent_codes_full.unsqueeze(1).repeat(1, nofp, 1)
            recon_input_codes_full = torch.cat([per_point_full.permute(0, 2, 1),
                                                recon_target_codes_full], dim=-1)
            # recon_full_p = recon_decoder_full(recon_input_codes_full)

            # re_residuals  full
            # re_source_latent_codes_full = src_latent_codes.unsqueeze(1).repeat(1, nofp, 1)
            # re_input_codes_full = torch.cat([recon_input_codes_full, re_source_latent_codes_full], dim=-1)
            # TODO: not related to src?
            re_residuals_full = re_order_decoder_full(re_input_codes_full)

            loss_re_full = torch.mean(torch.sum(torch.abs(re_residuals_full), dim=-1), dim=1)

            '''
            loss_re_full, _ = torch.topk(torch.sum(torch.abs(re_residuals_full), dim=-1),
                                        k=cfg['top_k'], dim=-1, largest=False)
            loss_re_partial, _ = torch.topk(torch.sum(torch.abs(re_residuals_partial), dim=-1),
                                            k=cfg['top_k'], dim=-1, largest=False)
            '''

            # re_loss_full_list.append(loss_re_full)
            
            # src_latent_codes: 2 x 256,  per_point_full: 2 x 2048 x 256
            params_full = param_decoder_full(target_latent_codes_full,
                                            src_latent_codes,
                                            per_point_full)


                # params_full = torch.stack(params_full)

                # # using params to get deformed shape
                # # def param, original bounding box info

            output_pc_from_full = get_shape(mat, params_full, def_param, cfg["alpha"])
            # TODO:calculate loss for each part
            output_pc_from_full = output_pc_from_full.reshape(bs, -1, 3)

            cd_loss_full = compute_cm_loss(output_pc_from_full.detach(), x, mask_part, batch_reduction=None)

            # re_loss_full_list = torch.stack(re_loss_full_list)

            best_cd_loss_full_otm = []
            best_re_loss_full_otm = []
            best_re_cd_loss_full_otm = []
            for q in range(cfg['batch_size']):
                cd_loss_full_list_now = cd_loss_full[q]
                best_cd_loss_full_otm.append(cd_loss_full_list_now)

                re_loss_full_list_now = loss_re_full[q]

                best_re_loss_full_otm.append(re_loss_full_list_now)
                best_re_cd_loss_full_otm.append(cd_loss_full_list_now)

            best_cd_loss_full.append(torch.stack(best_cd_loss_full_otm))
            best_re_loss_full.append(torch.stack(best_re_loss_full_otm))
            best_re_cd_loss_full.append(torch.stack(best_re_cd_loss_full_otm))
        
        best_cd_loss_full_all.append(torch.min(torch.stack(best_cd_loss_full), dim=0)[0])
        best_re_loss_full_all.append(torch.min(torch.stack(best_re_loss_full), dim=0)[0])
        best_re_cd_loss_full_all.append(torch.min(torch.stack(best_re_cd_loss_full), dim=0)[0])

    best_re_cd_loss_full_all = torch.stack(best_re_cd_loss_full_all)
    best_cd_loss_full_all = torch.stack(best_cd_loss_full_all)
    best_re_loss_full_all = torch.stack(best_re_loss_full_all)

    print("best full cd loss from retrieval=" + str(torch.mean(torch.stack(best_re_cd_loss_full)).cpu().numpy()),
          "best full cd loss=" + str(torch.mean(torch.stack(best_cd_loss_full)).cpu().numpy()),
          "best full re loss=" + str(torch.mean(torch.stack(best_re_loss_full)).cpu().numpy()),
          )


if __name__ == '__main__':
    import json

    config_path = "config/config_test_test.json"#sys.argv[1]
    config = json.load(open(config_path, 'rb'))
    main(config)
    
    pass
