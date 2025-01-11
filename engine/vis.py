# test the performance
# only train the deformation and metching network
# the source labels are selected randomly
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import torch.utils.data.dataloader
import sys
from sklearn import metrics
sys.path.append('./')
from train_utils.load_sources import load_sources

from tensorboardX import SummaryWriter
from dataset.dataset_utils import get_all_selected_models_pickle, get_source_points, get_labels_from_cl, cal_retrieval_score, compute_aabbox, get_tgt_semantics
from dataset.dataset_utils import get_source_info, get_source_latent_codes_fixed, get_shape, get_source_info_mesh, get_source_info_visualization, get_shape_numpy, output_visualization_mesh
from dataset.partnet_dataset import partnet_dataset
from network.simple_encoder import TargetEncoder as simple_encoder
from network.deformation_net import DeformNet_MatchingNet as DM_decoder
from network.deformation_net import re_residual_net

from loss.chamfer_loss import compute_cm_loss

## TODO: simplify the code

def main(cfg):
    # torch.autograd.set_detect_anomaly(True)

    writer = SummaryWriter(logdir=cfg["log_path"])
    if cfg["mode"] == 'train':
        DATA_SPLIT = 'train'
    else:
        DATA_SPLIT = 'test'

    dataset = partnet_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        shuffle=(True if cfg["mode"] == 'train' else False),
    )
    print("loading sources")
    SOURCE_MODEL_INFO, dist_src = load_sources(cfg)
    SOURCE_MODEL_INFO = SOURCE_MODEL_INFO
    # sim_matrix = np.exp(dist_src ** 2 / (2.0 * 0.01 ** 2))
    
    print("sources loaded")
    # construct deformation and matching network
    src_encoder_all = simple_encoder(cfg["source_latent_dim"], is_src=True, sem_size=cfg["sem_latent_dim"])
    recon_decoder_src = re_residual_net(cfg['source_latent_dim'] * 2)
    
    recon_decoder_full = re_residual_net(cfg['target_latent_dim'] * 2)
    target_encoder_full = simple_encoder(cfg['target_latent_dim'], sem_size=cfg["sem_latent_dim"])  #
    
    param_decoder_full = DM_decoder(cfg['source_latent_dim'] * 3, graph_dim=cfg['source_latent_dim'],
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

    # turn all branches into test mode
    target_encoder_full.eval()
    param_decoder_full.eval()
    recon_decoder_full.eval()
    re_order_decoder_full.eval()
    src_encoder_all.eval()
    recon_decoder_src.eval()
    embedding_layer.eval()

    np.random.rand(int(time.time()))

    # get src latent codes
    print("calculating source labels for retrieval")
    source_labels_all = np.expand_dims(np.arange(len(SOURCE_MODEL_INFO)), axis=1)
    source_labels_all = np.reshape(source_labels_all, (-1, 1))

    '''
     re_src_latent_codes_template = get_source_latent_codes_fixed(source_labels, RETRIEVAL_SOURCE_LATENT_CODES,
                                                                 device=cfg['device'])
     '''

    src_mats, src_default_params, src_sem_idx = get_source_info(source_labels_all, SOURCE_MODEL_INFO)

    src_points_cloud = get_source_points(source_labels_all, SOURCE_MODEL_INFO,
                                                                  device=cfg["device"])
    
    src_sem_f = embedding_layer(src_sem_idx.to(cfg["device"]))
    # src_sem_f = embedding_layer(src_sem_idx).to(cfg["device"], dtype=torch.float)
    
    src_latent_codes_all = []
    src_per_point_f = []
    for i in range(len(src_points_cloud)//512):
        src_points_cloud_now = src_points_cloud[i*512:(i+1)*512]
        src_sem_f_now = src_sem_f[i*512:(i+1)*512]
        with torch.no_grad():
            src_latent_codes_all_now, src_per_point_f_now = src_encoder_all(src_points_cloud_now, src_sem_f_now)
        src_latent_codes_all.append(src_latent_codes_all_now)
        src_per_point_f.append(src_per_point_f_now)
        
    src_points_cloud_now = src_points_cloud[(i+1)*512:]
    src_sem_f_now = src_sem_f[(i+1)*512:]
    with torch.no_grad():
        src_latent_codes_all_now, src_per_point_f_now = src_encoder_all(src_points_cloud_now, src_sem_f_now)
    src_latent_codes_all.append(src_latent_codes_all_now)
    src_per_point_f.append(src_per_point_f_now)
    
    src_latent_codes_all = torch.cat(src_latent_codes_all)
    src_per_point_f = torch.cat(src_per_point_f)
    src_per_point_f = src_per_point_f.permute(0, 2, 1)

    best_cd_loss_full = []
    best_re_loss_full = []
    best_re_cd_loss_full = []
    elem_all = 0
    total_all = 0

    for i, batch in enumerate(loader):
        print(str(i), '/', str(len(loader)))
        target_shapes, target_ids, target_labels, semantics, point_occ, point_occ_mask, ori_point_occ = batch
        # forward pass the deformation and matching network to get the loss
        x = [x.to(cfg["device"], dtype=torch.float) for x in target_shapes]
        x = torch.stack(x)

        # transfer data to gpu
        mat = [mat.to(cfg["device"], dtype=torch.float) for mat in src_mats]
        def_param = [def_param.to(cfg["device"], dtype=torch.float) for def_param in src_default_params]

        mat_all = torch.stack(mat)
        def_param_all = torch.stack(def_param)

        tgt_sem_idx = get_tgt_semantics(target_ids, target_labels)
        tgt_sem_f = embedding_layer(tgt_sem_idx.to(cfg["device"]))

        # forward pass
        # here maybe some problems
        with torch.no_grad():
            target_latent_codes_full, per_point_full = target_encoder_full(x, tgt_sem_f)
        per_point_full = per_point_full.permute(0, 2, 1)
        target_part_f = []
        re_input_codes_full = []
        mask_part = torch.zeros(cfg['batch_size'], cfg['MAX_NUM_PARTS']).to(cfg["device"])
        param_def = torch.zeros(cfg['batch_size'], cfg['MAX_NUM_PARTS'], 6).to(cfg["device"])
        for w in range(per_point_full.shape[0]):
            part_per_point_f_now = []
            part_f_now = []
            re_input_codes_full_now = []
            unique_sem = torch.unique(target_labels[w])
            for sem in unique_sem:
                sem_idx = target_labels[w,:] == sem
                param_def[w, int(sem)] = compute_aabbox(x[w, sem_idx, :])
                part_per_point_f_now.append(per_point_full[w, sem_idx, :])
                part_f_now.append(torch.mean(per_point_full[w, sem_idx, :], dim=0))
                nofp = part_per_point_f_now[-1].shape[0]
                re_input_codes_full_now.append(torch.cat([part_per_point_f_now[-1], part_f_now[-1].unsqueeze(0).repeat(nofp ,1)], dim=-1))
            mask_part[w, :len(part_f_now)] = 1
            target_part_f.append(torch.stack(part_f_now))
            re_input_codes_full.append(torch.cat(re_input_codes_full_now, dim=0))
        target_part_f = torch.stack(target_part_f)  # bs x MAX_NUM_PARTS x 256 
        re_input_codes_full = torch.stack(re_input_codes_full)
        
        target_part_f = F.normalize(target_part_f, dim=-1, p=2)
        src_latent_codes_all = F.normalize(src_latent_codes_all, dim=-1, p=2)
        
        # pdb.set_trace()
        # source_label_shape = np.full([target_shapes.shape[0], cfg['MAX_NUM_PARTS']], -1)  # batchsize
        topk_idx = torch.topk(target_part_f @ src_latent_codes_all.t(), k=5232, dim=-1)[1]
        # scores = metrics.ndcg_score(y_true, y_score)
        contrast_label = torch.full((target_shapes.shape[0], cfg['MAX_NUM_PARTS']), -1, dtype=torch.int32)
        contrast_label[0,:unique_sem.shape[0]] = torch.argmax(target_part_f @ src_latent_codes_all.t(), dim=-1)
        ndcg_score = cal_retrieval_score(target_ids, target_labels, (target_part_f @ src_latent_codes_all.t()).squeeze(0).cpu())
        # elem_all += elem
        # total_all += total
        source_labels = contrast_label
        # source_labels = get_random_labels(source_label_shape, len(SOURCE_MODEL_INFO), target_ids)
        with torch.no_grad():
            src_latent_codes = src_latent_codes_all[source_labels, :]
            mat = mat_all[source_labels, :].squeeze(2)
            def_param = def_param_all[source_labels, :].squeeze(2)
            def_param = None

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

            re_residuals_full = re_order_decoder_full(re_input_codes_full)

            loss_re_full, _ = torch.max(torch.sum(torch.abs(re_residuals_full), dim=-1), dim=-1)

            '''
            loss_re_full, _ = torch.topk(torch.sum(torch.abs(re_residuals_full), dim=-1),
                                            k=cfg['top_k'], dim=-1, largest=False)
            loss_re_partial, _ = torch.topk(torch.sum(torch.abs(re_residuals_partial), dim=-1),
                                            k=cfg['top_k'], dim=-1, largest=False)
            '''

            # re_loss_full_list.append(loss_re_full)

            params_full = param_decoder_full(target_latent_codes_full,
                                            src_latent_codes,
                                            per_point_full)


            # using params to get deformed shape
            # def param, original bounding box info
            # pdb.set_trace()

            output_pc_from_full = get_shape(mat, params_full, def_param, cfg["alpha"])
            # TODO:calculate loss for each part
            output_pc_from_full = output_pc_from_full.reshape(cfg['batch_size'], -1, 3)

            cd_loss_full = compute_cm_loss(output_pc_from_full.detach(), x, mask_part, batch_reduction=None)

        # process data for each batch
        # pdb.set_trace()
        for q in range(cfg['batch_size']):
            cd_loss_full_list_now = cd_loss_full#[q]
            best_cd_loss_full.append(cd_loss_full_list_now)

            re_loss_full_list_now = loss_re_full[q]

            best_re_loss_full.append(re_loss_full_list_now)
            best_re_cd_loss_full.append(cd_loss_full_list_now)

            # pdb.set_trace()
            src_vertices_mats, src_default_params_mesh = get_source_info_mesh(source_labels.squeeze(0), SOURCE_MODEL_INFO)
            src_points, src_labels, src_ids, _, src_vertices, src_faces, src_face_labels = get_source_info_visualization(source_labels.squeeze(0), SOURCE_MODEL_INFO, mesh=True)
            # pdb.set_trace()
            # curr_param = np.expand_dims(params_retrieved, -1)
            curr_param = params_full[q].to("cpu").detach().numpy()
            curr_mat = [x.detach().numpy() for x in src_vertices_mats]
            
            curr_conn_mat = None

            curr_default_param = torch.stack(src_default_params_mesh).detach().numpy().T
            param_def = param_def.to("cpu").squeeze().detach().numpy().T
            # curr_mat:[N, 96]->num_part x [N, 6], curr_param:[1, 96, 1]->[8, 6], curr_default_param:[96, 1]->[6, 8]
            
            output_vertices = []
            output_vertices_def = []
            for i in range(cfg['MAX_NUM_PARTS']):
                output = get_shape_numpy(curr_mat[i], np.reshape(curr_param[i], (1, 6, 1)), param_def[:,i,None], 0.1, connectivity_mat=curr_conn_mat)
                # output_def = get_shape_numpy(curr_mat[i], np.zeros((1, 6, 1)), curr_default_param[:,i, None], 0.1, connectivity_mat=curr_conn_mat)
                output_vertices.append(output)
                # output_vertices_def.append(output_def)

            curr_src_vertices = src_vertices
            # pdb.set_trace()
            print("target model id: {}, avrg ndcg score@40: {:.2f}%".format(int(target_ids[0].item()), sum(ndcg_score)*100/len(ndcg_score)))
            temp_fol = os.path.join(cfg['log_path'], "tmp_cl")
            temp_fol_def = os.path.join(cfg['log_path'], "tmp_cl_def")

            os.makedirs(temp_fol, exist_ok=True)
            # os.makedirs(temp_fol_def, exist_ok=True)
            output_visualization_mesh(output_vertices, curr_src_vertices, src_faces, target_shapes[0].to("cpu").detach().numpy(), src_face_labels, target_labels[0], src_ids, str(int(target_ids[0].item())), temp_fol)
            # output_visualization_mesh(output_vertices_def, curr_src_vertices, src_faces, target_shapes[0].to("cpu").detach().numpy(), src_face_labels, target_labels[0], src_ids, str(int(target_ids[0].item())), temp_fol_def)
    # pdb.set_trace()
    best_re_cd_loss_full = torch.stack(best_re_cd_loss_full)
    best_cd_loss_full = torch.stack(best_cd_loss_full)
    best_re_loss_full = torch.stack(best_re_loss_full)

    print("best full cd loss from retrieval=" + str(torch.mean(best_re_cd_loss_full).cpu().numpy()),
          "best full cd loss=" + str(torch.mean(best_cd_loss_full).cpu().numpy()),
          "best full re loss=" + str(torch.mean(best_re_loss_full).cpu().numpy()),
          "cl acc={:.2f}%".format(elem_all / total_all*100))


if __name__ == '__main__':
    import json

    config_path = "config/config_vis_test.json"#sys.argv[1]
    config = json.load(open(config_path, 'rb'))
    main(config)
    
    
    pass