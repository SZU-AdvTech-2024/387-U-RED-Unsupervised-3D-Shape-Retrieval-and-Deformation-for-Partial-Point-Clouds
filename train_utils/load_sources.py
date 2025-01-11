import numpy as np
import torch
import os
from dataset.dataset_utils import get_model, get_all_selected_models_pickle
from tqdm import tqdm
from collections import defaultdict

def load_sources(cfg):
    ## Get max number of params for the embedding size
    SOURCE_MODEL_INFO = []
    # SOURCE_PART_LATENT_CODES = []

    dist_src = np.load(cfg['src_connectivity'])
    # idx = np.load("/mnt/d/20240308/U-RED/results.pickle", allow_pickle=True)
    
    filename_pickle = os.path.join(cfg['base_dir'], "20240617/generated_datasplits", cfg['category'] + "_" + str(cfg['num_source']) + ".pickle")
    sources, _, _ = get_all_selected_models_pickle(filename_pickle)
    # sources = [sources[i] for i in idx]

    # SOURCE_LATENT_CODES = torch.autograd.Variable(torch.randn((len(sources), cfg["source_latent_dim"]),
    #                                                           dtype=torch.float, device=cfg["device"]),
    #                                               requires_grad=True)


    # src_data_fol = os.path.join(cfg["base_dir"], "20240617/data_aabb_all_models", cfg['category'], "h5")
    src_data_fol = os.path.join("/mnt/d/20240308/U-RED/workspace/0812/data_aabb_all_models/chair/h5")
    for source_model in tqdm(sources):
        src_filename = str(source_model)  # name + _leaves.h5
        box_params, default_param, points, points_mat, vertices, vertices_mat, faces, sem_label = get_model(os.path.join(src_data_fol, src_filename), semantic=True,
                                                        constraint=True, mesh=True)

        curr_source_dict = {}
        curr_source_dict["default_param"] = default_param # 18 -> 6
        curr_source_dict["points"] = points # 2048 x 3
        curr_source_dict["points_mat"] = points_mat # 6144 x 18 -> 6144 x 6
        curr_source_dict["vertices"] = vertices # 7161 x 18
        curr_source_dict["vertices_mat"] = vertices_mat # 21483 x 18
        curr_source_dict["faces"] = faces # 13186 x 3
        curr_source_dict["model_id"] = source_model # 1
        curr_source_dict["sem_label"] = sem_label # 1
        # curr_source_dict["point_semantic"] = point_semantic # 2048 
        # curr_source_dict["point_labels"] = point_labels # 2048
        # curr_source_dict["face_labels"] = face_labels # 13186

        SOURCE_MODEL_INFO.append(curr_source_dict)
        # Get number of parts of the model
        # num_parts = len(np.unique(point_labels))
        # curr_source_dict["num_parts"] = num_parts

        # For source semantics also get a list of unique labels
        # src_semantic = torch.from_numpy(point_semantic)
        # src_semantic = src_semantic.to(cfg['device'])
        # unique_labels = torch.unique(src_semantic)
        # SOURCE_SEMANTICS.append([src_semantic, unique_labels])

        # the implementation is very easy, but I remember there is a paper that use a libiary to implemennt it
        # part_latent_codes = torch.autograd.Variable(
        #     torch.randn((num_parts, cfg['part_latent_dim']), dtype=torch.float, device=cfg['device']),
        #     requires_grad=True)
        # SOURCE_PART_LATENT_CODES.append(part_latent_codes)

    return SOURCE_MODEL_INFO, dist_src


def load_sources_retrieval(cfg):
    ## Get max number of params for the embedding size
    MAX_NUM_PARAMS = -1
    MAX_NUM_PARTS = -1
    SOURCE_MODEL_INFO = []
    SOURCE_SEMANTICS = []
    SOURCE_PART_LATENT_CODES = []

    # get the database idx
    if cfg["complementme"]:
        filename_pickle = os.path.join(cfg['base_dir'], "generated_datasplits_complementme", cfg['middle_name'],
                                       "generated_datasplits_complementme",
                                       cfg['category'] + "_" + str(cfg['num_source']) + ".pickle")
    else:
        filename_pickle = os.path.join(cfg['base_dir'], "generated_datasplits", cfg['middle_name'],
                                       "generated_datasplits",
                                       cfg['category'] + "_" + str(cfg['num_source']) + ".pickle")
    sources, _, _ = get_all_selected_models_pickle(filename_pickle)

    SOURCE_LATENT_CODES = torch.autograd.Variable(torch.randn((len(sources), cfg["source_latent_dim"]),
                                                              dtype=torch.float, device=cfg["device"]),
                                                  requires_grad=True)
    RETRIEVAL_SOURCE_LATENT_CODES = torch.autograd.Variable(
        torch.randn((len(sources), cfg["source_latent_dim"]),
                    dtype=torch.float, device=cfg["device"]), requires_grad=True)
    SOURCE_VARIANCES = torch.autograd.Variable(
        torch.randn((len(sources), cfg["source_latent_dim"]),
                    dtype=torch.float, device=cfg["device"]), requires_grad=True)
    SOURCE_SIGMAS = torch.autograd.Variable(torch.randn((len(sources), 1), dtype=torch.float, device=cfg["device"]),
                                            requires_grad=True)

    src_data_fol = os.path.join(cfg["base_dir"], "data_aabb_constraints_keypoint", cfg['middle_name'],
                                "data_aabb_constraints_keypoint", cfg['category'], "h5")

    for source_model in sources:
        src_filename = str(source_model) + "_leaves.h5"  # name + _leaves.h5
        # box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(os.path.join(data_fol, src_filename), semantic=True)
        if cfg["use_connectivity"]:
            # box params  n x 12
            box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic, \
            constraint_mat, constraint_proj_mat = get_model(os.path.join(src_data_fol, src_filename), semantic=True,
                                                            constraint=True)
        else:
            box_params, orig_ids, default_param, points, point_labels, points_mat, point_semantic = get_model(
                os.path.join(src_data_fol, src_filename), semantic=True)

        curr_source_dict = {}
        curr_source_dict["default_param"] = default_param
        curr_source_dict["points"] = points
        curr_source_dict["point_labels"] = point_labels
        curr_source_dict["points_mat"] = points_mat
        curr_source_dict["point_semantic"] = point_semantic
        curr_source_dict["model_id"] = source_model

        if cfg["use_connectivity"]:
            curr_source_dict["constraint_mat"] = constraint_mat
            curr_source_dict["constraint_proj_mat"] = constraint_proj_mat

        # Get number of parts of the model
        num_parts = len(np.unique(point_labels))
        curr_source_dict["num_parts"] = num_parts

        curr_num_params = default_param.shape[0]
        if (MAX_NUM_PARAMS < curr_num_params):
            MAX_NUM_PARAMS = curr_num_params
            MAX_NUM_PARTS = int(MAX_NUM_PARAMS / 6)  # each part 6 parameters

        SOURCE_MODEL_INFO.append(curr_source_dict)

        # For source semantics also get a list of unique labels
        src_semantic = torch.from_numpy(point_semantic)
        src_semantic = src_semantic.to(cfg['device'])
        unique_labels = torch.unique(src_semantic)
        SOURCE_SEMANTICS.append([src_semantic, unique_labels])

        # the implementation is very easy, but I remember there is a paper that use a libiary to implemennt it
        part_latent_codes = torch.autograd.Variable(
            torch.randn((num_parts, cfg['part_latent_dim']), dtype=torch.float, device=cfg['device']),
            requires_grad=True)
        SOURCE_PART_LATENT_CODES.append(part_latent_codes)

    return SOURCE_MODEL_INFO, SOURCE_SEMANTICS, SOURCE_LATENT_CODES, \
           SOURCE_PART_LATENT_CODES, MAX_NUM_PARAMS, MAX_NUM_PARTS,\
           RETRIEVAL_SOURCE_LATENT_CODES, SOURCE_VARIANCES, SOURCE_SIGMAS
