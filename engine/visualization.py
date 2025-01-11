import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import torch
import pickle
import numpy as np
import shutil
from tqdm import tqdm
from run_preprocessing import collect_leaf_nodes, find_corresponding_meshes

from global_variables import *
from generate_pair import OBJ

class Vis(OBJ):
    def __init__(self, category='chair', mode='sources', path="/mnt/d/20240308/U-RED/workspace/0802", cluster=False):
        super().__init__(category, mode)
        self.src_path = src_obj_path
        self.tgt_path = path#tgt_path
        self.g_structurenet_input_dir = g_structurenet_input_dir
        self.g_partnet_dir = g_partnet_dir
        self.diff_sem = 0
        self.diff_sem_obj = 0
        self.map = sem_map
        self.src_connect = "/mnt/d/20240308/U-RED/workspace/0802/pickle/sources_connect.npy"#file_src_connect
        
        if cluster:
            idx = np.load("/mnt/d/20240308/U-RED/results.pickle", allow_pickle=True)
            self.part_obj_list_name = [self.part_obj_list_name[i] for i in idx]
        
    def get_sources_connect_vis(self):
        if not os.path.exists(self.src_connect):
            N = len(self.part_obj_list)
            sources_matrix_dcd = np.zeros((N, N))
            sources_matrix_cd_s = np.zeros((N, N))
            sources_matrix_cd_m = np.zeros((N, N))
            for i, model_id in tqdm(enumerate(self.part_obj_list_name)):
                file = os.path.join("/mnt/d/20240308/U-RED/workspace/0802/pickle/sources", model_id + ".pickle")
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                sources_matrix_dcd[i, i:] = data['dcd_loss']
                sources_matrix_cd_s[i, i:] = data['cd_s']
                sources_matrix_cd_m[i, i:] = data['cd_m']
            sources_matrix_dcd = sources_matrix_dcd + sources_matrix_dcd.T
            sources_matrix_cd_s = sources_matrix_cd_s + sources_matrix_cd_s.T
            sources_matrix_cd_m = sources_matrix_cd_m + sources_matrix_cd_m.T
            np.save(self.src_connect, np.stack([sources_matrix_dcd, sources_matrix_cd_s, sources_matrix_cd_m], axis=0))
        else:
            data = np.load(self.src_connect)
            sources_matrix_dcd = data[0]
            sources_matrix_cd_s = data[1]
            sources_matrix_cd_m = data[2]
                
        for i, model_id in tqdm(enumerate(self.part_obj_list_name)):
            self.check_sources_connect_single(model_id, sources_matrix_cd_m[i], metric='cd_m')
        print("cd_m diff_sem: {}, diff_sem_obj: {}".format(self.diff_sem, self.diff_sem_obj))
        
        self.diff_sem = 0
        self.diff_sem_obj = 0
        for i, model_id in tqdm(enumerate(self.part_obj_list_name)):
            self.check_sources_connect_single(model_id, sources_matrix_cd_s[i], metric='cd_s')
        print("cd_s diff_sem: {}, diff_sem_obj: {}".format(self.diff_sem, self.diff_sem_obj))
        
        self.diff_sem = 0
        self.diff_sem_obj = 0
        for i, model_id in tqdm(enumerate(self.part_obj_list_name)):
            self.check_sources_connect_single(model_id, sources_matrix_dcd[i], metric='dcd')
        print("dcd diff_sem: {}, diff_sem_obj: {}".format(self.diff_sem, self.diff_sem_obj))
            
    # vis source k nearest in sources
    def check_sources_connect_single(self, name, data, k=10, mode='sources_connect', metric='cd_m'):
        dist, indices = torch.topk(torch.tensor(data), k=k, largest=False)
        indices = indices.tolist()
        
        files = [self.part_obj_list_name[i] + "_mesh.obj" for i in indices]
        dist = dist.tolist()
        os.makedirs(os.path.join(self.tgt_path, "vis", mode, name, metric), exist_ok=True)
        
        src = files[0]
        src_model_id = src.split("_")[0]
        src_map_id = self.check_sem_id(src_model_id, int(src.split("_")[1]))
        vis_diff = False
        
        shutil.copy2(os.path.join(self.src_path, name + "_mesh.obj"), os.path.join(self.tgt_path, "vis", mode, name, name + '_' + src_map_id + "_mesh.obj"))
        for i, file in enumerate(files):
            model_id = file.split("_")[0]
            map_id = self.check_sem_id(model_id, int(file.split("_")[1]))
            if src_map_id != map_id:
                vis_diff = True
                # print("Not the same sem, src model_id: {}, src sem: {}, tgt model_id: {}, tgt sem: {}".\
                #     format(src_model_id, src_map_id, model_id, map_id))
                self.diff_sem += 1
            shutil.copy2(os.path.join(self.src_path, file), os.path.join(self.tgt_path, "vis", mode, name, metric, str(i) + "_" + str(round(dist[i], 3)) + '_' + map_id + '_' + file))
        
        if vis_diff:
            self.diff_sem_obj += 1
            shutil.copytree(os.path.join(self.tgt_path, "vis", mode, name, metric), os.path.join(self.tgt_path, "vis_diff_sem", mode, name, metric), dirs_exist_ok=True) 

            
    # vis source obj all parts
    def vis_sources_sig(self, model_id):
        model_list = [x for x in self.part_obj_list_name if x.split('_')[0] == model_id]
        os.makedirs(os.path.join(self.tgt_path, "vis", self.mode, model_id), exist_ok=True)
        shutil.copy2("/mnt/d/Dataset/PartNet/data_v0/{}/point_sample/sample-points-all-pts-label-10000.ply".format(model_id), \
            os.path.join(self.tgt_path, "vis", self.mode, model_id))
        for part in model_list:
            shutil.copy2(os.path.join(self.src_path, part + "_mesh.obj"), \
                os.path.join(self.tgt_path, "vis", self.mode, model_id))

    def vis_sources(self):
        for model_id in tqdm(self.datasplits['sources']):
            self.vis_sources_sig(model_id)
    
    # vis train/test k nearest in sources
    def get_vis_single(self, file, k, mode, is_print=False):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        dist, indices = torch.topk(torch.tensor(data['cd_m']), k=k, largest=False)
        indices = indices.tolist()
        if is_print:
            is_print(data.keys())
            is_print("dcd:", data['dcd_loss'])
        
        files = [self.part_obj_list_name[i] + "_mesh.obj" for i in indices]
        dist = dist.tolist()
        name = os.path.basename(os.path.splitext(file)[0])
        os.makedirs(os.path.join(self.tgt_path, "vis", mode, name), exist_ok=True)
        
        src = files[0]
        src_model_id = src.split("_")[0]
        src_map_id = self.check_sem_id(src_model_id, int(src.split("_")[1]))
        vis_diff = False
        
        shutil.copy2(os.path.join(self.src_path, name + "_mesh.obj"), os.path.join(self.tgt_path, "vis", mode, name, name + '_' + src_map_id + "_mesh.obj"))
        for i, file in enumerate(files):
            model_id = file.split("_")[0]
            map_id = self.check_sem_id(model_id, int(file.split("_")[1]))
            if src_map_id != map_id:
                vis_diff = True
                print("Not the same sem, src model_id: {}, src sem: {}, tgt model_id: {}, tgt sem: {}".\
                    format(src_model_id, src_map_id, model_id, map_id))
                self.diff_sem += 1
            shutil.copy2(os.path.join(self.src_path, file), os.path.join(self.tgt_path, "vis", mode, name, str(i) + "_" + str(round(dist[i]*1000, 2)) + '_' + map_id + '_' + file))
        
        if vis_diff:
            self.diff_sem_obj += 1
            shutil.copytree(os.path.join(self.tgt_path, "vis", mode, name), os.path.join(self.tgt_path, "vis_diff_sem", mode, name), dirs_exist_ok=True) 

            
    def get_vis(self, k):
        for mode in ['train', 'test']:
            pickle_path = os.path.join(self.tgt_path, "pickle", mode)
            pickle_list = glob.glob(os.path.join(pickle_path, '*.pickle'))
            for pickle in pickle_list:
                self.get_vis_single(pickle, k, mode)
        
    def check_sem_id(self, model_id, idx):
        
        json_file = os.path.join(self.g_structurenet_input_dir,
        '{}_hier'.format(self.category), '{}.json'.format(model_id))
        partnet_json = os.path.join(self.g_partnet_dir, str(model_id),
            'result_after_merging.json')
        
        leaves = collect_leaf_nodes(json_file)
        leaves = find_corresponding_meshes(partnet_json, leaves)
            
        return self.map[leaves[idx]['label'].split('/')[1]] 
    
if __name__ == "__main__":
    
    # dataset = Vis(path="/mnt/d/20240308/U-RED/workspace/0805", cluster=True)
    dataset = Vis()
    # s = time.time()
    # dataset.check_sources_connect(model_id, data)
    # dataset.get_vis_single(file="/mnt/d/20240308/U-RED/workspace/0731/pickle/train/172_7.pickle" ,k = 10 ,mode = 'train', print=True)
    dataset.get_vis(k=10)
    # dataset.get_sources_connect_vis()
    # dataset.get_sources_matrix()

