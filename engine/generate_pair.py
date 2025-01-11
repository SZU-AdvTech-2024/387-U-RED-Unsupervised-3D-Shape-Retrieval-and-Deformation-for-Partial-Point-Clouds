import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import glob
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from joblib import Parallel, delayed

from global_variables import *
from geometry_utils import compute_cm_loss, compute_cm_loss2, compute_emd_loss, read_h5, compute_dcd_loss


class OBJ(Dataset):
    def __init__(self, category='chair', mode='sources', pc=True):
        self.file_path = all_part_path
        self.datasplits_file = datasplits_file
        self.mode = mode
        self.category = category
        
        self.get_list()
        if pc:
            self.pc = self.load_h5()

    def get_list(self):
        with open(self.datasplits_file, 'rb') as f:
            self.datasplits = pickle.load(f)
        self.datasplits['train'] = self.datasplits['train'].astype(str).tolist()
        self.datasplits['test'] = self.datasplits['test'].astype(str).tolist()
        
        self.part_obj_list = glob.glob(os.path.join(self.file_path, '*.h5'))
        self.part_obj_list = [x for x in self.part_obj_list \
            if os.path.basename(os.path.splitext(x)[0]).split('_')[0] in self.datasplits[self.mode]]
        self.part_obj_list_name = [os.path.basename(os.path.splitext(x)[0])[:-7] \
            for x in self.part_obj_list]
    
    def load_h5(self):
        ## TODO: load data speed up???
        print("loading dataset from {} data, num: {}".format(self.mode, len(self.part_obj_list)))
        all_data = []
        for obj in tqdm(self.part_obj_list):
            pc = read_h5(obj)
            all_data.append(pc)
        all_data = torch.stack(all_data)
        return all_data

    def __getitem__(self, idx):
        points = self.pc[idx].cuda()
        return points
    
    def __len__(self):
        return len(self.part_obj_list)
        
if __name__ == "__main__":
    pair_path = "/mnt/d/20240308/U-RED/workspace/0805/pickle"
    pair_path_src = "/mnt/d/20240308/U-RED/workspace/0802/pickle"
    idx_list = np.load("/mnt/d/20240308/U-RED/results.pickle", allow_pickle=True)
    
    os.makedirs(os.path.join(pair_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(pair_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(pair_path, "sources"), exist_ok=True)
    
    dataset_src = OBJ(mode='sources')
    dataset_train = OBJ(mode='train')
    dataset_test = OBJ(mode='test')
    
    # some variables not difined in function
    def get_src_pair(idx, dataset, mode):
        data_path = os.path.join(pair_path, mode, dataset.part_obj_list_name[idx] + '.pickle')

        dcd_loss = []
        cd_s_loss = []
        cd_m_loss = []
        
        for i in range(idx, len(dataset)):
            dcd, cd_s, cd_m = compute_dcd_loss(dataset[i], dataset[idx])
            dcd_loss.append(dcd.cpu().numpy().item())
            cd_s_loss.append(cd_s.cpu().numpy().item())
            cd_m_loss.append(cd_m.cpu().numpy().item())
        
        dict = {'dcd_loss':np.array(dcd_loss), 'cd_s':np.array(cd_s_loss), 'cd_m':np.array(cd_m_loss)}
        with open(data_path, 'wb') as f:
            pickle.dump(dict, f)
            print("saving {}".format(data_path))
            
    def get_data_pair(idx, dataset, mode):
        dcd_loss = []
        cd_s_loss = []
        cd_m_loss = []
        emd_loss = []
        data_path = os.path.join(pair_path, mode, dataset.part_obj_list_name[idx] + '.pickle')
        data_path_src = os.path.join(pair_path_src, mode, dataset.part_obj_list_name[idx] + '.pickle')
        
        # for data in dataset_src:
        #     dcd, cd_s, cd_m = compute_dcd_loss(dataset[idx], data)
        #     dcd_loss.append(dcd.cpu().numpy().item())
        #     cd_s_loss.append(cd_s.cpu().numpy().item())
        #     cd_m_loss.append(cd_m.cpu().numpy().item())
            
        # # TODO:faster emd
        # _, indices = torch.topk(torch.tensor(cd_m_loss), k=20, largest=False)
        # for data in dataset_src[indices]:
        #     emd_loss.append(compute_emd_loss(dataset[idx], data))
        
        # dict = {'dcd_loss':np.array(dcd_loss), 'cd_s':np.array(cd_s_loss), 'cd_m':np.array(cd_m_loss), 'emd_loss_topk':np.array(torch.tensor(emd_loss))}
        # with open(data_path, 'wb') as f:
        #     pickle.dump(dict, f)
        #     print("saving {}".format(data_path))
        
        with open(data_path_src, 'rb') as f:
            dict = pickle.load(f)
            
        new_dict = {'dcd_loss':dict['dcd_loss'], 'cd_s':dict['cd_s'], 'cd_m':dict['cd_m']}
        dcd_loss = [dict['dcd_loss'][i] for i in idx_list]
        cd_s_loss = [dict['cd_s'][i] for i in idx_list]
        cd_m_loss = [dict['cd_m'][i] for i in idx_list]
        new_dict = {'dcd_loss':np.array(dcd_loss), 'cd_s':np.array(cd_s_loss), 'cd_m':np.array(cd_m_loss)}
        
        with open(data_path, 'wb') as f:
            pickle.dump(new_dict, f)
            print("saving {}".format(data_path))
        
            
    num_cores = 48
        
    results = Parallel(n_jobs=num_cores)(delayed(
        get_data_pair)(i, dataset_train, 'train') for i in range(len(dataset_train)))
    
    results = Parallel(n_jobs=num_cores)(delayed(
        get_data_pair)(i, dataset_test, 'test') for i in range(len(dataset_test)))
    
    # results = Parallel(n_jobs=num_cores)(delayed(
    #     get_src_pair)(i, dataset_src, 'sources') for i in range(len(dataset_src)))
    pass

