import numpy as np
import torchvision.transforms as transforms
import os
import glob
import json
from torch.utils.data import Dataset

from dataset.dataset_utils import *
from dataset.gen_occ_point import generate_occ_point_ball, generate_occ_point_slice, generate_occ_point_random, generate_occ_point_part
from train_utils.random_rot import rotation_matrix_3d


class partnet_dataset(Dataset):
    def __init__(self, config):
        
        filename = os.path.join(config["base_dir"], "20240617/generated_datasplits",  config["category"] + '_' + str(config["num_source"]) + '_' + config["mode"] + ".h5")

        self.dis_mat_path = os.path.join(config["base_dir"], "dis_mat", config["category"])

        #all_target_points: 1207 x 2048 x 3, (all_target_labels: 1207 x 2048), all_target_semantics: 1207 x 2048, all_target_model_id: 1207
        all_target_points, all_target_labels, all_target_semantics, all_target_model_id = load_h5(filename)
        self.target_points = all_target_points
        self.target_labels = all_target_labels
        self.target_semantics = all_target_semantics
        self.target_ids = all_target_model_id

        self.n_samples = all_target_points.shape[0]
        self.random_rot = config["random_rot"]
        
        print("Number of targets: " + str(self.n_samples))
        
        # load sources
        with open(datasplits_file, 'rb') as f:
            self.datasplits = pickle.load(f)
        # self.part_obj_list = glob.glob(os.path.join(all_part_path, '*.h5'))
        self.part_obj_list = glob.glob(os.path.join("/mnt/d/20240308/U-RED/workspace/0812/data_aabb_all_models/chair/h5", '*.h5'))
        self.part_obj_list = [x for x in self.part_obj_list \
            if os.path.basename(os.path.splitext(x)[0]).split('_')[0] in self.datasplits['sources']]
        self.part_obj_list_name = [os.path.basename(os.path.splitext(x)[0])[:-7] \
            for x in self.part_obj_list]

    def __getitem__(self, index):
        # occlusion handling
        points = self.target_points[index]  # size 2048 x 3
        # note that ids and labels  are only used in visualization and retrieval
        ids = self.target_ids[index]  # 1
        labels = self.target_labels[index]  # 2048   view label, from which view ?
        semantics = self.target_semantics[index]  # 2048  part segementation
        ##  randomly generate occ points
        choose_one_occ = 0# np.random.rand()
        if choose_one_occ < 0.3:
            points_occ, points_occ_mask = generate_occ_point_ball(points, ids, save_pth=self.dis_mat_path)
        elif choose_one_occ < 0.6:
            points_occ, points_occ_mask = generate_occ_point_random(points)
        elif choose_one_occ < 0.9:
            points_occ, points_occ_mask = generate_occ_point_slice(points)
        else:
            points_occ, points_occ_mask = generate_occ_point_part(points, semantics)
        # focalization
        ori_point_occ = points_occ
        points_occ_mean = np.mean(points_occ, axis=0, keepdims=True)
        points_occ = points_occ - points_occ_mean
        #  numpy check if there is none or inf
        '''
        if ((True in np.isnan(points_occ)) or (True in np.isnan(points_occ_mask))
                or (True in np.isinf(points_occ)) or (True in np.isinf(points_occ_mask))):
            print(str(ids), str(index), str(choose_one_occ))
            return self.__getitem__((index + 1) % self.__len__())
        if (True in np.isnan(points)):
            print(str(1024), str(ids), str(index), str(choose_one_occ))
            return self.__getitem__((index + 1) % self.__len__())
        '''
        if self.random_rot:
            angle = np.random.uniform(low=-10.0, high=10.0, size=6)
            #R_full = rotation_matrix_3d(angle[0], angle[1], angle[2])[:3, :3]
            R_partial = rotation_matrix_3d(angle[3], angle[4], angle[5])[:3, :3]
            #points = (np.matmul(R_full, points.T)).T
            points_occ = (np.matmul(R_partial, points_occ.T)).T
            
        # This will slow down the training process a lot
        # img = "/mnt/d/Dataset/PartNet/data_v0/{}/parts_render/0.png".format(int(ids))
        # image = Image.open(img)
        # transform = transforms.ToTensor()
        # img = transform(image)
        
        return points, ids, labels, semantics, points_occ, points_occ_mask, ori_point_occ#, img

    def __len__(self):
        return self.n_samples

if __name__ == '__main__':
    config = json.load(open("config/config_train_test.json", 'rb'))
    Data = partnet_dataset(config)
    # print(Data[0])
    pass