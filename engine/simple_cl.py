import os
import numpy as np
import shutil
import argparse
import tqdm
import time
import pickle
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.cluster import SpectralClustering

from global_variables import *
from generate_pair import OBJ


class SimpleCL(OBJ):
    def __init__(self, num_clusters, sigma):
        super().__init__(pc=False)
        self.num_clusters = num_clusters
        self.sigma = sigma
        self.file_temp = "workspace/0721_script/results_{}_{}_iter{}.npy"
        self.src_path = src_obj_path
        self.tgt_path = tgt_path
        self.dist_m = np.load(file_src_connect)
        self.save_samples = None
        self.final_num = 0
        self.iter_name = self.part_obj_list_name
        
        
    def cluster_init(self):
        pass


    def cal_similarity(self, file, num_clusters, sigma):
        sim_matrix = np.exp(-self.dist_m ** 2 / (2.0 * sigma ** 2))

        s = time.time()
        clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
        labels = clustering.fit_predict(sim_matrix)
        np.save(file, labels)
        print("Time:", time.time() - s)
    
        return labels
    
    
    def get_similarities(self, num_clusters, sigma, iter=0, idx_list=None):
        print("Clustering iter:{}".format(iter))
        print("Number of clusters:{} Sigma:{} ".format(num_clusters, sigma))
        
        file = self.file_temp.format(num_clusters[:iter+1], sigma[:iter+1], iter)
        if os.path.exists(file):
            labels = np.load(file)
        else:
            labels = self.cal_similarity(file, num_clusters[iter], sigma[iter])
            
        self.print_results(labels)
    
    
    def print_results(self, labels):
        label_counts = Counter(labels)
        self.sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        total = 0
        for i, item in enumerate(self.sorted_counts):
            if item[1]>1:
                total = total + item[1]
                print("num:", i, "label:", item[0], "number:", item[1])
        print("total 1 class:", len(self.dist_m)-total)

        self.cate_samples = defaultdict(list)
        for index, label in enumerate(labels):
            self.cate_samples[label].append(self.iter_name[index])
            
        
    def check_similarity(self, id1, id2, k=10):
        idx1 = self.iter_name.index(id1)
        idx2 = self.iter_name.index(id2)
        dist1 = self.dist_m[idx1]
        dist2 = self.dist_m[idx2]  
        topk_indices1 = np.argpartition(dist1, k)[:k]
        topk_indices2 = np.argpartition(dist2, k)[:k]
        return idx1 in topk_indices2 and idx2 in topk_indices1
    
    def check_similarity_batch(self, batch, id, k=10):
        for i in range(len(batch)):
            if not self.check_similarity(batch[i], id, k):
                return False
        return True
        
    def clear_cluster(self, label):
        obj_list = self.cate_samples[label]
        bool_matrix = np.full((len(obj_list), len(obj_list)), True)
        
        for i in range(len(obj_list)):
            for j in range(i+1, len(obj_list)):
                flag = self.check_similarity(obj_list[i], obj_list[j])
                bool_matrix[i, j] = flag

        a = np.logical_and(bool_matrix, bool_matrix.T)

        while not a.all():
            idx = np.argmin(a.sum(0))
            elem = obj_list.pop(np.argmin(a.sum(0)))
            self.cate_samples[-1].append(elem)
            a = np.delete(np.delete(a, idx, 0), idx, 1)
            
    def post_process(self):
        pass
    
    
    def refine_cluster(self):
        for iter in range(len(self.sigma)):
            self.get_similarities(self.num_clusters, self.sigma, iter=iter)
            
            for item in tqdm.tqdm(self.sorted_counts):
                if item[1]<50 and item[1]>1:
                    self.clear_cluster(item[0])
                    
            self.rest_samples = []
            if self.save_samples is None:
                self.save_samples = defaultdict(list)
                
            for label in self.cate_samples:
                if len(self.cate_samples[label]) == 1 or len(self.cate_samples[label]) >= 50:    
                    self.rest_samples += self.cate_samples[label]
                else:
                    self.save_samples[self.final_num] = self.cate_samples[label]
                    self.final_num += 1
                    
            # self.vis_results(iter)
                    
            rest_idx = [self.iter_name.index(x) for x in self.rest_samples]
            self.iter_name = [self.iter_name[i] for i in rest_idx]
            self.dist_m = self.dist_m[rest_idx, :][:, rest_idx]
            
            print("{} clusters have been saved".format(self.final_num))
            print("Total number of rest samples:", len(rest_idx))
            print("*"*50)

        self.iter_name = self.part_obj_list_name
        self.dist_m = np.load(file_src_connect)
        self.post_process()
        # self.search_label()
        
        for i in range(len(self.rest_samples)):
            self.save_samples[self.final_num] = [self.rest_samples[i]]
            self.final_num += 1
        select = []
        for label in self.save_samples:
            select.append(random.choice(self.save_samples[label]))
        index = [self.part_obj_list_name.index(x) for x in select]
        with open("results.pickle", 'wb') as f:
            pickle.dump(index, f)
            
        self.vis_results(iter, True)
        print("Final number of clusters:", len(self.save_samples))
        
    def search_label(self):
        for label in self.save_samples:
            if len(self.save_samples[label]) < 4:
                continue
            temp = []
            for id in self.rest_samples:
                if self.check_similarity_batch(self.save_samples[label] + temp, id):
                    self.save_samples[label].append(id)
                    temp.append(id)
                    self.rest_samples.remove(id)
                    print("add {} to cluster {}".format(id, label))
        pass
        
    def merge_label(self):
        pass
    
    def add_label(self):
        pass
    
    def delete_label(self):
        pass


    def vis_results(self, iter, final=False):
        for i in range(len(self.save_samples)):
            if final:
                file_path = os.path.join(self.tgt_path, "vis", "cluster_" + str(self.num_clusters) + "_" + str(self.sigma), str(i))
            else:
                file_path = os.path.join(self.tgt_path, "vis", "cluster_" + str(self.num_clusters[:iter+1]) + "_" + str(self.sigma[:iter+1]) + "_" + str(iter), str(i))
            os.makedirs(file_path, exist_ok=True)
            for model_id in self.save_samples[i]:
                shutil.copy2(os.path.join(self.src_path, model_id + "_mesh.obj"), file_path)
        
        if final:
            plt.figure(figsize=(6, 4))
            labels = [-1 for i in range(len(self.part_obj_list_name))]
            for label, obj_list in self.save_samples.items():
                for obj in obj_list:
                    labels[self.part_obj_list_name.index(obj)] = label
            random.shuffle(labels)
            plt.scatter(range(len(labels)), labels, c=labels, cmap='viridis')
            plt.xlabel('Object index')
            plt.ylabel('Cluster label')
            plt.title('Spectral Clustering Result')
            plt.show()
                
    
    def __str__(self):
        return f"SimpleCL: num_clusters={self.num_clusters}, sigma={self.sigma}"
    
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num", type=int, nargs='+', default=[3000 ,2000, 2000, 2000, 1500, 1000])
    argparser.add_argument("--sigma", type=float, nargs='+', default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    args = argparser.parse_args()
    
    Pro = SimpleCL(args.num, args.sigma)
    Pro.refine_cluster()
    
    pass
    