'''
    file name: main.py
    author: rql
    create time: 3/5/2022
    modify time: 3/7/2022 19:37
'''

from xml import dom
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import os
import random
import torch
import time
import math
import pickle
import pdb
import os
import sys
# PM: Pretrained Model
class PMDataset(Dataset):
    def __init__(self, path, phase="train"):
        modelzoo_name_list = ["modelzoo-EMNIST"]
        # self.tm_category_list = ["modelzoo-EMNIST", "modelzoo-DIDA", "modelzoo-USPS"]
        self.tm_category_list = ["modelzoo-DIDA", "modelzoo-USPS"]
        self.channel_num = []
        # self.tm_category_list = os.listdir(path)
        if phase == "train":
            model_num = 100
            print("Loading pre-train model...")
        self.tm_save_dict_list = []
        loaded_model_num = 0
        for i in range(len(self.tm_category_list)):
            if self.tm_category_list[i] == "modelzoo-EMNIST":
                self.channel_num.extend([1] * model_num)
            else:
                self.channel_num.extend([3] * model_num)
            
            cur_loaded_model_num = 0
            self.tm_name_list = os.listdir(os.path.join(path, self.tm_category_list[i]))
            for j in range(len(self.tm_name_list)):
                if loaded_model_num % 50 == 0:
                    print("already loaded: {}".format(loaded_model_num))
                if cur_loaded_model_num == model_num:
                    break
                if os.path.isfile(os.path.join(path, self.tm_category_list[i], self.tm_name_list[j], "final.pth.tar")):
                    save_dict = torch.load(os.path.join(path, self.tm_category_list[i], self.tm_name_list[j], "final.pth.tar"), map_location=torch.device("cpu")) 
                    # print(save_dict.keys())
                    stat_path = os.path.join(path, self.tm_category_list[i], self.tm_name_list[j], "stat.pkl")
                    # print(stat_path)
                    with open(stat_path, "rb") as f:
                        stat_dict = pickle.load(f)
                        if stat_dict["acc"] < 90:
                            print("acc: ", stat_dict["acc"])
                            continue
                    value = save_dict["state_dict"]["conv1.weight"][0][0][0][0].data.cpu().numpy()
                    if math.isnan(value):
                        print(value)
                        print("model contains nan")
                        continue
                    loaded_model_num += 1
                    cur_loaded_model_num += 1
                    self.tm_save_dict_list.append(save_dict)
        # elif phase == "test":
        #     model_num = 100
        #     print("Loading test black-box model...")
        
    def get_len(self):
        return len(self.tm_save_dict_list)

    def __len__(self) -> int:
        return len(self.tm_save_dict_list)

    def __getitem__(self, index: int):
        state_dict = self.tm_save_dict_list[index]["state_dict"]
        control = self.tm_save_dict_list[index]["control"]
        return state_dict, control

class OutputsDataset(Dataset):
    def __init__(self, path, attr_list_dict, phase="train", label_form="multi", domain_num=2):
        # print(os.listdir())
        # print(os.getcwd())
        self.label_form = label_form
        self.attr_list_dict = attr_list_dict
        self.output_list = []
        self.domain_num = domain_num
        self.output_domain_list = [[] for i in range(domain_num)]
        self.control_domain_list = [[] for i in range(domain_num)]

        self.control_list = []
        print("Loading prepared outputs...")
        with open(path, "rb") as f:
            output_control_list = pickle.load(f)["output"]
            # print("output_control_list", output_control_list)
            per_domain_num = int(len(output_control_list) / domain_num)
            num = 0
            for i in range(len(output_control_list)):
                self.output_list.append(output_control_list[i]["value"].cpu())
                self.control_list.append(output_control_list[i]["control"])
            
            
            
            self.output_list = np.array(self.output_list)
            self.control_list = np.array(self.control_list)

            for i in range(domain_num):
                self.output_domain_list[i] = self.output_list[i * per_domain_num : (i+1) * per_domain_num]
                self.control_domain_list[i] = self.control_list[i * per_domain_num : (i+1) * per_domain_num]

    def find_index(self, obj_attr, attr_list):
        for i, attr in enumerate(attr_list):
            if attr == obj_attr:
                return i
        raise Exception("object attribute doesn't exist")

    def process_control(self, control):
        attr_list_dict_keys = sorted(self.attr_list_dict.keys())
        index_list = []
        opt_attr_list = ["optimiser", "batch_size"]
        net_attr_list = ["drop", "n_fc", "n_conv", "ks", "act", "pool", "bn"]
        for key in attr_list_dict_keys:
            if key in opt_attr_list:
                value = control["opt"][key]
            elif key in net_attr_list:
                value = control["net"][key]
            # elif key == "subset":
            #     value = control["data"][key]
            index = self.find_index(value, self.attr_list_dict[key])
            index_list.append(index)

        return index_list

    def process_gt_control2multi_label(self, control):
        def index2one_hot(index, attr):
            max_len = len(self.attr_list_dict[attr])
            onehot_vec = np.zeros((max_len,))
            onehot_vec[index] = 1
            return onehot_vec
        
        # attr_list_dict_keys = sorted(['act', 'drop', 'pool', 'ks', 'n_conv', 'n_fc', 'optimiser', 'batch_size'])
        attr_list_dict_keys = sorted(self.attr_list_dict.keys())

        one_hot_label_list = []
        opt_attr_list = ["optimiser", "batch_size"]
        net_attr_list = ["drop", "n_fc", "n_conv", "ks", "act", "pool", "bn"]
        for key in attr_list_dict_keys:
            if key in opt_attr_list:
                value = control["opt"][key]
            elif key in net_attr_list:
         

                value = control["net"][key]
               
            index = self.find_index(value, self.attr_list_dict[key])
            one_hot_label = index2one_hot(index, key)
            one_hot_label_list.extend(one_hot_label)
        return np.array(one_hot_label_list, dtype=np.float32)

    def __getitem__(self, index):
        output, model_multi_label, model_label, domain = [], [], [], []
        for i in range(self.domain_num):
            output.append(self.output_domain_list[i][index])
            
            tmp_control = self.control_domain_list[i][index]
            domain.append(tmp_control["data"]["domain"])
            tmp_model_multi_label = self.process_gt_control2multi_label(tmp_control)
            tmp_model_label = self.process_control(tmp_control)
            tmp_model_label = torch.Tensor(tmp_model_label).cuda()
            tmp_model_multi_label = torch.Tensor(tmp_model_multi_label).cuda()
            model_multi_label.append(tmp_model_multi_label)
            model_label.append(tmp_model_label)

        return output, model_label, model_multi_label, domain

    def __len__(self) -> int:
        return len(self.output_domain_list[0])
             
if __name__ == "__main__":
    dataset = OutputsDataset('./prepared_data/prepared_outputs_train(cartoon&sketch).pkl', 'train')
    output, control = dataset.__getitem__(5)
    print(output)
    print(control)
    print(output.size())