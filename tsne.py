import imp
from xml import dom
import torch.nn as nn
import torchvision
import numpy as np
import os
import random
from torch.utils.data import Dataset, DataLoader
from MDGAN import Classification, FeatureExtractor, Discriminator
import torch
from sklearn.manifold import TSNE
from dataset import PMDataset, OutputsDataset
import matplotlib.pyplot as plt 
def random_seeding(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic =True
    torch.cuda.manual_seed_all(seed_value)  # gpu vars

seed = 6666
# batch_size = 32
tm_train_data_path = "./cache"
tm_outputs_path = "./prepared_data/prepared_outputs_train(cartoon&sketch).pkl"
bb_outputs_path = "./prepared_data/prepared_outputs_test(photo).pkl"
target_alg_label = {
    "0": "cartoon",
    "1": "sketch",
}
train_info = { 
    "use_GAN": True, 
    "pretrain_ac_epoches": 200,
    "train_feature_epoches": 0,
    "train_meta_epoches" : 120,
    "train_meta_lr": 1e-4,
    "train_FEA_lr": 1e-5,
    "num_workers" : 0,
    "batch_size" : 100,
    "fix_fea_when_meta" : False,
    "domain_num": 2,
    "alpha": 10,
    "meta_stru": [1000, 1000, 3]
}

def make_dataloader(attr_list_dict, train_info=train_info, tm_outputs_path=tm_outputs_path, bb_outputs_path=bb_outputs_path,  phase="train"):
    num_workers = train_info["num_workers"]
    def collate_fn(batch):
        output_list = []
        label_list = []
        multi_label_list = []
        domain_list = []
        domain_num = len(batch[0][0]) if phase == "train" else 1        
        for i in range(domain_num):
            t_output_list = []
            t_label_list = []
            t_multi_label_list = []
            t_domain_list = []
            for data in batch:
                t_output_list.append(data[0][i])
                t_label_list.append(data[1][i])
                t_multi_label_list.append(data[2][i])
                t_domain_list.append(data[3][i])
            output_list.append(torch.stack(t_output_list, dim=0))
            label_list.append(torch.stack(t_label_list, dim=0))
            multi_label_list.append(torch.stack(t_multi_label_list, dim=0))
            domain_list.append(t_domain_list)
        return torch.cat(output_list, dim=0).cuda(), torch.cat(label_list, dim=0).cuda(), torch.cat(multi_label_list, dim=0).cuda(), domain_list 

    if phase == "train":
        dataset = OutputsDataset(tm_outputs_path, attr_list_dict, phase, domain_num=train_info["domain_num"])
    else:
        dataset = OutputsDataset(bb_outputs_path, attr_list_dict, phase, domain_num=1)

    dataloader = DataLoader(
        dataset,
        batch_size=train_info["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True
    )
    return dataloader


def main():
    target_alg_label = {
        # domain
        "0": "cartoon",
        "1": "sketch",
        "2": "photo_test"
    }
    seed=6666
    random_seeding(seed)
    model = FeatureExtractor()
    attr_list_dict = {'bn': ['normal', 'none']}
    train_dataloader = make_dataloader(attr_list_dict=attr_list_dict, phase="train")
    test_dataloader = make_dataloader(attr_list_dict=attr_list_dict, phase="test")
    
    epoch_list = [20]
    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(5)


    for iii, epoch in enumerate(epoch_list):
        state_dict = torch.load("./ckpt/train_G_epoch_{}.pth".format(epoch))

        model.load_state_dict(state_dict['net'])
        model = model.cuda()
        all_fea = []
        all_lbl = []
        for iter, data in enumerate(train_dataloader):
            
            tm_outputs, model_label, model_multi_label, domain_list = data
            tm_outputs = torch.flatten(tm_outputs, start_dim=1)

            fea, _ = model(tm_outputs)
            # fea, _, _ = model(tm_outputs)
            
            train_fea = fea.detach().cpu().numpy()
            
            # domain 信息
            domain_lbl1 = [0 for _ in range(100)]
            domain_lbl2 = [1 for _ in range(100)]
            domain_lbl1.extend(domain_lbl2)
            train_lbl = np.array(domain_lbl1)
            
            # 类信息
            # train_lbl = model_label.long().cpu().numpy()
            # print(train_lbl)
            # middle_outputs, outputs = model(test_fea)
            # middle_outputs = torch.mean(middle_outputs, dim=1)
            break
        
        for iter, data in enumerate(test_dataloader):
            tm_outputs, model_label, model_multi_label, domain_list = data
            tm_outputs = torch.flatten(tm_outputs, start_dim=1)

            fea, _ = model(tm_outputs)
            # fea, _, _ = model(tm_outputs)

            test_fea = fea.detach().cpu().numpy()
            
            # domain信息
            domain_lbl1 = [2 for _ in range(100)]
            test_lbl = np.array(domain_lbl1)
            
            # 类信息
            # test_lbl = model_label.long().cpu().numpy()
            
            # middle_outputs, outputs = model(test_fea)
            # middle_outputs = torch.mean(middle_outputs, dim=1)
            break
        
        all_fea = np.concatenate((train_fea, test_fea), axis=0)
        all_lbl = np.concatenate((train_lbl, test_lbl), axis=0).squeeze()
        print(all_lbl)
        # print("_____________________",all_lbl.shape)
        random_seeding(seed)
        fea_embedded = TSNE(n_components=2, learning_rate=100).fit_transform(all_fea) 

        lbl_num = len(np.unique(all_lbl))
        lbl_fea_list = [[] for i in range(lbl_num)]
        for i in range(len(all_lbl)):
            lbl_fea_list[all_lbl[i]].append(fea_embedded[i])
        print(iii)
        number = int("11"+str(iii+1))
        print(number)
        ax = plt.subplot(number)
        ax.set_title("epoch_{}".format(epoch))
        for i in range(lbl_num):
            lbl_fea_list[i] = np.array(lbl_fea_list[i])
            print(type(lbl_fea_list[i]))
            print(lbl_fea_list)
            plt.scatter(lbl_fea_list[i][:,0], lbl_fea_list[i][:,1], label=target_alg_label[str(i)])

        # plt.scatter(fea_embedded[:,0], fea_embedded[:,1], c=all_lbl)
        plt.legend(loc='upper right', fontsize=15, handlelength=0, labelspacing=0.2, handletextpad=0.4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # ax.set_title('Epoch #{}'.format(epoch),fontsize= 22)
        ax.set_title('Epoch #20',fontsize= 22)
        
    plt.savefig("./tsne_show/epoch{}.png".format(int(epoch)))
    # sv_path = './tsne_show/'
    # plt.savefig(f'{sv_path}/PACS_epoch10.pdf')

main()