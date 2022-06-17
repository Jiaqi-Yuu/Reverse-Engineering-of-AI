'''
    file name: main.py
    create time: 3/26/2022
    modify time: 3/26/2022 19:37
'''
# from audioop import avg
# from calendar import c
# from curses import meta
# from email import generator
# import imp
# import modulefinder
import random
# from re import A
import torch
import torch.nn as nn
import torch.nn.functional as F
from MDGAN import Classification, FeatureExtractor, Discriminator
from dataset import PMDataset, OutputsDataset
from torch.utils.data import  DataLoader
from PACS import mnet
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torchvision
from torchvision import transforms
import pdb
import math
import cv2
from torchvision import transforms
import torchvision.models as models
# torch.multiprocessing.set_sharing_strategy("file_system")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,3,2,0"

torch.backends.cudnn.enabled = False

def random_seeding(seed_value):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic =True
    torch.cuda.manual_seed_all(seed_value)  # gpu vars

tm_outputs_path = "./prepared_data/prepared_outputs_train(cartoon&sketch).pkl"
bb_outputs_path = "./prepared_data/prepared_outputs_test(photo).pkl"
train_info = { 
    "use_GAN": True, 
    "train_meta_epoches" : 50,
    "train_meta_lr": 1e-4,
    "train_FEA_lr": 5e-5,
    "num_workers" : 0,
    "batch_size" : 100,
    "fix_fea_when_meta" : False,
    "domain_num": 2,
    "alpha": 15,
}
attr_list_dict_tot = dict(
    # act=['relu', 'elu', 'prelu', 'tanh'],
    # drop=['normal', 'none'],
    # pool=['max_2', 'none'],
    # ks=[3, 5],
    # n_conv=[2, 3, 4],
    # n_fc=[2, 3, 4],
    # optimiser=['SGD', 'ADAM', 'RMSprop'],
    # batch_size=[32, 64, 128],
    bn=['normal', 'none'],
)
random_seed = 19033
write_file_name = "ours_id1_seed:{}".format(random_seed)

def make_dataloader(train_info=train_info, tm_outputs_path=tm_outputs_path, bb_outputs_path=bb_outputs_path, attr_list_dict=attr_list_dict_tot, phase="train"):
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
        shuffle=True if phase == "train" else False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True
    )
    return dataloader


def train(write_file_name, random_seed, train_info, attr_list_dict_tot):
    def compute_loss(prob_list, gt_list, criterion, attr_list_dict_keys, loss_each):
        loss = []
        for j in range(len(prob_list)):
            gt = gt_list[:, j]
            prob = prob_list[j] #tensor
            gt = gt.long().cuda()
            tmp_loss = criterion(prob, gt)
            print(tmp_loss)
            loss_each[attr_list_dict_keys[j]] = float(tmp_loss.data.cpu().numpy())
            loss.append(tmp_loss.view(1))
        return loss

    def train_each_attr(attr_list_dict, train_dataloader):
        attr_list_dict_keys = sorted(attr_list_dict.keys())

        FEA = FeatureExtractor().cuda()
        model = Classification(attr_list_dict).cuda()
        if train_info["fix_fea_when_meta"] == True:
            meta_optim = torch.optim.Adam(model.meta.parameters(), lr=train_info["train_meta_lr"])
        else:
            meta_optim = torch.optim.Adam([{'params': model.parameters(), 'lr': train_info["train_meta_lr"]}, {'params': FEA.parameters(), 'lr': train_info["train_FEA_lr"]}])
        
        
        D_list =[]
        for i in range(train_info['domain_num']):
            D_list.append(Discriminator().cuda())
        CEloss = nn.CrossEntropyLoss()
        
        max_epoches = train_info["train_meta_epoches"]
        keys = list(attr_list_dict.keys())
        tot_num = len(train_dataloader) * train_dataloader.batch_size
        test_max_acc = {}
        test_max_avg_acc = 0
        for epoch in range(1, max_epoches + 1):
            test_acc_dict, test_avg_acc = test(attr_list_dict=attr_list_dict, FEA=FEA, meta=model)    
            if test_avg_acc > test_max_avg_acc:
                test_max_avg_acc = test_avg_acc
            model.train()
            FEA.train()
            act_model_label = []
            act_preds = []
            acc_dict = {}
            for attr_keys in attr_list_dict.keys():
                acc_dict[attr_keys] = 0.0
                if attr_keys not in test_max_acc:
                    test_max_acc[attr_keys] = 0
                if test_acc_dict[attr_keys] > test_max_acc[attr_keys]:
                    test_max_acc[attr_keys] = test_acc_dict[attr_keys]
            avg_acc = 0.0
            for iter, data in enumerate(train_dataloader):
                loss_each = {}
                for attr_keys in attr_list_dict_keys:
                    loss_each[attr_keys] = 0
                tm_outputs, model_label, model_multi_label, domain_list = data
                tm_outputs = torch.flatten(tm_outputs, start_dim=1)        
                fea, _ = FEA(tm_outputs)  
                if train_info["use_GAN"]:
                    gan_train(FEA, model, model_label, D_list, fea, train_info["domain_num"], epoch, max_epoches, acc_dict, attr_list_dict_keys)
                else:
                    act_model_label, act_preds, acc_dict = train_meta(model, fea, model_label, CEloss, attr_list_dict_keys, loss_each, acc_dict, act_model_label, act_preds,epoch, max_epoches, meta_optim, keys)

            for attr_keys, v in acc_dict.items():
                acc_dict[attr_keys] /= (tot_num) * train_info["domain_num"]
                avg_acc += acc_dict[attr_keys]
            avg_acc /= len(attr_list_dict.keys())

               
           
            torch.save({
                    'net': model.state_dict(),
                    "optim": meta_optim
                },
                "./ckpt/meta/train_meta_epoch_{}.pth".format(epoch)
            )
        return test_max_acc

    def train():
        final_avg_acc = 0
        result_file_name = write_file_name
        result_file_path = os.path.join("./result/", result_file_name)
        with open(result_file_path, "a") as f:
            print("ALPHA={}, random_seed={}, train_meta_lr={}, train_FEA_lr={}:".format(train_info["alpha"], random_seed, train_info["train_meta_lr"], train_info["train_FEA_lr"]), file=f)
            for k, v in attr_list_dict_tot.items():
                attr_list_dict = {}
                attr_list_dict[k] = v
                train_dataloader = make_dataloader(attr_list_dict=attr_list_dict, phase="train")
                test_max_acc = train_each_attr(attr_list_dict, train_dataloader)
                for attr_keys in attr_list_dict.keys():
                    print("test.{}.max_accuracy : {}% (Random {}%)".format(attr_keys, test_max_acc[attr_keys] * 100, 100 / len(attr_list_dict[attr_keys])))       
                    print("\ttest.{}.max_accuracy : {}% (Random {}%)".format(attr_keys, test_max_acc[attr_keys] * 100, 100 / len(attr_list_dict[attr_keys])), file=f)
                final_avg_acc += test_max_acc[attr_keys] * 100
            print("test.avg.accurage : {}%".format(final_avg_acc / len(attr_list_dict_tot)), file=f)


    def train_meta(model, fea_max_mi, model_label, CEloss, attr_list_dict_keys, loss_each, acc_dict, act_model_label, act_preds,epoch, max_epoches, meta_optim, keys):
        prob_list = model(fea_max_mi)
        loss_ = compute_loss(prob_list, model_label, CEloss, attr_list_dict_keys, loss_each)
        loss = torch.cat(loss_).sum()
        loss.backward()
        for i, prob in enumerate(prob_list):
            label = model_label[:, i]
            prob_index = torch.argmax(prob, dim=1)
            acc_dict[keys[i]] += float(torch.sum(prob_index == label).cpu().numpy())
            act_model_label.extend(label.cpu())
            act_preds.extend(prob_index.cpu())

        print("Train meta Epoch: [{}/{}] \t CE:{:.6f}\t mmd_loss:{:.6f}\t".format(epoch, max_epoches, loss.item(), 0))
        meta_optim.step()
        meta_optim.zero_grad()
        return act_model_label, act_preds, acc_dict


    def gan_train(FEA, model, model_label, D_list, fea, domain_num, epoch, max_epoch, acc_dict, attr_list_dict_keys):
        '''
            Our GAN consists of domain_num's Discriminators and one Generator,
            When training D, we treat data of domain_i as real samples and 
            others as fake samples, and vice versa for training G. What's more,
            we also feed data of all domains into meta classifier to preserve
            label informain.
        '''
        loss_each = {}
        for attr_keys in attr_list_dict_keys:
            loss_each[attr_keys] = 0
        bce = nn.BCELoss()
        ce = nn.CrossEntropyLoss()
        FEA_optim = torch.optim.Adam([{'params': model.parameters(), 'lr': train_info["train_meta_lr"]}, {'params': FEA.parameters(), 'lr': train_info["train_FEA_lr"]}])
        meta_only_optim = torch.optim.Adam([{'params': model.parameters(), 'lr': train_info["train_meta_lr"]}])
        per = fea.size(0) // domain_num 

        train_d_all_loss = 0 # tot_d_loss + meta_loss
        tot_d_loss = 0 # summation of d{i}_loss (i=1:domain_num)
        for i in range(domain_num):
            fea_source = fea[i * per: (i+1) * per]
            D = D_list[i]
            D_optim = torch.optim.Adam(D.parameters(), lr=1e-4)
            D_optim.zero_grad()
            tot_domain_d_loss = 0
            for j in range(domain_num):
                if j != i:
                    fea_target = fea[j * per: (j+1) * per]

                    fea_source_detach = fea_source.detach()
                    fea_target_detach = fea_target.detach()
                    prob_s_detach = D(fea_source_detach)
                    prob_t_detach = D(fea_target_detach)
                    domain_d_loss = bce(torch.cat([prob_s_detach, prob_t_detach]), torch.cat([torch.ones_like(prob_s_detach), torch.zeros_like(prob_t_detach)]))
                    tot_domain_d_loss += domain_d_loss


                    
            tot_d_loss += tot_domain_d_loss

        train_d_all_loss = tot_d_loss
        train_d_all_loss.backward()
        D_optim.step()

        tot_g_loss = 0 # save 
        train_g_all_loss = 0

        # only input <target_domain> into D to misclassify them.
        for i in range(domain_num):
            fea_source = fea[i * per: (i+1) * per]
            D = D_list[i]
            FEA_optim.zero_grad()
            tot_domain_g_loss = 0
            for j in range(domain_num):
                if j != i:
                    # compute g{i}_loss
                    fea_target = fea[j * per: (j+1) * per]
                    prob_t = D(fea_target)
                    domain_g_loss = bce(prob_t, torch.ones_like(prob_t))
                    tot_domain_g_loss += domain_g_loss
            # compute summation of g{i}_loss (i=1:domain_num)
            tot_g_loss += tot_domain_g_loss

        # compute meta_loss during phase G
        prob_list = model(fea)
        
        loss_ = compute_loss(prob_list, model_label, ce, attr_list_dict_keys, loss_each)
        meta_loss = torch.cat(loss_).sum()
        
        train_g_all_loss = train_info["alpha"]*tot_g_loss + meta_loss

        train_g_all_loss.backward()
        FEA_optim.step()
        print("Training GAN Epoch: [{}/{}] \t D_loss:{}  G_loss:{}".format(epoch, max_epoch, train_g_all_loss.item(), train_d_all_loss.item()))
    
    train()
    



def test(attr_list_dict, FEA, meta, train_info=train_info):
    test_dataloader = make_dataloader(attr_list_dict=attr_list_dict, phase="test")
    tot_num = len(test_dataloader) * test_dataloader.batch_size
    print("tot_num:", tot_num)
    def compute_loss(prob_list, gt_list, criterion, attr_list_dict_keys, loss_each):
        loss = []
        for j in range(len(prob_list)):
            gt = gt_list[:, j]
            prob = prob_list[j] #tensor
            gt = gt.long().cuda()
            
            tmp_loss = criterion(prob, gt)
            loss_each[attr_list_dict_keys[j]] = tmp_loss.data.cpu().numpy()
            loss.append(tmp_loss.view(1))
        return loss
    
    def compute_acc(acc_dict, avg_acc, prob_list, model_label, act_model_label, act_preds):
        keys = list(attr_list_dict.keys())
        for i, prob in enumerate(prob_list):
            label = model_label[:, i]
            prob_index = torch.argmax(prob, dim=1)
            acc_dict[keys[i]] += float(torch.sum(prob_index == label).cpu().numpy())
            act_model_label_dict[keys[i]].extend(label.cpu())
            act_preds_dict[keys[i]].extend(prob_index.cpu())
    attr_list_dict_keys = sorted(attr_list_dict.keys())
    FEA.eval()
    meta.eval()
    CEloss = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        acc_dict = {}
        act_model_label_dict = {}
        act_preds_dict = {}
        for key in attr_list_dict.keys():
            acc_dict[key] = 0.0
            act_model_label_dict[key] = []
            act_preds_dict[key] = []
        avg_acc = 0.0

        

        for iter, data in enumerate(test_dataloader):
            loss_each = {}
            for attr_keys in attr_list_dict_keys:
                loss_each[attr_keys] = 0
            
            tm_outputs, model_label, model_multi_label, _ = data
            tm_outputs = torch.flatten(tm_outputs, start_dim=1)
            fea_max_mi, _ = FEA(tm_outputs)
            prob_list = meta(fea_max_mi)
            loss_ = compute_loss(prob_list, model_label, CEloss, 
                attr_list_dict_keys, loss_each)
            compute_acc(acc_dict, avg_acc, prob_list, model_label, act_model_label_dict, act_preds_dict)
            loss = torch.cat(loss_).sum()


            print("Test Meta Iter: [{}/{}] \t CE:{:.6f}\t".format(iter, tot_num / test_dataloader.batch_size, loss.item()))

       

        for attr_keys, v in acc_dict.items():
            cm_sources = confusion_matrix(act_model_label_dict[attr_keys], act_preds_dict[attr_keys], labels = [0, 1, 2, 3])
            print("{}_matrix:\n{},".format(attr_keys, cm_sources))
            acc_dict[attr_keys] /= (tot_num)
            avg_acc += acc_dict[attr_keys]
        avg_acc /= len(attr_list_dict.keys())
        
        for attr_keys in attr_list_dict.keys():
            print("{} : {}% (Random {}%)".format(attr_keys, acc_dict[attr_keys] * 100, 100 / len(attr_list_dict[attr_keys])))
        print("test average accuracy: {}%".format(avg_acc))
        
    return acc_dict, avg_acc

random_seeding(random_seed)
train(write_file_name, random_seed, train_info, attr_list_dict_tot)
