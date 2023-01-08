import jittor as jt
from jittor import nn, Module
import numpy as np
import sys, os
from jittor import init
import math 
import random
import pickle
from jittor.dataset import Dataset
import pygmtools as pygm
pygm.BACKEND = "jittor"
from configs import *
from utils import *
class WillowObject(Dataset):

    def __init__(self, Train = True, Shuffle = False, Eval = False):
        '''
        For training, please only set the Train = True and Eval = False,
            If use shuffle = True, you need to change the ground_truth matching matrix manually.
        For evaluation, please set Train = False, Eval = True and shuffle = False, since the eval function provided by pygm use the ground_truth in dataset.
        For plot, feel free to use Shuffle = False to test whether the results has dependence on the permutation of key points. (In the training we only see 1 maps to 1 and 2 maps to 2, but nothing like 1 maps 3!)
        If Shuffle = True, at the end of the get_item, there will be an extra index to represents the index.
        '''
        super(WillowObject, self).__init__()
        self.Benchmark = pygm.Benchmark("WillowObject", "train" if Train else "test", obj_resize)
        self.cls = self.Benchmark.classes
        ls, self.pair_counts = self.Benchmark.get_id_combination()
        self.Train = Train
        self.Shuffle = Shuffle
        self.Eval = Eval 
        self.index_list = ls # list of (list of pairs)
        self.img_list_1 = []
        self.kpts_list_1 = []
        self.img_list_2 = []
        self.kpts_list_2 = []
        if self.Eval:
            self.cls_list_1 = []
            self.cls_list_2 = []
            self.idx_list_1 = []
            self.idx_list_2 = []
        for ls in self.index_list:
            for a,b in ls:
                data_list, _ , idx = self.Benchmark.get_data([a,b], shuffle = False)
                self.img_list_1.append(data_list[0]['img'])
                Kp = data_list[0]['kpts']
                x = []
                y = []
                for dic in Kp:
                    x.append(dic['x'])
                    y.append(dic['y'])
                self.kpts_list_1.append([x,y])
                self.img_list_2.append(data_list[1]['img'])
                Kp = data_list[1]['kpts']
                x = []
                y = []
                for dic in Kp:
                    x.append(dic['x'])
                    y.append(dic['y'])
                self.kpts_list_2.append([x,y])

                if self.Eval:
                    self.cls_list_1.append(data_list[0]['cls'])
                    self.cls_list_2.append(data_list[1]['cls'])
                    self.idx_list_1.append(idx[0])
                    self.idx_list_2.append(idx[1])

                if (not self.Eval and not self.Train):
                    break

    def __getitem__(self, index):
        '''
        For training, we only will return img, kpts and A as type of jitter.Var, and shuffle index if Shuffle = True
        For plot(Train = False, Eval = False), we will return origin img, together with img, kpts and A as type of jittor.Var, and shuffle index if necessary
        For evaluation, we will return img, kpts, A, classification and index of img, and the perm_mat.
        '''
        img1, kpts1 = self.img_list_1[index], self.kpts_list_1[index]
        img2, kpts2 = self.img_list_2[index], self.kpts_list_2[index]
        img1 = jt.Var(np.array(img1, dtype=np.float32) / 256).transpose((2,0,1))
        img2 = jt.Var(np.array(img2, dtype=np.float32) / 256).transpose((2,0,1))
        kpts1 = jt.Var(kpts1)
        kpts2 = jt.Var(kpts2)

        if (self.Shuffle):
            a = np.arange(kpts1.shape[1]).astype(int)
            np.random.shuffle(a)
            kpts2 = kpts2[:,a]

        A1 = delaunay_triangulation(kpts1)
        A2 = delaunay_triangulation(kpts2)
        if (self.Train):
            if (self.Shuffle):
                return img1, img2, kpts1, kpts2, A1, A2, a
            else:
                return img1, img2, kpts1, kpts2, A1, A2
        else:
            if (not self.Eval):
                return self.img_list_1[index], self.img_list_2[index], img1, img2, kpts1, kpts2, A1, A2
            else:
                return img1, img2, kpts1, kpts2, A1, A2, self.idx_list_1[index], self.idx_list_2[index], self.cls_list_1[index],self.cls_list_2[index]
    
    def __len__(self):
        return len(self.img_list_1)


    def eval(self, prediction, classes, verbose, rm_gt_cache = False):
        return self.Benchmark.eval(prediction, classes, verbose, rm_gt_cache= rm_gt_cache)