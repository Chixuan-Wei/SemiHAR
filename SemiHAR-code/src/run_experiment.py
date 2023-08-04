import argparse
import json
import os
import time

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random
import math

from utils import datamanager as dm
from lab import models as m
from lab import engine as engine
from utils.exp_log import Logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import *
from utils import augmentation as aug

parser = argparse.ArgumentParser(description='Run experiment.')
parser.add_argument('-d', '--dataset', default='mhealthy', help="mhealthy opportunity pamap2 uci-har")
parser.add_argument('-f', '--nfolds', default=5, help="number of folds", type=int)
parser.add_argument('-p', '--preprocess',default='standardize', help='standardize or scale')
parser.add_argument('-i', '--image',default=25, help='imagesize')
parser.add_argument('-b', '--bins',default=6, help='6')
parser.add_argument('-s', '--save', dest='save', action='store_true')


class Experiment:
     def __init__(self,  dataset, n_folds, preprocess, imagesize, bins, save_log):
        self.dataset = dataset
        self.n_folds = n_folds
        self.k_fold = 1
        self.preprocess = preprocess
        self.gyro = True
        self.image = imagesize
        self.bins =bins
        self.save_log = save_log
        self.logger = None

     @staticmethod
     def __load_data__(dataset, gyro, preprocess, bins, image):
        return dm.load_dataset(dataset, bins=bins, images=image, seq_length=100, gyro=gyro, preprocess=preprocess)

     def update(self, **kwargs):
        return self.logger.update(self.k_fold, **kwargs)

     def aug_sample(self, x_tr_labeled, y_tr_labeled, x):
         label, counts = np.unique(y_tr_labeled.numpy(),return_counts=True)
         ratio_np = np.divide(counts,y_tr_labeled.shape[0])
         weight = torch.Tensor(np.reciprocal(ratio_np))
         num_sample = x.shape[0] - x_tr_labeled.shape[0]
         weight_sampler = list(torch.utils.data.sampler.WeightedRandomSampler(weight, num_sample, replacement=True))
         up_label, ip_counts = np.unique(np.array(weight_sampler),return_counts=True)
         print(up_label,ip_counts)
         upsample_c = None
         upsample_y = []
         for c in range(0,len(up_label)):
             mask = y_tr_labeled.eq(c)
             x_label_data = x_tr_labeled[mask]
             upsample_index = list(torch.utils.data.sampler.RandomSampler(x_label_data,num_samples=int(ip_counts[c]),replacement=True))
             for i in range(0,len(upsample_index)):
                 upsample_y.append(c)
                 index = upsample_index[i]
                 upsample = x_label_data[index]
                 upsample = upsample[np.newaxis,:,:,:]
                 if upsample_c is None:
                     upsample_c = upsample
                 else:
                     upsample_c = torch.cat((upsample_c,upsample),dim=0)

         upsample_y = np.array(upsample_y)
         upsample_y = torch.from_numpy(upsample_y)
         print(upsample_c.shape,np.mean(upsample_c.numpy()),np.std(upsample_c.numpy()))
         upsample_c = torch.squeeze(upsample_c,dim=1)
         upsample_c = aug.jitter(upsample_c)
         upsample_c = torch.unsqueeze(upsample_c,dim=1)
         print(upsample_c.shape,np.mean(upsample_c.numpy()),np.std(upsample_c.numpy()))

         x_tr_labeled = torch.cat((x_tr_labeled,upsample_c),dim=0)
         y_tr_labeled = torch.cat((y_tr_labeled,upsample_y),dim=0)

         return x_tr_labeled, y_tr_labeled


     def run(self):
        log_path = os.path.dirname('{}{}log{}'.format(os.path.dirname(os.path.abspath(__file__)),
                                                          os.sep, os.sep))
        self.logger = Logger(dataset=self.dataset, n_folds=self.n_folds,
                             save_log=self.save_log, log_path=log_path)

        ds = self.__load_data__(self.dataset, gyro=self.gyro, preprocess=self.preprocess, bins=self.bins, image=self.image)
        gyro = self.gyro and ds.gyr_data_labeled is not None

        x_acc_train = np.concatenate((ds.acc_data_labeled,ds.acc_data_unlabeled),axis=0)
        x_gyr_train = np.concatenate((ds.gyr_data_labeled,ds.gyr_data_unlabeled),axis=0)

        x_acc_mk = dm.load_image(self.dataset,filename='acc',data=x_acc_train,image_size=self.image,n_bins=self.bins)
        x_gyr_mk = dm.load_image(self.dataset,filename='gyr',data=x_gyr_train,image_size=self.image,n_bins=self.bins)
        print('x_acc_mk',x_acc_mk.shape)
        print('x_gyr_mk',x_gyr_mk.shape)

        x_mk = np.concatenate((x_acc_mk, x_gyr_mk), axis=1)
        x_mk = np.mean(x_mk, axis=1)
        print('x_mk',x_mk.shape)

        x_labeled, y_labeled, u_labeled = np.concatenate((ds.acc_data_labeled, ds.gyr_data_labeled), axis=2) if gyro else ds.acc_data_labeled, ds.y_labeled_train, ds.u_labeled_train
        x_unlabeled, y_unlabeled, u_unlabeled = np.concatenate((ds.acc_data_unlabeled, ds.gyr_data_unlabeled), axis=2) if gyro else ds.acc_data_unlabeled, ds.y_unlabeled_train, ds.u_unlabeled_train

        x_ts_accmk = dm.load_image(self.dataset,filename='ts_acc',data=ds.acc_data_test,image_size=self.image,n_bins=self.bins)
        x_ts_gyrmk = dm.load_image(self.dataset,filename='ts_gyr',data=ds.gyr_data_test,image_size=self.image,n_bins=self.bins)
        print('x_ts_accmk',x_ts_accmk.shape)
        print('x_ts_gyrmk',x_ts_gyrmk.shape)
        x_ts_mk = np.concatenate((x_ts_accmk, x_ts_gyrmk), axis=1)
        x_ts_mk = np.mean(x_ts_mk, axis=1)
        print('x_ts_mk',x_ts_mk.shape)

        x_ts_np, y_ts_np = np.concatenate((ds.acc_data_test, ds.gyr_data_test), axis=2) if gyro else ds.acc_data_test, ds.y_test
        x_ts_np = np.reshape(x_ts_np, newshape=(x_ts_np.shape[0], 1, x_ts_np.shape[1], x_ts_np.shape[2]))

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        print(f'Using device: {device}')

        x_ts = torch.from_numpy(x_ts_np).float().to(device)
        y_ts = torch.from_numpy(y_ts_np).long().to(device)
        print('x_ts.shape',x_ts.shape, y_ts.shape)

        u = np.concatenate((u_labeled,u_unlabeled),axis=0)
        x = np.concatenate((x_labeled,x_unlabeled),axis=0)

        n_out = np.unique(y_labeled).size
        u_out = np.unique(u).size
        image_size = self.image
        print('classes',n_out)
        print('users',u_out)
        print('image_size',image_size)

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=0)
        self.k_fold = 1
        print("Test: features shape, labels shape, mean, standard deviation")
        print(x_ts_np.shape, y_ts_np.shape, np.mean(x_ts_np), np.std(x_ts_np))
        print("x: features shape, labels shape(labeled),user shape, mean, standard deviation")
        print(x.shape,y_labeled.shape,u.shape,np.mean(x),np.std(x))
        for tr_i, va_i in skf.split(X=x_labeled, y=y_labeled):
            print(tr_i.shape)
            x_tr_labeled, x_va_labeled = x_labeled[tr_i], x_labeled[va_i]
            y_tr_labeled, y_va_labeled = y_labeled[tr_i], y_labeled[va_i]
            x_val_accmk = dm.load_image(self.dataset,filename='val_acc_'+ str(self.k_fold)+str(time.time()).split('.')[0],data=x_va_labeled[:,:,0:3],image_size=self.image,n_bins=self.bins)
            x_val_gyrmk = dm.load_image(self.dataset,filename='val_gyr_'+ str(self.k_fold)+str(time.time()).split('.')[0],data=x_va_labeled[:,:,4:6],image_size=self.image,n_bins=self.bins)
            print('x_val_accmk',x_val_accmk.shape)
            print('x_val_gyrmk',x_val_gyrmk.shape)
            x_val_mk = np.concatenate((x_val_accmk, x_val_gyrmk), axis=1)
            x_val_mk = np.mean(x_val_mk, axis=1)
            print('x_val_mk',x_val_mk.shape)
            x_val_mk = np.reshape(x_val_mk, newshape=(x_val_mk.shape[0], 1, x_val_mk.shape[1], x_val_mk.shape[2]))

            x_tr_labeled = np.reshape(x_tr_labeled, newshape=(x_tr_labeled.shape[0], 1, x_tr_labeled.shape[1], x_tr_labeled.shape[2]))
            x_va_labeled = np.reshape(x_va_labeled, newshape=(x_va_labeled.shape[0], 1, x_va_labeled.shape[1], x_va_labeled.shape[2]))
            if self.k_fold == 1:
                x = np.reshape(x, newshape=(x.shape[0],1, x.shape[1], x.shape[2]))
                x_mk = np.reshape(x_mk, newshape=(x_mk.shape[0], 1, x_mk.shape[1], x_mk.shape[2]))
                x_ts_mk = np.reshape(x_ts_mk, newshape=(x_ts_mk.shape[0], 1, x_ts_mk.shape[1], x_ts_mk.shape[2]))
            else:
                x = x.cpu().numpy()
                x_mk = x_mk.cpu().numpy()
                u = u.cpu().numpy()
                x_ts_mk = x_ts_mk.cpu().numpy()


            print("Training: features shape, labels shape, mean, standard deviation")
            print(x_tr_labeled.shape, y_tr_labeled.shape, np.mean(x_tr_labeled), np.std(x_tr_labeled))
            print("Training_mk: features shape, mean, standard deviation")
            print(x_mk.shape, np.mean(x_mk), np.std(x_mk))
            print("Validation: features shape, labels shape, mean, standard deviation")
            print(x_va_labeled.shape, y_va_labeled.shape, np.mean(x_va_labeled), np.std(x_va_labeled))
            print("User: features shape, labels shape, mean, standard deviation")
            print(x.shape, u.shape, np.mean(x), np.std(x))

            x_mk = torch.from_numpy(x_mk).float().to(device)
            x_ts_mk = torch.from_numpy(x_ts_mk).float().to(device)
            x_val_mk = torch.from_numpy(x_val_mk).float().to(device)
            u = torch.from_numpy(u).long().to(device)
            x = torch.from_numpy(x).float().to(device)
            x_va_labeled = torch.from_numpy(x_va_labeled).float().to(device)
            y_va_labeled = torch.from_numpy(y_va_labeled).long().to(device)
            y_tr_labeled = torch.from_numpy(y_tr_labeled)
            x_tr_labeled = torch.from_numpy(x_tr_labeled)


            print(np.unique(y_tr_labeled.cpu().numpy(), return_counts=True))
            print(np.unique(y_va_labeled.cpu().numpy(), return_counts=True))
            print(np.unique(y_ts.cpu().numpy(), return_counts=True))
            print(np.unique(u.cpu().numpy(), return_counts=True))

            x_tr_labeled,y_tr_labeled = self.aug_sample(x_tr_labeled,y_tr_labeled,x)

            x_tr_labeled = x_tr_labeled.float().to(device)
            y_tr_labeled = y_tr_labeled.long().to(device)


            encoder = m.Encoder(x_mk, x_tr_labeled, u_out, n_out, image_size*image_size)
            print(encoder)
            trainer = engine.TrainerEnco(encoder,learning_rate=0.001, batch_size=100, weight_decay=0.00005,max_epoch=2000)
            trainer.train(train_data=TensorDataset(x_mk, x_tr_labeled, y_tr_labeled, x, u),
                          validation_data=TensorDataset(x_val_mk, x_va_labeled, y_va_labeled),
                          test_data=TensorDataset(x_ts_mk, x_ts, y_ts),update_callback=self.update)


            self.k_fold += 1


if __name__ == "__main__":
    args = parser.parse_args()
    experiment = Experiment(args.dataset, args.nfolds, args.preprocess, args.image, args.bins, args.save)
    experiment.run()




