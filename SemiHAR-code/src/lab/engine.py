import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import numpy as np
from abc import abstractmethod
import torch.nn.functional as F

from lab import models as Encoder

class Trainer:
     def __init__(self, model):
         self.model = model
         if torch.cuda.is_available():
            self.model.cuda()

     @abstractmethod
     def train(self, train_data, validation_data=None, test_data=None, update_callback=None):
        raise NotImplementedError


class TrainerEnco(Trainer):
    def __init__(self, model, learning_rate, batch_size, weight_decay, max_epoch):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.optimizer = torch.optim.Adagrad(self.model.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)

    def train(self, train_data, validation_data=None, test_data=None, update_callback=None):
        max_epochs = self.max_epoch
        loss_validation = 0
        predicted = None
        truth = None
        should_stop = False
        run_test = False

        dl_trn = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if validation_data is not None:
         dl_val = DataLoader(dataset=validation_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if test_data is not None:
         dl_tst = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=True, drop_last=False)

        if update_callback is not None:
            update_callback(begin=True, max_epochs=max_epochs)

        for epoch in range(max_epochs):
            loss_train, att_epoch_data = self.__train_epoch(dl_trn)
            if  validation_data  is not None:
                loss_validation, predicted, truth = self.evaluate(dl_val)
                loss_test,predicted_test,truth_test = self.evaluate(dl_tst)
            if update_callback is not None:
                kwargs = {'validation': True}
                kwargs['epoch'] = epoch + 1
                kwargs['early_stop'] = 50
                kwargs['loss_train'] = loss_train
                kwargs['loss_validation'] = loss_validation
                kwargs['prediction'] = predicted
                kwargs['truth'] = truth
                kwargs['prediction_test'] = predicted_test
                kwargs['truth_test'] = truth_test
                kwargs['att_data'] = att_epoch_data
                should_stop, run_test = update_callback(**kwargs)
            if should_stop:
                break

            if run_test and test_data is not None:
                kwargs = {'test': True}
                _, predicted, truth = self.evaluate(dl_tst)
                kwargs['prediction'] = predicted
                kwargs['truth'] = truth
                _, _ = update_callback(**kwargs)


    def __train_epoch(self,dataset):
        total_loss = 0
        n = 0
        att_data = []
        self.model.train()

        for i,(x_mk_batch, x_labeled_batch, y_labeled_batch, x_batch, u_batch) in enumerate(dataset):
            n += y_labeled_batch.shape[0]

            self.optimizer.zero_grad()
            i_hat, u_hat, c_hat = self.model(x_mk_batch, x_labeled_batch, x_batch)
            loss_i = nn.MSELoss()(i_hat,x_mk_batch)
            loss_u = F.cross_entropy(u_hat,u_batch)
            loss_c = F.cross_entropy(c_hat,y_labeled_batch)
            loss_all = loss_u+loss_c+loss_i

            loss_all.backward()
            self.optimizer.step()

            total_loss += loss_all.item()

        return total_loss / n

    def evaluate(self, dataset):
        total_loss = 0
        n = 0
        prediction = []
        truth = []
        self.model.eval()

        for i,(ts_mk_batch, observations, labels) in enumerate(dataset):
            n += labels.shape[0]
            _,_,outputs = self.model(ts_mk_batch, observations, observations)
            loss = F.cross_entropy(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            prediction.extend(predicted.cpu().numpy().tolist())
            truth.extend(labels.cpu().data.numpy().tolist())

            total_loss += loss.item()

        return total_loss / n, np.array(prediction), truth








