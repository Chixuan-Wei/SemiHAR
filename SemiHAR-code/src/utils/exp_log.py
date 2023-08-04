import os
import time
import sklearn.metrics as metrics
import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
PATH_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
paths_dict = {
              'opportunity': os.path.join(PATH_DATA, 'opportunity'),
              'mhealthy': os.path.join(PATH_DATA, 'mhealthy'),
              'pamap2': os.path.join(PATH_DATA, 'pamap2'),
              'uci-har': os.path.join(PATH_DATA, 'uci-har'),
              }


class Logger:

    def __init__(self, dataset, n_folds, save_log=False, log_path=None):
        self.dataset = dataset
        self.n_folds = n_folds
        self.save_log = save_log
        self.log_file = '{}{}{}.csv'.format(log_path, os.sep, dataset)
        self.init_time = 0
        self.last_epoch_time = 0
        self.max_epochs = 0
        self.test_metric = 0
        self.best_epoch = -1
        self.best_acc = -1
        self.best_f1 = -1

    def __reset__(self):
        self.best_epoch = -1
        self.best_acc = -1
        self.best_f1 = -1
        self.test_metric = 0

    def update(self, k_fold, **kwargs):
        should_stop = False
        run_test = False

        if 'begin' in kwargs:
            self.init_time = self.last_epoch_time = time.time()
            self.__reset__()
            self.max_epochs = kwargs['max_epochs']

        elif 'test' in kwargs:
            self.test_metric = metrics.f1_score(kwargs['truth'], kwargs['prediction'], average='weighted')

        elif 'validation' in kwargs:
            curr_time = time.time()
            total_elapsed = curr_time - self.init_time
            epoch_elapsed = curr_time - self.last_epoch_time
            self.last_epoch_time = curr_time

            accuracy = metrics.accuracy_score(kwargs['truth'], kwargs['prediction'])
            f1 = metrics.f1_score(kwargs['truth'], kwargs['prediction'], average='weighted')
            re_call = metrics.recall_score(kwargs['truth_test'], kwargs['prediction_test'], average='weighted')
            accuracy_test = metrics.accuracy_score(kwargs['truth_test'], kwargs['prediction_test'])
            f1_test = metrics.f1_score(kwargs['truth_test'], kwargs['prediction_test'], average='weighted')
            if 'early_stop' in kwargs:
                early_stop = kwargs['early_stop']
            else:
                early_stop = self.max_epochs - kwargs['epoch']

            if accuracy > self.best_acc:
                self.best_acc = accuracy


            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = kwargs['epoch']
                run_test = True
            else:
                should_stop = kwargs['epoch'] - self.best_epoch > early_stop

            if kwargs['epoch'] % 25 == 0 or kwargs['epoch'] == 1:
                print('\n{}Experiment, Dataset: {}, Fold: {}/{}'
                      .format(' '*10, self.dataset, k_fold, self.n_folds))
                print('       │    elapsed time    │          loss           │                 metrics                              │  best')
                print(' epoch │  epoch  │   total  │  training  │ validation │ accuracy │   f1    │accuracy_test│f1_test│recall_test│   f1')
                print('───────┼─────────┼──────────┼────────────┼────────────┼──────────┼─────────┼─────────────┼───────┼───────────┼──────')

            print("{:6d} │ {:7.2f} │ {:8.2f} │ {:10.8f} │ {:10.8f} │ {:8.4f} │ {:7.4f} │ {:8.4f} │ {:7.4f} │{:8.4f}│{:6.4f} ({:3d})"
                  .format(kwargs['epoch'], epoch_elapsed, total_elapsed, kwargs['loss_train'],
                          kwargs['loss_validation'], accuracy, f1, accuracy_test, f1_test, re_call, self.best_f1, self.best_epoch))

            if should_stop or kwargs['epoch'] == self.max_epochs:
                self.__on_stop(k_fold, total_elapsed)
                if 'att_data' in kwargs:
                    att_np_data = kwargs['att_data']
                    att_np_data = np.array(att_np_data.cpu().detach().numpy(),dtype=float)
                    att_file = '{}{}att_{}_{}_{}_'.format(paths_dict[self.dataset], os.sep, self.best_epoch,'att_matrix',self.best_acc)
                    np.save(att_file,att_np_data)
                self.__reset__()

        return should_stop, run_test

    def __on_stop(self, k_fold, time_elapsed):
        print("K-fold {} best epoch: {:3d}, f1: {:.3f}  ---> test f1: {:.3f}"
              .format(k_fold, self.best_epoch, self.best_f1, self.test_metric))
        if self.save_log:
            header = ''
            if k_fold == 1:
                header = 'fold, time, best_epoch, best_accuracy, best_validation_f1, best_test_f1\n'
            with open(self.log_file, "a+") as text_file:
                text_file.write("{}{},{},{},{},{},{}\n"
                                .format(header, k_fold, time_elapsed, self.best_epoch, self.best_acc,
                                        self.best_f1, self.test_metric))


class Color:
    red = '\033[91m'
    green = '\033[92m'
    blue = '\033[94m'
    cyan = '\033[96m'
    white = '\033[97m'
    yellow = '\033[93m'
    magenta = '\033[95m'
    grey = '\033[90m'
    light_grey = '\033[37m'
    black = '\033[90m'
    default = light_grey
