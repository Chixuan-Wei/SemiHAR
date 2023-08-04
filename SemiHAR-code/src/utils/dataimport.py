import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split

PATH_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

position_list = ['unknown', 'wrist', 'waist', 'chest', 'ankle', 'arm', 'pocket']
device_list = ['unknown', 'smartphone', 'smartwatch', 'imu']
paths_dict = {
              'opportunity': os.path.join(PATH_DATA, 'opportunity'),
              'mhealthy': os.path.join(PATH_DATA, 'mhealthy'),
              'pamap2': os.path.join(PATH_DATA, 'pamap2'),
              'uci-har': os.path.join(PATH_DATA, 'uci-har'),
              }
SEED = 2018

datasets_activities = {
    'mhealth': ['standing', 'sitting', 'laying', 'walking', 'stairs_up', 'bend forward', 'arms elevation',
                'crouching', 'cycling', 'jogging', 'running', 'jump front and back'],
    'pamap2': ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic walking',
               'ascending stairs', 'descending stairs', 'vacuum cleaning', 'ironing', 'rope jumping'],
    'uci-har': ['walking', 'upstairs', 'downstairs', 'sitting', 'standing', 'laying'],
    'opportunity': ['Stand', 'Walk', 'Sit', 'Lie'],
}


class Dataset:

    def __init__(self, acc_data_labeled, gyr_data_labeled, y_labeled_train, u_labeled_train,
                       acc_data_unlabeled, gyr_data_unlabeled, y_unlabeled_train, u_unlabeled_train,
                       acc_data_test, gyr_data_test, y_test, u_test):
        self.acc_data_labeled = acc_data_labeled
        self.gyr_data_labeled = gyr_data_labeled
        self.y_labeled_train = y_labeled_train
        self.u_labeled_train = u_labeled_train
        self.acc_data_unlabeled = acc_data_unlabeled
        self.gyr_data_unlabeled = gyr_data_unlabeled
        self.y_unlabeled_train = y_unlabeled_train
        self.u_unlabeled_train = u_unlabeled_train
        self.acc_data_test = acc_data_test
        self.gyr_data_test = gyr_data_test
        self.y_test = y_test
        self.u_test = u_test

def preprocess_data(data, preprocess=None):
    if preprocess == 'standardize':
        return __standardize(data)
    elif preprocess == 'scale':
        return __scale(data)
    else:
        return data

def build_opportunity_unlabeled(seq_length, dataset_name, bins,images):
    dir_path = os.path.join(paths_dict['opportunity'], 'dataset')

    columns = [[[37, 40], [40, 43]], [[50, 53], [53, 56]], [[63, 66], [66, 69]], [[76, 79], [79, 82]],
               [[89, 92], [92, 95]], [[108, 111], [111, 114]], [[124, 127], [127, 130]]]

    files = []
    x_acc_u2 = []
    x_gyr_u2 = []
    y_u2 = []
    u_u2 = []

    for file in os.listdir(dir_path):
        if file.endswith(".dat"):
            files.append(os.path.join(dir_path, file))

    for file in files:
        if os.path.basename(file).startswith("S4"):
            user_id = 1
            data = pd.read_csv(file, header=None, delim_whitespace=True).values
            data_act = data[:, 243]
            n = data_act.shape[0]

            for i in range(0, n, seq_length // 2):
                act = data_act[i:(i + seq_length)]
                if len(np.unique(act)) == 1 and act[0] > 0:
                    activity = [1, 2, 4, 5].index(act[0])
                    for col in columns:
                        acc = data[i:(i + seq_length), col[0][0]:col[0][1]]
                        gyr = data[i:(i + seq_length), col[1][0]:col[1][1]]
                        if not (np.isnan(acc).any() or np.isnan(gyr).any()):
                            x_acc_u2.append(acc)
                            x_gyr_u2.append(gyr)
                            y_u2.append(activity)
                            u_u2.append(user_id)

    acc_u2_data = np.array(x_acc_u2,dtype=float)
    gyr_u2_data = np.array(x_gyr_u2,dtype=float)
    y_u2_data = np.array(y_u2)
    u_u2_data = np.array(u_u2)
    acc_file_un = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    gyr_file_un = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    y_file_un = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    u_file_un = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    print(np.unique(y_u2_data , return_counts=True))
    print(np.unique(u_u2_data , return_counts=True))
    print('un_Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_un,acc_u2_data)
    np.save(gyr_file_un,gyr_u2_data)
    np.save(y_file_un,y_u2_data)
    np.save(u_file_un,u_u2_data)

def build_opportunity_labeled(seq_length, dataset_name,bins,images):
    dir_path = os.path.join(paths_dict['opportunity'], 'dataset')

    columns = [[[37, 40], [40, 43]], [[50, 53], [53, 56]], [[63, 66], [66, 69]], [[76, 79], [79, 82]],
               [[89, 92], [92, 95]], [[108, 111], [111, 114]], [[124, 127], [127, 130]]]

    files = []
    x_acc_u1 = []
    x_gyr_u1 = []
    y_u1 = []
    u_u1 = []

    for file in os.listdir(dir_path):
        if file.endswith(".dat"):
            files.append(os.path.join(dir_path, file))

    for file in files:
        if os.path.basename(file).startswith("S1"):
            user_id = 0
            data = pd.read_csv(file, header=None, delim_whitespace=True).values
            data_act = data[:, 243]
            n = data_act.shape[0]

            for i in range(0, n, seq_length // 2):
                act = data_act[i:(i + seq_length)]
                if len(np.unique(act)) == 1 and act[0] > 0:
                    activity = [1, 2, 4, 5].index(act[0])
                    for col in columns:
                        acc = data[i:(i + seq_length), col[0][0]:col[0][1]]
                        gyr = data[i:(i + seq_length), col[1][0]:col[1][1]]
                        if not (np.isnan(acc).any() or np.isnan(gyr).any()):
                            x_acc_u1.append(acc)
                            x_gyr_u1.append(gyr)
                            y_u1.append(activity)
                            u_u1.append(user_id)

    acc_u1_data = np.array(x_acc_u1,dtype=float)
    gyr_u1_data = np.array(x_gyr_u1,dtype=float)
    y_u1_data = np.array(y_u1)
    u_u1_data = np.array(u_u1)
    acc_file_tv = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    gyr_file_tv = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    y_file_tv = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    u_file_tv = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    acc_file_ts = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    gyr_file_ts = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    y_file_ts = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    u_file_ts = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    indices = range(y_u1_data.shape[0])
    ind_la, ind_ts = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y_u1_data)
    print(np.unique(y_u1_data , return_counts=True))
    print(np.unique(u_u1_data , return_counts=True))
    print('la_Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_tv, acc_u1_data[ind_la])
    np.save(gyr_file_tv, gyr_u1_data[ind_la])
    np.save(y_file_tv, y_u1_data[ind_la])
    np.save(u_file_tv, u_u1_data[ind_la])
    np.save(acc_file_ts,acc_u1_data[ind_ts])
    np.save(gyr_file_ts,gyr_u1_data[ind_ts])
    np.save(y_file_ts,y_u1_data[ind_ts])
    np.save(u_file_ts,u_u1_data[ind_ts])


def build_mhealth_unlabeled(seq_length, dataset_name, bins,images):
    ankle_columns_acc = [5, 6, 7]
    ankle_columns_gyr = [8, 9, 10]
    arm_columns_acc = [14, 15, 16]
    arm_columns_gyr = [17, 18, 19]
    data_files = ['mHealth_subject' + str(i + 1) + '.log' for i in range(10)]

    x_acc_u2 = []
    x_gyr_u2 = []
    y_u2 = []
    u_u2 = []
    user_id = -1

    for data_file in data_files:
        print(data_file)
        user_id += 1
        if data_file.endswith("subject2.log"):
                file = os.path.join(paths_dict['mhealthy'], data_file)
                print(user_id)
                data_np = pd.read_csv(file, sep='\t').values
                activities = range(1, len(datasets_activities['mhealth']) + 1)
                for activity in activities:
                    act_np = data_np[data_np[:, 23] == activity, :]

                    print('user: %d, device: %d, activity: %d, data length: %d' %
                      (user_id, device_list.index('imu'), activity, act_np.shape[0]))

                    for index in range(0, act_np.shape[0] - seq_length + 1, seq_length // 2):
                        x_acc_u2.append(act_np[index:(index + seq_length), ankle_columns_acc])
                        x_gyr_u2.append(act_np[index:(index + seq_length), ankle_columns_gyr])
                        y_u2.append([device_list.index('imu'), position_list.index('ankle'), activity - 1])
                        u_u2.append(user_id)
                        x_acc_u2.append(act_np[index:(index + seq_length), arm_columns_acc])
                        x_gyr_u2.append(act_np[index:(index + seq_length), arm_columns_gyr])
                        y_u2.append([device_list.index('imu'), position_list.index('wrist'), activity - 1])
                        u_u2.append(user_id)
                acc_u2_data = np.array(x_acc_u2, dtype=float)
                gyr_u2_data = np.array(x_gyr_u2, dtype=float)
                y_u2_data = np.array(y_u2)[:, 2]
                u_u2_data = np.array(u_u2)
                acc_file_un = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
                gyr_file_un = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
                y_file_un = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
                u_file_un = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
                print(np.unique(y_u2_data , return_counts=True))
                print(np.unique(u_u2_data , return_counts=True))
                print('Saving data to' + paths_dict[dataset_name])
                np.save(acc_file_un,acc_u2_data)
                np.save(gyr_file_un,gyr_u2_data)
                np.save(y_file_un,y_u2_data)
                np.save(u_file_un,u_u2_data)

def build_mhealth_labeled(seq_length, dataset_name, bins,images):
    ankle_columns_acc = [5, 6, 7]
    ankle_columns_gyr = [8, 9, 10]
    arm_columns_acc = [14, 15, 16]
    arm_columns_gyr = [17, 18, 19]
    data_files = ['mHealth_subject' + str(i + 1) + '.log' for i in range(10)]

    x_acc_u1 = []
    x_gyr_u1 = []
    y_u1 = []
    u_u1 = []
    user_id = -1

    for data_file in data_files:
        print(data_file)
        user_id += 1
        if data_file.endswith("subject1.log"):
            file = os.path.join(paths_dict['mhealthy'], data_file)
            data_np = pd.read_csv(file, sep='\t').values
            activities = range(1, len(datasets_activities['mhealth']) + 1)
            print(user_id)
            for activity in activities:
                act_np = data_np[data_np[:, 23] == activity, :]

                print('user: %d, device: %d, activity: %d, data length: %d' %
                  (user_id, device_list.index('imu'), activity, act_np.shape[0]))

                for index in range(0, act_np.shape[0] - seq_length + 1, seq_length // 2):
                    x_acc_u1.append(act_np[index:(index + seq_length), ankle_columns_acc])
                    x_gyr_u1.append(act_np[index:(index + seq_length), ankle_columns_gyr])
                    y_u1.append([device_list.index('imu'), position_list.index('ankle'), activity - 1])
                    u_u1.append(user_id)
                    x_acc_u1.append(act_np[index:(index + seq_length), arm_columns_acc])
                    x_gyr_u1.append(act_np[index:(index + seq_length), arm_columns_gyr])
                    y_u1.append([device_list.index('imu'), position_list.index('wrist'), activity - 1])
                    u_u1.append(user_id)
            acc_u1_data = np.array(x_acc_u1, dtype=float)
            gyr_u1_data = np.array(x_gyr_u1, dtype=float)
            y_u1_data = np.array(y_u1)[:, 2]
            u_u1_data = np.array(u_u1)
            acc_file_tv = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
            gyr_file_tv = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
            y_file_tv = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
            u_file_tv = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
            acc_file_ts = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
            gyr_file_ts = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
            y_file_ts = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
            u_file_ts = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
            print(np.unique(y_u1_data , return_counts=True))
            print(np.unique(u_u1_data , return_counts=True))
            indices = range(y_u1_data.shape[0])
            ind_la, ind_ts = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y_u1_data)
            print('Saving data to' + paths_dict[dataset_name])
            np.save(acc_file_tv, acc_u1_data[ind_la])
            np.save(gyr_file_tv, gyr_u1_data[ind_la])
            np.save(y_file_tv, y_u1_data[ind_la])
            np.save(u_file_tv, u_u1_data[ind_la])
            np.save(acc_file_ts,acc_u1_data[ind_ts])
            np.save(gyr_file_ts,gyr_u1_data[ind_ts])
            np.save(y_file_ts,y_u1_data[ind_ts])
            np.save(u_file_ts,u_u1_data[ind_ts])


def __downsample(data, seq_length):
    step = data.shape[0] / seq_length
    index = 0.0
    indices = []
    for i in range(seq_length):
        indices.append(round(index))
        index += step
    return data[indices]

def build_pamap2_unlabled(seq_length, dataset_name, bins,images):
    time_step = (seq_length / 100)
    margin = 0.05
    lower_ts = time_step * (1 - margin)
    upper_ts = time_step * (1 + margin)
    lower_len = int(seq_length * (1 - margin))
    activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    device = device_list.index('imu')

    x_acc_u2 = []
    x_gyr_u2 = []
    y_u2 = []
    u_u2 = []

    for user in range(2, 3):
        fn = '%s%s%d.dat' % (paths_dict['pamap2'], os.sep + 'Protocol' + os.sep + 'subject10', user)
        data = pd.read_csv(fn, header=None, delim_whitespace=True)
        data.sort_values(by=0)
        data = data.values
        user_id = user-1
        for activity in activities:
            data_act = data[data[:, 1] == activity, :]
            if data_act.size == 0:
                continue
            print('user {}, activity {}, measurements: {}'.format(user_id, activity, data_act.shape[0]))
            index = 0
            while index < data_act.shape[0]:
                it = data_act[index, 0]
                et = it + time_step
                t_data = data_act[(data_act[:, 0] >= it) & (data_act[:, 0] < et), :]
                ts = t_data[t_data.shape[0] - 1, 0] - t_data[0, 0]

                if t_data.shape[0] >= lower_len and lower_ts < ts < upper_ts:
                    act = activities.index(activity)

                    position = position_list.index('wrist')
                    acc1 = t_data[~np.isnan(t_data[:, 4:7]).any(axis=1), 4:7]
                    acc2 = t_data[~np.isnan(t_data[:, 7:10]).any(axis=1), 7:10]
                    gyr = t_data[~np.isnan(t_data[:, 10:13]).any(axis=1), 10:13]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u2.append(__downsample(acc1, seq_length))
                        x_gyr_u2.append(__downsample(gyr, seq_length))
                        y_u2.append([device, position, act])
                        u_u2.append(user_id)
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u2.append(__downsample(acc2, seq_length))
                        x_gyr_u2.append(__downsample(gyr, seq_length))
                        y_u2.append([device, position, act])
                        u_u2.append(user_id)

                    position = position_list.index('chest')
                    acc1 = t_data[~np.isnan(t_data[:, 21:24]).any(axis=1), 21:24]
                    acc2 = t_data[~np.isnan(t_data[:, 24:27]).any(axis=1), 24:27]
                    gyr = t_data[~np.isnan(t_data[:, 27:30]).any(axis=1), 27:30]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u2.append(__downsample(acc1, seq_length))
                        x_gyr_u2.append(__downsample(gyr, seq_length))
                        y_u2.append([device, position, act])
                        u_u2.append(user_id)
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u2.append(__downsample(acc2, seq_length))
                        x_gyr_u2.append(__downsample(gyr, seq_length))
                        y_u2.append([device, position, act])
                        u_u2.append(user_id)

                    position = position_list.index('ankle')
                    acc1 = t_data[~np.isnan(t_data[:, 38:41]).any(axis=1), 38:41]
                    acc2 = t_data[~np.isnan(t_data[:, 41:44]).any(axis=1), 41:44]
                    gyr = t_data[~np.isnan(t_data[:, 44:47]).any(axis=1), 44:47]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u2.append(__downsample(acc1, seq_length))
                        x_gyr_u2.append(__downsample(gyr, seq_length))
                        y_u2.append([device, position, act])
                        u_u2.append(user_id)
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u2.append(__downsample(acc2, seq_length))
                        x_gyr_u2.append(__downsample(gyr, seq_length))
                        y_u2.append([device, position, act])
                        u_u2.append(user_id)

                    index += t_data.shape[0] // 2 + 1
                else:
                    index += t_data.shape[0]

    acc_u2_data = np.array(x_acc_u2, dtype=float)
    gyr_u2_data = np.array(x_gyr_u2, dtype=float)
    y_u2_data = np.array(y_u2)[:, 2]
    u_u2_data = np.array(u_u2)
    acc_file_un = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    gyr_file_un = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    y_file_un = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    u_file_un = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    print(np.unique(y_u2_data , return_counts=True))
    print(np.unique(u_u2_data , return_counts=True))
    print('Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_un,acc_u2_data)
    np.save(gyr_file_un,gyr_u2_data)
    np.save(y_file_un,y_u2_data)
    np.save(u_file_un,u_u2_data)


def build_pamap2_labled(seq_length, dataset_name, bins,images):
    time_step = (seq_length / 100)
    margin = 0.05
    lower_ts = time_step * (1 - margin)
    upper_ts = time_step * (1 + margin)
    lower_len = int(seq_length * (1 - margin))
    activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    device = device_list.index('imu')

    x_acc_u1 = []
    x_gyr_u1 = []
    y_u1 = []
    u_u1 = []

    for user in range(1, 2):
        fn = '%s%s%d.dat' % (paths_dict['pamap2'], os.sep + 'Protocol' + os.sep + 'subject10', user)
        data = pd.read_csv(fn, header=None, delim_whitespace=True)
        data.sort_values(by=0)
        data = data.values
        user_id = user-1
        for activity in activities:
            data_act = data[data[:, 1] == activity, :]
            if data_act.size == 0:
                continue
            print('user {}, activity {}, measurements: {}'.format(user_id, activity, data_act.shape[0]))
            index = 0
            while index < data_act.shape[0]:
                it = data_act[index, 0]
                et = it + time_step
                t_data = data_act[(data_act[:, 0] >= it) & (data_act[:, 0] < et), :]
                ts = t_data[t_data.shape[0] - 1, 0] - t_data[0, 0]

                if t_data.shape[0] >= lower_len and lower_ts < ts < upper_ts:
                    act = activities.index(activity)

                    position = position_list.index('wrist')
                    acc1 = t_data[~np.isnan(t_data[:, 4:7]).any(axis=1), 4:7]
                    acc2 = t_data[~np.isnan(t_data[:, 7:10]).any(axis=1), 7:10]
                    gyr = t_data[~np.isnan(t_data[:, 10:13]).any(axis=1), 10:13]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u1.append(__downsample(acc1, seq_length))
                        x_gyr_u1.append(__downsample(gyr, seq_length))
                        y_u1.append([device, position, act])
                        u_u1.append(user_id)
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u1.append(__downsample(acc2, seq_length))
                        x_gyr_u1.append(__downsample(gyr, seq_length))
                        y_u1.append([device, position, act])
                        u_u1.append(user_id)

                    position = position_list.index('chest')
                    acc1 = t_data[~np.isnan(t_data[:, 21:24]).any(axis=1), 21:24]
                    acc2 = t_data[~np.isnan(t_data[:, 24:27]).any(axis=1), 24:27]
                    gyr = t_data[~np.isnan(t_data[:, 27:30]).any(axis=1), 27:30]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u1.append(__downsample(acc1, seq_length))
                        x_gyr_u1.append(__downsample(gyr, seq_length))
                        y_u1.append([device, position, act])
                        u_u1.append(user_id)
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u1.append(__downsample(acc2, seq_length))
                        x_gyr_u1.append(__downsample(gyr, seq_length))
                        y_u1.append([device, position, act])
                        u_u1.append(user_id)

                    position = position_list.index('ankle')
                    acc1 = t_data[~np.isnan(t_data[:, 38:41]).any(axis=1), 38:41]
                    acc2 = t_data[~np.isnan(t_data[:, 41:44]).any(axis=1), 41:44]
                    gyr = t_data[~np.isnan(t_data[:, 44:47]).any(axis=1), 44:47]
                    if acc1.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u1.append(__downsample(acc1, seq_length))
                        x_gyr_u1.append(__downsample(gyr, seq_length))
                        y_u1.append([device, position, act])
                        u_u1.append(user_id)
                    if acc2.shape[0] >= lower_len and gyr.shape[0] >= lower_len:
                        x_acc_u1.append(__downsample(acc2, seq_length))
                        x_gyr_u1.append(__downsample(gyr, seq_length))
                        y_u1.append([device, position, act])
                        u_u1.append(user_id)

                    index += t_data.shape[0] // 2 + 1
                else:
                    index += t_data.shape[0]

    acc_u1_data = np.array(x_acc_u1, dtype=float)
    gyr_u1_data = np.array(x_gyr_u1, dtype=float)
    y_u1_data = np.array(y_u1)[:, 2]
    u_u1_data = np.array(u_u1)
    acc_file_tv = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    gyr_file_tv = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    y_file_tv = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    u_file_tv = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    acc_file_ts = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    gyr_file_ts = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    y_file_ts = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    u_file_ts = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    print(np.unique(y_u1_data , return_counts=True))
    print(np.unique(u_u1_data , return_counts=True))
    indices = range(y_u1_data.shape[0])
    ind_la, ind_ts = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y_u1_data)
    print('Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_tv, acc_u1_data[ind_la])
    np.save(gyr_file_tv, gyr_u1_data[ind_la])
    np.save(y_file_tv, y_u1_data[ind_la])
    np.save(u_file_tv, u_u1_data[ind_la])
    np.save(acc_file_ts,acc_u1_data[ind_ts])
    np.save(gyr_file_ts,gyr_u1_data[ind_ts])
    np.save(y_file_ts,y_u1_data[ind_ts])
    np.save(u_file_ts,u_u1_data[ind_ts])

def build_uci_har_unlabeled(seq_length, dataset_name, bins,images):
    key_file_name = os.path.join(paths_dict['uci-har'], 'RawData', 'labels.txt')
    activities = range(1, len(datasets_activities['uci-har']) + 1)

    x_acc_u2 = []
    x_gyr_u2 = []
    y_u2 = []
    u_u2 = []

    for key_line in open(key_file_name):
        keys = key_line[:-1].split(' ')
        if int(keys[1]) == 6 or int(keys[1]) == 7 or int(keys[1]) == 8 or int(keys[1]) == 9 or int(keys[1]) == 10:
            activity = int(keys[2])
            if activity not in activities:
                continue
            keys[0] = '0%s' % keys[0] if int(keys[0]) < 10 else keys[0]
            keys[1] = '0%s' % keys[1] if int(keys[1]) < 10 else keys[1]
            acc_file_name = '%s%sRawData%sacc_exp%s_user%s.txt' % (paths_dict['uci-har'], os.sep, os.sep, keys[0], keys[1])
            gyr_file_name = '%s%sRawData%sgyro_exp%s_user%s.txt' % (paths_dict['uci-har'], os.sep, os.sep, keys[0], keys[1])
            acc_np = pd.read_csv(acc_file_name, sep=' ', header=None).values[int(keys[3]):(int(keys[4]) + 1)]
            gyr_np = pd.read_csv(gyr_file_name, sep=' ', header=None).values[int(keys[3]):(int(keys[4]) + 1)]

            for index in range(0, acc_np.shape[0] - seq_length + 1, seq_length // 2):
                x_acc_u2.append(acc_np[index:(index + seq_length), :])
                x_gyr_u2.append(gyr_np[index:(index + seq_length), :])
                y_u2.append([device_list.index('smartphone'), position_list.index('waist'), activity - 1])
                u_u2.append(int(keys[1])-1)
    acc_u2_data = np.array(x_acc_u2, dtype=float)
    gyr_u2_data = np.array(x_gyr_u2, dtype=float)
    y_u2_data = np.array(y_u2)[:, 2]
    u_u2_data = np.array(u_u2)
    acc_file_un = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    gyr_file_un = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    y_file_un = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    u_file_un = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    print(np.unique(y_u2_data , return_counts=True))
    print(np.unique(u_u2_data , return_counts=True))
    print('Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_un,acc_u2_data)
    np.save(gyr_file_un,gyr_u2_data)
    np.save(y_file_un,y_u2_data)
    np.save(u_file_un,u_u2_data)

def build_uci_har_labeled(seq_length, dataset_name, bins,images):
    key_file_name = os.path.join(paths_dict['uci-har'], 'RawData', 'labels.txt')
    activities = range(1, len(datasets_activities['uci-har']) + 1)

    x_acc_u1 = []
    x_gyr_u1 = []
    y_u1 = []
    u_u1 = []

    for key_line in open(key_file_name):
        keys = key_line[:-1].split(' ')
        if int(keys[1]) == 1 or int(keys[1]) == 2 or int(keys[1]) == 3 or int(keys[1]) == 4 or int(keys[1]) == 5 :
            activity = int(keys[2])
            if activity not in activities:
                continue
            keys[0] = '0%s' % keys[0] if int(keys[0]) < 10 else keys[0]
            keys[1] = '0%s' % keys[1] if int(keys[1]) < 10 else keys[1]
            acc_file_name = '%s%sRawData%sacc_exp%s_user%s.txt' % (paths_dict['uci-har'], os.sep, os.sep, keys[0], keys[1])
            gyr_file_name = '%s%sRawData%sgyro_exp%s_user%s.txt' % (paths_dict['uci-har'], os.sep, os.sep, keys[0], keys[1])
            acc_np = pd.read_csv(acc_file_name, sep=' ', header=None).values[int(keys[3]):(int(keys[4]) + 1)]
            gyr_np = pd.read_csv(gyr_file_name, sep=' ', header=None).values[int(keys[3]):(int(keys[4]) + 1)]

            for index in range(0, acc_np.shape[0] - seq_length + 1, seq_length // 2):
                x_acc_u1.append(acc_np[index:(index + seq_length), :])
                x_gyr_u1.append(gyr_np[index:(index + seq_length), :])
                y_u1.append([device_list.index('smartphone'), position_list.index('waist'), activity - 1])
                u_u1.append(int(keys[1])-1)

    acc_u1_data = np.array(x_acc_u1, dtype=float)
    gyr_u1_data = np.array(x_gyr_u1, dtype=float)
    y_u1_data = np.array(y_u1)[:, 2]
    u_u1_data = np.array(u_u1)
    acc_file_tv = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    gyr_file_tv = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    y_file_tv = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    u_file_tv = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    acc_file_ts = '{}{}x_acc_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    gyr_file_ts = '{}{}x_gyr_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    y_file_ts = '{}{}y_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    u_file_ts = '{}{}u_{}_{}_{}_{}_train'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    print(np.unique(y_u1_data , return_counts=True))
    print(np.unique(u_u1_data , return_counts=True))
    indices = range(y_u1_data.shape[0])
    ind_la, ind_ts = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=y_u1_data)
    print('Saving data to' + paths_dict[dataset_name])
    np.save(acc_file_tv, acc_u1_data[ind_la])
    np.save(gyr_file_tv, gyr_u1_data[ind_la])
    np.save(y_file_tv, y_u1_data[ind_la])
    np.save(u_file_tv, u_u1_data[ind_la])
    np.save(acc_file_ts,acc_u1_data[ind_ts])
    np.save(gyr_file_ts,gyr_u1_data[ind_ts])
    np.save(y_file_ts,y_u1_data[ind_ts])
    np.save(u_file_ts,u_u1_data[ind_ts])

def load_saved_data(dataset_name, bins, images, seq_length=100, gyro=False, preprocess=None):
    acc_file_tv = '{}{}x_acc_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    gyr_file_tv = '{}{}x_gyr_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    y_file_tv = '{}{}y_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    u_file_tv = '{}{}u_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'labeled',bins,images)
    acc_file_un = '{}{}x_acc_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    gyr_file_un = '{}{}x_gyr_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    y_file_un = '{}{}y_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    u_file_un = '{}{}u_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'unlabeled',bins,images)
    acc_file_ts = '{}{}x_acc_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    gyr_file_ts = '{}{}x_gyr_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    y_file_ts = '{}{}y_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)
    u_file_ts = '{}{}u_{}_{}_{}_{}_train.npy'.format(paths_dict[dataset_name], os.sep, seq_length,'test',bins,images)

    print(acc_file_tv)

    if os.path.isfile(acc_file_tv) and os.path.isfile(gyr_file_tv) and os.path.isfile(y_file_tv) and os.path.isfile(u_file_tv)\
            and os.path.isfile(acc_file_un) and os.path.isfile(gyr_file_un) and os.path.isfile(y_file_un) and os.path.isfile(u_file_un)\
            and os.path.isfile(acc_file_ts) and os.path.isfile(gyr_file_ts) and os.path.isfile(y_file_ts) and os.path.isfile(u_file_ts):
        print('Loading data from ' + paths_dict[dataset_name])
        acc_data_labeled = np.load(acc_file_tv)
        acc_data_labeled = preprocess_data(acc_data_labeled, preprocess)
        acc_data_unlabeled = np.load(acc_file_un)
        acc_data_unlabeled = preprocess_data(acc_data_unlabeled, preprocess)
        acc_data_test = np.load(acc_file_ts)
        acc_data_test = preprocess_data(acc_data_test, preprocess)
        print('acc_data_labeled',acc_data_labeled.shape)
        print('acc_data_unlabeled',acc_data_unlabeled.shape)
        print('acc_data_test',acc_data_test.shape)

        gyr_data_labeled = np.load(gyr_file_tv)
        gyr_data_labeled = preprocess_data(gyr_data_labeled, preprocess)
        gyr_data_unlabeled = np.load(gyr_file_un)
        gyr_data_unlabeled = preprocess_data(gyr_data_unlabeled, preprocess)
        gyr_data_test = np.load(gyr_file_ts)
        gyr_data_test = preprocess_data(gyr_data_test, preprocess)
        print('gyr_data_labeled',gyr_data_labeled.shape)
        print('gyr_data_unlabeled',gyr_data_unlabeled.shape)
        print('gyr_data_test',gyr_data_test.shape)

        y_labeled_train = np.load(y_file_tv)
        u_labeled_train = np.load(u_file_tv)
        y_unlabeled_train = np.load(y_file_un)
        u_unlabeled_train = np.load(u_file_un)
        y_test = np.load(y_file_ts)
        u_test = np.load(u_file_ts)

        return Dataset(acc_data_labeled,gyr_data_labeled,y_labeled_train,u_labeled_train,
                       acc_data_unlabeled,gyr_data_unlabeled,y_unlabeled_train,u_unlabeled_train,
                       acc_data_test,gyr_data_test,y_test,u_test)
    return None

def __standardize(data):
    print('standardizing data...')
    data_mean = np.mean(data)
    data_std = np.std(data)
    return (data - data_mean) / data_std

def __scale(data):
    print('scaling data...')
    for i in range(data.shape[2]):
        min_tri = np.min(data[:, :, i])
        min_tsi = np.min(data[:, :, i])
        data[:, :, i] = data[:, :, i] - min_tri
        data[:, :, i] = data[:, :, i] - min_tsi
        max_tri = np.max(data[:, :, i])
        max_tsi = np.max(data[:, :, i])
        data[:, :, i] = data[:, :, i] / max_tri
        data[:, :, i] = data[:, :, i] / max_tsi

    return data


