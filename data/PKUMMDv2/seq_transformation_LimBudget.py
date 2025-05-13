# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

root_path = './'
stat_path = osp.join(root_path, 'statistics')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
label_file = osp.join(stat_path, 'label.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(stat_path, 'frames_cnt.txt')

save_path = './'


if not osp.exists(save_path):
    os.mkdir(save_path)

def seq_translation(skes_joints):
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences with the same frame length.

    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # 300
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            # aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, ske_joints))
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 120))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector

def split_dataset(skes_joints, label, performer, evaluation, save_path, num_samples_per_class=None):
    train_indices, test_indices = get_indices(performer, evaluation, num_samples_per_class)

    # Save labels and num_frames for each sequence of each data set
    if num_samples_per_class is not None:
        save_name = f'PKUMMDv2_{evaluation}_{num_samples_per_class}.npz'
    else:
        save_name = 'PKUMMDv2_%s.npz' % evaluation
    if train_indices is not None:
        train_labels = label[train_indices]
        train_x = skes_joints[train_indices]
        train_y = one_hot_vector(train_labels)
    if test_indices is not None:
        test_labels = label[test_indices]
        test_x = skes_joints[test_indices]
        test_y = one_hot_vector(test_labels)
    if train_indices is not None and test_indices is not None:
        np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)
    elif train_indices is not None:
        np.savez(save_name, x_train=train_x, y_train=train_y)
    elif test_indices is not None:
        np.savez(save_name, x_test=test_x, y_test=test_y)


def get_indices(performer, evaluation='CSub', num_samples_per_class=None):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

        
    # organize data using pandas
    # load columns from different txt files, e.g. camera.txt stands for column camera
    camera = pd.read_csv(camera_file, header=None, names=['camera'])
    label = pd.read_csv(label_file, header=None, names=['label'])
    performer = pd.read_csv(performer_file, header=None, names=['performer'])
    index = pd.DataFrame(np.arange(0, len(label)), columns=['index'])

    # merge all columns into one dataframe
    df = pd.concat([camera, performer, label, index], axis=1)

    # Debug information
    # find unique labels
    unique_labels = df['label'].unique()
    unique_labels.sort()
    print("Unique labels: ",len(unique_labels))

    if evaluation == 'CSub':  # Cross Subject (Subject IDs)
        train_ids = [1, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        test_ids = [2, 3, 7]

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(np.int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(np.int)
    elif evaluation == 'LimBudget':  # Limited Budget
        # Select num_samples_per_class samples for each class, following these protocols:
        # 1. select from least subjects
        train_ids = [1, 4, 5, 6, 8, 9, 10, 11, 12, 13]
        test_ids = [2, 3, 7]

        # organize data using pandas
        # load columns from different txt files, e.g. camera.txt stands for column camera
        camera = pd.read_csv(camera_file, header=None, names=['camera'])
        label = pd.read_csv(label_file, header=None, names=['label'])
        performer = pd.read_csv(performer_file, header=None, names=['performer'])
        index = pd.DataFrame(np.arange(0, len(label)), columns=['index'])
        
        # merge all columns into one dataframe
        df = pd.concat([camera, performer, label, index], axis=1)
        # drop rows not in train_ids
        df = df[df['performer'].isin(train_ids)]
        # sort data by performer
        df = df.sort_values(by=['performer'])

        # for each label, select num_samples_per_class samples
        for i in range(41):
            df_temp = df[df['label'] == i+1]
            if num_samples_per_class > len(df_temp):
                raise ValueError(f'num_samples_per_class for {i} should be less than the number of samples in each class')
            df_temp = df_temp[0:num_samples_per_class]
            df_temp = df_temp.sort_values(by=['performer'])
            train_indices = np.hstack((train_indices, df_temp.index.values)).astype(np.int)
        
        print('train_indices', train_indices.shape)
        print('train_indices:', train_indices)
        # sort train_indices, since the following re-indexing asserts that the indices are sorted
        train_indices = np.sort(train_indices)
        
        # store the selected samples' df as csv
        df_temp = df.loc[train_indices]
        df_temp.sort_values(by=['performer', 'label'], inplace=True)
        # Labels in csv starts from 0
        df_temp['label'] = df_temp['label'] - 1
        # copies row index to a new column named "original_index"
        df_temp['original_index'] = df_temp.index
        # sort by index and start from 0 again
        df_temp.sort_values(by=['index'], inplace=True)
        df_temp['index'] = np.arange(0, len(df_temp))

        # store the selected samples' df as csv
        df_temp.to_csv(f'train_indices_info_{num_samples_per_class}.csv', index=False)

        return train_indices, None
        # For debug
        # # count number of samples for each label
        # df_cnt = df.groupby('label').count()
        # # print the count of each label
        # print(df_cnt)
    elif evaluation == 'CSet':
        raise NotImplementedError('CSet is not implemented yet')
    return train_indices, test_indices
    


if __name__ == '__main__':
    performer = np.loadtxt(performer_file, dtype=np.int)  # subject id: 1~106
    label = np.loadtxt(label_file, dtype=np.int) - 1  # action label: 0~119

    frames_cnt = np.loadtxt(frames_file, dtype=np.int)  # frames_cnt

    # split_dataset(None, label, performer, 'LimBudget', save_path, num_samples_per_class=10)
    # exit()

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length

    for num_samples_per_class in [10, 20, 30, 50]:
        split_dataset(skes_joints, label, performer, 'LimBudget', save_path, num_samples_per_class=num_samples_per_class)
