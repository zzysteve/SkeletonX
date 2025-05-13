# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import os.path as osp
import numpy as np
import pickle
import logging
import h5py
from sklearn.model_selection import train_test_split
import pandas as pd

root_path = './'
stat_path = osp.join(root_path, 'statistics')
setup_file = osp.join(stat_path, 'setup.txt')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'


if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]

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


def frame_translation(skes_joints, skes_name, frames_cnt):
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


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
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, ske_joints))
            # aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 60))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_train_val(train_indices, method='sklearn', ratio=0.05):
    """
    Get validation set by splitting data randomly from training set with two methods.
    In fact, I thought these two methods are equal as they got the same performance.

    """
    if method == 'sklearn':
        return train_test_split(train_indices, test_size=ratio, random_state=10000)
    else:
        np.random.seed(10000)
        np.random.shuffle(train_indices)
        val_num_skes = int(np.ceil(0.05 * len(train_indices)))
        val_indices = train_indices[:val_num_skes]
        train_indices = train_indices[val_num_skes:]
        return train_indices, val_indices


def split_dataset(skes_joints, label, performer, setup, evaluation, save_path, skes_name):
    aux_indices, anchor_indices, eval_indices = get_indices(label, skes_name, evaluation)
    print(f"aux_indices:{len(aux_indices)}, anchor_indices:{len(anchor_indices)}, eval_indices:{len(eval_indices)}")

    # Save labels and num_frames for each sequence of each data set
    aux_labels = label[aux_indices]
    anchor_labels = label[anchor_indices]
    eval_labels = label[eval_indices]

    aux_x = skes_joints[aux_indices]
    aux_y = one_hot_vector(aux_labels)
    aux_name = skes_name[aux_indices]

    anchor_x = skes_joints[anchor_indices]
    anchor_y = one_hot_vector(anchor_labels)
    anchor_name = skes_name[anchor_indices]

    eval_x = skes_joints[eval_indices]
    eval_y = one_hot_vector(eval_labels)
    eval_name = skes_name[eval_indices]

    save_name = 'NTU60_%s.npz' % evaluation
    np.savez(save_name, x_aux=aux_x, y_aux=aux_y, name_aux=aux_name, x_anchor=anchor_x, y_anchor=anchor_y,
             name_anchor=anchor_name, x_eval=eval_x, y_eval=eval_y, name_eval=eval_name)


def get_indices(in_label, skes_name, evaluation='1Shot'):
    eval_indices = np.empty(0)
    aux_indices = np.empty(0)
    anchor_indices = np.empty(0)

    if evaluation == '1Shot':
        eval_ids = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55]
        aux_ids = [i for i in range(1, 61) if i not in eval_ids]
        anchor_names = [b"S001C003P008R001A001", b"S001C003P008R001A007", b"S001C003P008R001A013", b"S001C003P008R001A019",
                       b"S001C003P008R001A025", b"S001C003P008R001A031", b"S001C003P008R001A037", b"S001C003P008R001A043",
                       b"S001C003P008R001A049", b"S001C003P008R001A055"]

        for idx in aux_ids:
            temp = np.where(in_label == (idx - 1))[0]
            aux_indices = np.hstack((aux_indices, temp)).astype(np.int)

        for name in anchor_names:
            temp = np.where(skes_name == name)[0]
            anchor_indices = np.hstack((anchor_indices, temp)).astype(np.int)

        for idx in eval_ids:
            temp = np.where(in_label == (idx - 1))[0]
            eval_indices = np.hstack((eval_indices, temp)).astype(np.int)

        for idx in anchor_indices:
            eval_indices = np.delete(eval_indices, np.where(eval_indices == idx)[0])

        # load columns from different txt files
        camera = pd.read_csv(camera_file, header=None, names=['camera'])
        label = pd.read_csv(label_file, header=None, names=['label'])
        performer = pd.read_csv(performer_file, header=None, names=['performer'])
        replication = pd.read_csv(replication_file, header=None, names=['replication'])
        setup = pd.read_csv(setup_file, header=None, names=['setup'])
        index = pd.DataFrame(np.arange(0, len(label)), columns=['index'])        
        # reduce label by 1
        label.label = label.label - 1

        # assert in_label is the same as label
        assert np.all(in_label == label.label.values)

        # merge all columns into one dataframe
        df = pd.concat([camera, performer, setup, replication, label, index], axis=1)
        
        aux_df = df.iloc[aux_indices]
        anchor_df = df.iloc[anchor_indices]
        eval_df = df.iloc[eval_indices]

        # reset column 'index' in aux_df, anchor_df, eval_df from 0
        aux_df['index'] = np.arange(0, len(aux_df))
        anchor_df['index'] = np.arange(0, len(anchor_df))
        eval_df['index'] = np.arange(0, len(eval_df))

        aux_df.to_csv('one-shot_aux_info.csv', index=False)
        anchor_df.to_csv('one-shot_anchor_info.csv', index=False)
        eval_df.to_csv('one-shot_eval_info.csv', index=False)


    return aux_indices, anchor_indices, eval_indices


if __name__ == '__main__':
    setup = np.loadtxt(setup_file, dtype=np.int)  # camera id: 1~32
    performer = np.loadtxt(performer_file, dtype=np.int)  # subject id: 1~106
    label = np.loadtxt(label_file, dtype=np.int) - 1  # action label: 0~119

    frames_cnt = np.loadtxt(frames_file, dtype=np.int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)

    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length

    evaluations = ['1Shot']
    for evaluation in evaluations:
        split_dataset(skes_joints, label, performer, setup, evaluation, save_path, skes_name)
