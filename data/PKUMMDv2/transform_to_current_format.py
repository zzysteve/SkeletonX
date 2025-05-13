import pandas
import numpy as np
import logging
import os.path as osp
import pickle as pkl
import os

CSV_file = 'output/PKU-MMDv2.csv'
numpy_dir = 'output/numpy'
output_dir = 'statistics'

# make directories
os.makedirs(output_dir, exist_ok=True)

# setup a logger
logger = logging.getLogger('running_log')
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(osp.join(output_dir, 'log.txt')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(output_dir, 'missing_skes.log')))
missing_skes_logger.info('{:^20}\t{}\t{}'.format('Skeleton', 'num_frames', 'num_missing'))

missing_skes_logger1 = logging.getLogger('missing_frames_1')
missing_skes_logger1.setLevel(logging.INFO)
missing_skes_logger1.addHandler(logging.FileHandler(osp.join(output_dir, 'missing_skes_1.log')))
missing_skes_logger1.info('{:^20}\t{}\t{}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1',
                                                              'Actor2', 'Start', 'End'))

missing_skes_logger2 = logging.getLogger('missing_frames_2')
missing_skes_logger2.setLevel(logging.INFO)
missing_skes_logger2.addHandler(logging.FileHandler(osp.join(output_dir, 'missing_skes_2.log')))
missing_skes_logger2.info('{:^20}\t{}\t{}\t{}'.format('Skeleton', 'num_frames', 'Actor1', 'Actor2'))

missing_count = 0

label_map = {}

def remove_missing_frames(ske_name, joints):
    """
    Cut off missing frames which all joints positions are 0s

    For the sequence with 2 actors' data, also record the number of missing frames for
    actor1 and actor2, respectively (for debug).
    """
    num_frames = joints.shape[0]
    num_bodies = joints.shape[1] // 75

    if num_bodies == 2:  # DEBUG
        missing_indices_1 = np.where(joints[:, :75].sum(axis=1) == 0)[0]
        missing_indices_2 = np.where(joints[:, 75:].sum(axis=1) == 0)[0]
        cnt1 = len(missing_indices_1)
        cnt2 = len(missing_indices_2)

        start = 1 if 0 in missing_indices_1 else 0
        end = 1 if num_frames - 1 in missing_indices_1 else 0
        if max(cnt1, cnt2) > 0:
            if cnt1 > cnt2:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}\t{:^5d}\t{:^3d}'.format(ske_name, num_frames,
                                                                            cnt1, cnt2, start, end)
                missing_skes_logger1.info(info)
            else:
                info = '{}\t{:^10d}\t{:^6d}\t{:^6d}'.format(ske_name, num_frames, cnt1, cnt2)
                missing_skes_logger2.info(info)

    # Find valid frame indices that the data is not missing or lost
    # For two-subjects action, this means both data of actor1 and actor2 is missing.
    valid_indices = np.where(joints.sum(axis=1) != 0)[0]  # 0-based index
    missing_indices = np.where(joints.sum(axis=1) == 0)[0]
    num_missing = len(missing_indices)

    if num_missing > 0:  # Update joints
        joints = joints[valid_indices]
        global missing_count
        missing_count += 1
        missing_skes_logger.info('{}\t{:^10d}\t{:^11d}'.format(ske_name, num_frames, num_missing))

    return joints

if __name__ == '__main__':
    # Load the CSV file
    df = pandas.read_csv(CSV_file)

    # Drop columns with num_person == 0
    df = df[df['num_person'] != 0]

    # Load the numpy files
    skes_numpy = []
    
    skes_filenames = df['ske_index'].values
    labels = df['action'].values
    performers = df['subject'].values
    cameras = df['camera_id'].values
    num_persons = df['num_person'].values
    original_filenames = df['original_filename'].values

    samples_with_missing_skes = []
    num_frames = []

    same_filename_action_count = 0
    last_filename = original_filenames[0]

    for ske_idx, ske_filename in enumerate(skes_filenames):
        if last_filename == original_filenames[ske_idx]:
            same_filename_action_count += 1
        else:
            same_filename_action_count = 1
            last_filename = original_filenames[ske_idx]
        original_filenames[ske_idx] = f"{original_filenames[ske_idx]}_{same_filename_action_count}"

        ske_numpy = np.load(f'{numpy_dir}/{ske_filename}.npy')
        # remove missing frames
        ske_numpy = remove_missing_frames(ske_filename, ske_numpy)
        num_frames.append(ske_numpy.shape[0])


        # sanity check
        if num_persons[ske_idx] == 0:
            logger.warning(f"Num_person is 0 for {ske_filename}")
            samples_with_missing_skes.append(ske_idx)
        elif num_persons[ske_idx] == 1:
            if ske_numpy.shape[1] != 75:
                logger.warning(f"Num_person is 1 but shape is {ske_numpy.shape[1]} for {ske_filename}")
                samples_with_missing_skes.append(ske_idx)
        elif num_persons[ske_idx] == 2:
            if ske_numpy.shape[1] != 150:
                logger.warning(f"Num_person is 2 but shape is {ske_numpy.shape[1]} for {ske_filename}")
                samples_with_missing_skes.append(ske_idx)
        else:
            logger.warning(f"Num_person is {num_persons[ske_idx]} for {ske_filename}")
            samples_with_missing_skes.append(ske_idx)

        skes_numpy.append(ske_numpy)
    
    # remove samples with missing skeletons in skes_numpy and df
    new_skes_numpy = np.delete(skes_numpy, samples_with_missing_skes)
    df = df.drop(samples_with_missing_skes)
    labels = df['action'].values
    performers = df['subject'].values
    cameras = df['camera_id'].values

    # map labels to continuous integers
    unique_labels = np.unique(labels)
    label_map = {label: i+1 for i, label in enumerate(unique_labels)}
    mapped_labels = [label_map[label] for label in labels]

    # save samples_with_missing_skes, labels, performers, cameras to txt, each line contains one element
    with open(f'{output_dir}/samples_with_missing_skes.txt', 'w') as f:
        for sample in samples_with_missing_skes:
            f.write(f"{sample}\n")
    with open(f'{output_dir}/original_label.txt', 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    with open(f'{output_dir}/label.txt', 'w') as f:
        for mapped_label in mapped_labels:
            f.write(f"{mapped_label}\n")
    with open(f'{output_dir}/performer.txt', 'w') as f:
        for performer in performers:
            f.write(f"{performer}\n")
    with open(f'{output_dir}/camera.txt', 'w') as f:
        for camera in cameras:
            f.write(f"{camera}\n")
    with open(f'{output_dir}/skes_available_name.txt', 'w') as f:
        for ske_filename in original_filenames:
            f.write(f"{ske_filename}\n")
    
    # save label_map in txt, with format "label: mapped_label\n"
    with open(f'{output_dir}/label_map.txt', 'w') as f:
        for label, mapped_label in label_map.items():
            f.write(f"{label}: {mapped_label}\n")
    
    # make directory
    os.makedirs('./denoised_data', exist_ok=True)

    # save skes_numpy using pickle
    with open(f'./denoised_data/raw_denoised_joints.pkl', 'wb') as f:
        pkl.dump(new_skes_numpy, f)
    
    # save num_frames
    with open(f'{output_dir}/frames_cnt.txt', 'w') as f:
        for num_frame in num_frames:
            f.write(f"{num_frame}\n")

    # final sanity check
    assert len(new_skes_numpy) == len(labels) == len(performers) == len(cameras) == len(num_frames)