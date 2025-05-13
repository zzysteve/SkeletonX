# This script directly transform PKU-MMDv2 raw skeleton data to npz file.

import os
import os.path as osp
import time
import numpy as np
import pandas as pd
import tqdm
from io import StringIO
import logging

root_dir = './'
label_dir = osp.join(root_dir, 'label')
skeleton_dir = osp.join(root_dir, 'skeletons')
output_dir = osp.join(root_dir, 'output')
camera_id_map = {'L': 1, 'M': 2, 'R': 3}
num_joints = 25
debug_skip_to = None
DEBUG = False

# make directories
os.makedirs(output_dir, exist_ok=True)
# setup a logger
logger = logging.getLogger('running_log')
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(osp.join(output_dir, 'running_log.txt')))

skes_list = []

output_csv = osp.join(output_dir, 'PKU-MMDv2.csv')
# if output_csv exists, move it to output_csv_<time>.csv
if osp.exists(output_csv):
    os.rename(output_csv, output_csv + '_' + str(time.time()) + '.csv')


def csv_write(data):
    with open(output_csv, 'a') as f:
        for line in data:
            f.write(line + '\n')

if __name__ == '__main__':
    # make dir
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    # list all files in label_dir
    label_files = os.listdir(label_dir)
    skes_action_index = 0
    # create a pandas dataframe to store information
    csv_write(['ske_index,A_origin,subject,action,camera_id,num_person,original_filename'])
    for (index, file) in tqdm.tqdm(enumerate(label_files)):
        if debug_skip_to and index < debug_skip_to:
            print("index: ", index, "file: ", file)
            continue
        # parse filename with format AaaNnn-B.txt
        # Aaa: Unknown yet, for example, 'A01'
        # Nnn: Subject id, for example, 'N01'
        # B: can be L/M/R, means left/middle/right camera
        file_name = file.split('.')[0]
        original_filename = file_name
        file_name = file_name.split('-')
        former_part, camera_id = file_name[0], camera_id_map[file_name[1][0]]
        Aaa, subject_id = former_part.split('N')
        original_A = Aaa[1:]
        subject_id = int(subject_id)
        original_A = int(original_A)
        if DEBUG:
            pass
            # print("File: ", file)

        # read label file
        with open(osp.join(label_dir, file), 'r') as f:
            action_lines = f.readlines()
        # read skeleton file
        with open(osp.join(skeleton_dir, file), 'r') as f:
            skeletons = f.readlines()
        print("File: ", file)
        # ONE action sequence
        for (action_index, action_line) in enumerate(action_lines):
            # print("[Debug] action_index: ", action_index)
            action_id, start, end = action_line.split(',')[:3]
            action_seq = skeletons[int(start):int(end)]
            action_seq_numpy = np.zeros((int(end)-int(start), 2 * 3 * num_joints))
            skip_first = False
            action_person_1_ever_exist = False
            action_person_2_ever_exist = False
            for action_frame_index, action_frame in enumerate(action_seq):
                # print("[Debug] action_frame_index: ", action_frame_index)
                action_frame = action_frame.strip()
                # using pandas to read line with 150 floats, split by space
                action_data = pd.read_csv(StringIO(action_frame), sep=' ', header=None)

                
                # split action_data into two persons, and transform to numpy array
                action_person_1 = action_data.values[:, 0:75]
                action_person_2 = action_data.values[:, 75:150]

                # count number of persons
                action_person_1_exist = False
                action_person_2_exist = False
                if np.sum(action_person_1) != 0.0:
                    action_person_1_exist = True
                    action_person_1_ever_exist = True
                if np.sum(action_person_2) != 0.0:
                    action_person_2_exist = True
                    action_person_2_ever_exist = True
                    if not action_person_1_exist:
                        logger.warning(f"Warning: action_person_2_exist == True, but action_person_1_exist == False, with filename: {file}")
                if action_person_1_exist + action_person_2_exist == 0:
                    # skip this frame
                    if skip_first:
                        logger.warning(f"Warning Empty frame is not the first frame, with filename: {file}@{action_frame_index}")
                        skip_first = True
                    continue
                # write to file in numpy, with shape of (num_frames, 2 * num_joints * 3)
                if action_person_1_exist:
                    action_seq_numpy[action_frame_index, 0:75] = action_person_1
                if action_person_2_exist:
                    action_seq_numpy[action_frame_index, 75:150] = action_person_2
            
            num_person = (action_person_1_ever_exist + action_person_2_ever_exist)
            csv_write([f'{skes_action_index},{original_A},{subject_id},{action_id},{camera_id},{num_person},{original_filename}'])
            # make numpy directory
            os.makedirs(osp.join(output_dir, 'numpy'), exist_ok=True)
            # save numpy array as skes_index.npy
            if action_person_1_ever_exist and action_person_2_ever_exist:
                np.save(osp.join(output_dir, 'numpy' ,f'{skes_action_index}.npy'), action_seq_numpy)
            elif action_person_1_ever_exist:
                np.save(osp.join(output_dir, 'numpy' ,f'{skes_action_index}.npy'), action_seq_numpy[:, 0:75])
            elif action_person_2_ever_exist:
                np.save(osp.join(output_dir, 'numpy' ,f'{skes_action_index}.npy'), action_seq_numpy[:, 75:150])
                logger.warning("Action_person_2_ever_exist, but action_person_1_ever_exist == False, with filename: ", file)
            else:
                logger.error("Action_person_1_ever_exist == False and action_person_2_ever_exist == False, with filename: ", file)
            skes_action_index += 1