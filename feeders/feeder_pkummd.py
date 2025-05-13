import numpy as np

from torch.utils.data import Dataset

from feeders import tools

# Temporarily hardcode the class names
pkummdv2_class_name = [f"A{i+1}" for i in range(41)]

class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, class_group=None):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if self.bone is not False:
            raise NotImplementedError("bone modality in __getitem__ is not working now!")
        if self.vel is not False:
            raise NotImplementedError("vel modality in __getitem__ is not working now!")
        if normalization:
            self.get_mean_map()
        
        if class_group is not None and class_group.startswith('os'):
            # extract N
            try:
                osN = int(class_group[2:])
            except:
                raise ValueError("class_group {} is not valid!".format(class_group))
            # select the first osN classes exclude A(6i+1), i in [0, 19]
            select_index = []
            select_class = []
            one_shot_class = [4,9,12,15,18,20,25,29,34,39]
            for i in range(41):
                if i not in one_shot_class:
                    select_class.append(i)
            select_class = select_class[:osN]
            print("[Dataset] Selected classes {} due to class_group {}".format(select_class, class_group))
            
            for id, label in enumerate(self.label):
                if label in select_class:
                    select_index.append(id)
            self.data = self.data[select_index]
            self.label = self.label[select_index]
        elif class_group is not None:
            select_index = []
            select_class = class_group
            for id, label in enumerate(self.label):
                if label in select_class:
                    select_index.append(id)
            self.data = self.data[select_index]
            self.label = self.label[select_index]

        
        # calculate valid frames
        self.valid_frames = np.zeros(len(self.data))
        for i in range(len(self.data)):
            # print("[Debug] Calculating valid frames: {}/{}".format(i, len(self.data)))
            # print("[Debug] Data shape: {}".format(self.data[i].shape))
            self.valid_frames[i] = np.sum(self.data[i].sum(0).sum(-1).sum(-1) != 0)
        self.valid_frames = self.valid_frames.astype(int)

        # print samples per class
        cnt = np.zeros(41)
        for label in self.label:
            cnt[label] += 1
        print("[Dataset] Split: {}. Samples per class: {}".format(self.split, cnt))


    def load_data(self):
        # data: N C T V M   (output)
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        elif self.split == 'aux':
            self.data = npz_data['x_aux']
            self.label = np.where(npz_data['y_aux'] > 0)[1]
            self.sample_name = ['aux_' + str(i) for i in range(len(self.data))]
        elif self.split == 'anchor':
            self.data = npz_data['x_anchor']
            self.label = np.where(npz_data['y_anchor'] > 0)[1]
            self.sample_name = ['anchor_' + str(i) for i in range(len(self.data))]
        elif self.split == 'eval':
            self.data = npz_data['x_eval']
            self.label = np.where(npz_data['y_eval'] > 0)[1]
            self.sample_name = ['eval_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = self.valid_frames[index]
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        # This is supported by helper functions in the main.py
        # if self.bone:
        #     from .bone_pairs import ntu_pairs
        #     bone_data_numpy = np.zeros_like(data_numpy)
        #     for v1, v2 in ntu_pairs:
        #         bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        #     data_numpy = bone_data_numpy
        # if self.vel:
        #     data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        #     data_numpy[:, -1] = 0
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
