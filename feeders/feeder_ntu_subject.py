import numpy as np

from torch.utils.data import Dataset

from feeders import tools

import pandas as pd

ntu120_class_name = [
    "A1. drink water", "A2. eat meal/snack", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat/cap",
    "A21. take off a hat/cap", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call/answer phone", "A29. playing with phone/tablet",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head/bow", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze/cough", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache/heart pain)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)/feeling warm", "A50. punching/slapping other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other", "A61. put on headphone",
    "A62. take off headphone", "A63. shoot at the basket", "A64. bounce ball", "A65. tennis bat swing", "A66. juggling table tennis balls",
    "A67. hush (quite)", "A68. flick hair", "A69. thumb up", "A70. thumb down", "A71. make ok sign", "A72. make victory sign",
    "A73. staple book", "A74. counting money", "A75. cutting nails", "A76. cutting paper (using scissors)", "A77. snapping fingers",
    "A78. open bottle", "A79. sniff (smell)", "A80. squat down", "A81. toss a coin", "A82. fold paper", "A83. ball up paper",
    "A84. play magic cube", "A85. apply cream on face", "A86. apply cream on hand back", "A87. put on bag", "A88. take off bag",
    "A89. put something into a bag", "A90. take something out of a bag", "A91. open a box", "A92. move heavy objects", "A93. shake fist",
    "A94. throw up cap/hat", "A95. hands up (both hands)", "A96. cross arms", "A97. arm circles", "A98. arm swings", "A99. running on the spot",
    "A100. butt kicks (kick backward)", "A101. cross toe touch", "A102. side kick", "A103. yawn", "A104. stretch oneself",
    "A105. blow nose", "A106. hit other person with something", "A107. wield knife towards other person",
    "A108. knock over other person (hit with body)", "A109. grab other person's stuff", "A110. shoot at other person with a gun",
    "A111. step on foot", "A112. high-five", "A113. cheers and drink", "A114. carry something with other person",
    "A115. take a photo of other person", "A116. follow other person", "A117. whisper in other person's ear",
    "A118. exchange things with other person", "A119. support somebody with hand", "A120. finger-guessing game (playing rock-paper-scissors)"
]

ntu120_class_name_short = [
    "A1. drink water", "A2. eat meal", "A3. brushing teeth", "A4. brushing hair", "A5. drop", "A6. pickup",
    "A7. throw", "A8. sitting down", "A9. standing up (from sitting position)", "A10. clapping", "A11. reading",
    "A12. writing", "A13. tear up paper", "A14. wear jacket", "A15. take off jacket", "A16. wear a shoe",
    "A17. take off a shoe", "A18. wear on glasses", "A19. take off glasses", "A20. put on a hat",
    "A21. take off a hat", "A22. cheer up", "A23. hand waving", "A24. kicking something", "A25. reach into pocket",
    "A26. hopping (one foot jumping)", "A27. jump up", "A28. make a phone call", "A29. playing with phone",
    "A30. typing on a keyboard", "A31. pointing to something with finger", "A32. taking a selfie", "A33. check time (from watch)",
    "A34. rub two hands together", "A35. nod head", "A36. shake head", "A37. wipe face", "A38. salute", "A39. put the palms together",
    "A40. cross hands in front (say stop)", "A41. sneeze", "A42. staggering", "A43. falling", "A44. touch head (headache)",
    "A45. touch chest (stomachache)", "A46. touch back (backache)", "A47. touch neck (neckache)", "A48. nausea or vomiting condition",
    "A49. use a fan (with hand or paper)", "A50. punching other person", "A51. kicking other person",
    "A52. pushing other person", "A53. pat on back of other person", "A54. point finger at the other person",
    "A55. hugging other person", "A56. giving something to other person", "A57. touch other person's pocket",
    "A58. handshaking", "A59. walking towards each other", "A60. walking apart from each other", "A61. put on headphone",
    "A62. take off headphone", "A63. shoot at the basket", "A64. bounce ball", "A65. tennis bat swing", "A66. juggling table tennis balls",
    "A67. hush (quite)", "A68. flick hair", "A69. thumb up", "A70. thumb down", "A71. make ok sign", "A72. make victory sign",
    "A73. staple book", "A74. counting money", "A75. cutting nails", "A76. cutting paper (using scissors)", "A77. snapping fingers",
    "A78. open bottle", "A79. sniff (smell)", "A80. squat down", "A81. toss a coin", "A82. fold paper", "A83. ball up paper",
    "A84. play magic cube", "A85. apply cream on face", "A86. apply cream on hand back", "A87. put on bag", "A88. take off bag",
    "A89. put something into a bag", "A90. take something out of a bag", "A91. open a box", "A92. move heavy objects", "A93. shake fist",
    "A94. throw up cap", "A95. hands up (both hands)", "A96. cross arms", "A97. arm circles", "A98. arm swings", "A99. running on the spot",
    "A100. butt kicks (kick backward)", "A101. cross toe touch", "A102. side kick", "A103. yawn", "A104. stretch oneself",
    "A105. blow nose", "A106. hit other person with something", "A107. wield knife towards other person",
    "A108. knock over other person (hit with body)", "A109. grab other person's stuff", "A110. shoot at other person with a gun",
    "A111. step on foot", "A112. high-five", "A113. cheers and drink", "A114. carry something with other person",
    "A115. take a photo of other person", "A116. follow other person", "A117. whisper in other person's ear",
    "A118. exchange things with other person", "A119. support somebody with hand", "A120. finger-guessing game (playing rock-paper-scissors)"
]


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False, class_group=None, train_indices_info_path=None):
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
        self.train_indices_info_path = train_indices_info_path
        self.DASP_same_action_cnt = 0
        self.SADP_same_performer_cnt = 0
        if normalization:
            self.get_mean_map()

        if class_group is not None and class_group.startswith('os'):
            # extract N
            try:
                osN = int(class_group[2:])
            except:
                raise ValueError("class_group {} is not valid!".format(class_group))
            # select the first osN classes exclude A(6i), i in [0, 19]
            select_index = []
            select_class = [i for i in range(120) if i % 6 != 0]
            select_class = select_class[:osN]
            print("[Dataset] Selected classes {} due to class_group {}".format(select_class, class_group))
            
            print("Minimum value in label: {}".format(np.min(self.label)))
            
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
        self.valid_frames = self.valid_frames.astype(np.int)

        # print samples per class
        cnt = np.zeros(120)
        for label in self.label:
            cnt[label] += 1
        print("[Dataset] Split: {}. Samples per class: {}".format(self.split, cnt))

        # Subject disentanglement: read train indices info in csv format using pandas, and build quick access
        if self.split == 'train' or self.split == 'aux' or self.split == 'anchor' or (self.split == 'test' and train_indices_info_path is not None):
            train_indices_info = pd.read_csv(self.train_indices_info_path)
            # reassign train_indices_info if select_index exists
            try:
                # resample train_indices_info, and set index to start from 0
                print("[Dataset] Resampling train_indices_info")
                train_indices_info = train_indices_info.iloc[select_index]
                train_indices_info['index'] = np.arange(len(train_indices_info))
                train_indices_info.reset_index(inplace=True, drop=True)
                print(train_indices_info)
            except NameError:
                pass
            # sort by performer
            train_indices_info = train_indices_info.sort_values(by=['index'])
            self.performer = train_indices_info['performer'].values
            # assert max label in train_indices_info is equal to max label in label
            assert self.split == 'aux' or self.split == 'anchor' \
            or np.max(train_indices_info['label']) == np.max(self.label), "max label in train_indices_info is not equal to max label in label!"

            # sanity check on label
            if 'label' in train_indices_info.columns:
                # print("[Debug] self.label: {}".format(self.label))
                # print("[Debug] train_indices_info['label'].values: {}".format(train_indices_info['label'].values))
                matched_label = train_indices_info['label'].values == self.label
                # print("[Debug] matched_label: {}".format(matched_label))
                # print("[Dataset] label matched: {}/{}".format(np.sum(matched_label), len(matched_label)))
                if np.sum(matched_label) != len(matched_label):
                    print("[Dataset] first unmatched label: {}".format(self.label[np.where(matched_label == False)[0][0]]))
                # assert the length of label in train_indices_info is equal to label
                assert len(train_indices_info['label'].values) == len(self.label), "length of label in train_indices_info is not equal to label!"
                assert np.all(train_indices_info['label'].values == self.label), "label in train_indices_info is not equal to label in label!"
            else:
                raise ValueError("label is not in train_indices_info.columns!")
            # build quick access dict for performer
            self.performer_cls_dict = {}
            dedup_performers = list(set(self.performer))
            print("[Debug] Split: {}. dedup_performers: {}".format(self.split, dedup_performers))
            for perf in dedup_performers:
                self.performer_cls_dict[perf] = {}
                # select performer == perf in train_indices_info
                perf_data = train_indices_info[train_indices_info['performer'] == perf]
                actions = list(set(perf_data['label']))
                for action in actions:
                    self.performer_cls_dict[perf][action] = list(perf_data[perf_data['label'] == action]['index'])

            # build quick access dict for class
            self.cls_performer_dict = {}
            dedup_classes = list(set(self.label))
            print("[Debug] Split: {}. dedup_classes: {}".format(self.split, dedup_classes))
            for cls in dedup_classes:
                self.cls_performer_dict[cls] = {}
                # select label == cls in train_indices_info
                cls_data = train_indices_info[train_indices_info['label'] == cls]
                performers = list(set(cls_data['performer']))
                for performer in performers:
                    self.cls_performer_dict[cls][performer] = list(cls_data[cls_data['performer'] == performer]['index'])

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
    
    def data_aug(self, data_numpy, index):
        valid_frame_num = self.valid_frames[index]
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        return data_numpy

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        data_numpy = self.data_aug(data_numpy, index)
        # reshape Tx(MVC) to CTVM
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
    

    def get_subject(self, index):
        # input: mini-batch index
        # output: subject id
        return self.performer[index]
    
    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_pair_data(self, indexes, mode='SASP', in_performer=None, in_labels=None):
        '''
        The input is a batch of data, and the output is also batch of data.
        '''
        batch_data_numpy = []
        batch_label = []
        batch_idx = []
        for index in indexes:
            if in_performer is None:
                if mode == 'SASP':
                    # SASP: same action, same performer
                    performer = self.performer[index]
                    label = self.label[index]
                elif mode == 'SADP':
                    # SADP: same action, different performer
                    label = self.label[index]
                    performers = list(self.cls_performer_dict[label])
                    performers.remove(self.performer[index])
                    if len(performers) == 0:
                        performer = self.performer[index]
                        self.SADP_same_performer_cnt += 1
                    else:
                        performer = np.random.choice(performers)
                elif mode == 'DASP':
                    # DASP: different action, same performer
                    performer = self.performer[index]
                    labels = list(self.performer_cls_dict[performer])
                    labels.remove(self.label[index])
                    if len(labels) == 0:
                        label = self.label[index]
                        self.DASP_same_action_cnt += 1
                    else:
                        label = np.random.choice(labels)
                else:
                    raise ValueError("mode {} is not supported!".format(mode))
            else:
                assert mode == 'AnchorAug'
                # Only support AnchorAug for now
                # draw sample from other performer, does not matter if it is the same action or not
                performers = list(self.performer_cls_dict.keys())
                performers.remove(in_performer)
                
                performer = np.random.choice(performers)
                labels = list(self.performer_cls_dict[performer])
                label = np.random.choice(labels)

            if self.performer_cls_dict[performer][label] == []:
                raise ValueError("performer {}, label {} has no samples!".format(performer, label))
            pair_idx = np.random.choice(self.performer_cls_dict[performer][label])
            pair_data_numpy = np.array(self.data[pair_idx])
            pair_label = self.label[pair_idx]
            pair_data_numpy = self.data_aug(pair_data_numpy, pair_idx)
            batch_data_numpy.append(pair_data_numpy)
            batch_label.append(pair_label)
            batch_idx.append(pair_idx)
        
        # stack batch data
        batch_data_numpy = np.stack(batch_data_numpy, axis=0)
        batch_label = np.stack(batch_label, axis=0)
        batch_idx = np.stack(batch_idx, axis=0)
        # print("[DEBUG] shape of batch_idx: {}".format(batch_idx.shape))
        # print("[DEBUG] shape of original index: {}".format(indexes.shape))
        # print("[DEBUG] original label: {}".format(self.label[indexes]))
        # print("[DEBUG] pair label: {}".format(batch_label))
        return batch_data_numpy, batch_label, batch_idx
    

    def output_pair_sample_stat(self) -> str:
        return f'[Dataset] SASP_same_action_cnt: {self.DASP_same_action_cnt}, SADP_same_performer_cnt: {self.SADP_same_performer_cnt}'

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
