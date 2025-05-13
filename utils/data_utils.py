from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
import sys
import traceback

def import_class(import_str):
    # Dynamic import
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros), dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
    r2 = torch.stack((zeros, ones, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
    ry = torch.cat((ry1, r2, ry3), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def calc_diff_modality(data, bone=False, vel=False):
    if data is None:
        return None
    if data.shape[3] != 25:
        # only work for ntu dataset
        return data
    if bone:
        from feeders.bone_pairs import ntu_pairs
        ret_data = torch.zeros_like(data).to(data.device)
        for v1, v2 in ntu_pairs:
            ret_data[:, :, :, v1 - 1] = data[:, :, :, v1 - 1] - data[:, :, :, v2 - 1]
        data = ret_data
    if vel:
        data[:, :, :-1] = data[:, :, 1:] - data[:, :, :-1]
        data[:, :, -1] = 0
    return data


def mixup_data(x, y, x_partner=None, y_partner=None, attn_map=None, attn_map_partner=None, alpha=1.0, mode='value_mix',
               lam=None, rand_index=None, bone_type='feeders.bone_pairs.ntu_upper'):
    # x: [N, C, T, V, M]
    # attn_map: [N, V]
    # Returns mixed inputs, pairs of targets, and lambda
    if mode == 'skeleton_mix':
        alpha = 1. / 16
    if lam is None:
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    if x_partner is None or y_partner is None:
        if rand_index is None:
            batch_size = x.size()[0]
            rand_index = torch.randperm(batch_size).to(x.device)
        x_partner = x[rand_index]
        y_partner = y[rand_index]
        if attn_map is not None:
            attn_map_partner = attn_map[rand_index]

    if mode == 'value_mix':
        mixed_x = lam * x + (1 - lam) * x_partner
    elif mode == 'value_mix_v2':
        # only mix different class
        for sample_id in range(len(x_partner)):
            rand_id = np.random.randint(len(x_partner))
            rand_cnt = 1
            while y[rand_id] == y[sample_id] and rand_cnt < 10:
                rand_id = np.random.randint(len(x_partner))
                rand_cnt += 1
            x_partner[sample_id] = x[rand_id]
            y_partner[sample_id] = y[rand_id]

        mixed_x = lam * x + (1 - lam) * x_partner
    elif mode == 'value_mix_v3':
        # only mix same class
        for sample_id in range(len(x_partner)):
            rand_id = np.random.randint(len(x_partner))
            rand_cnt = 1
            while y[rand_id] != y[sample_id] and rand_cnt < 10:
                rand_id = np.random.randint(len(x_partner))
                rand_cnt += 1
            x_partner[sample_id] = x[rand_id]
            y_partner[sample_id] = y[rand_id]

        mixed_x = lam * x + (1 - lam) * x_partner
    elif mode == 'pure_mix':
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    elif mode == 'frame_mix':
        T = x.size()[2]
        cut = int((1 - lam) * T)
        cen = np.random.randint(T)

        st = np.clip(cen - cut // 2, 0, T)
        ed = np.clip(cen + cut // 2, 0, T)

        mixed_x = x
        mixed_x[:, :, st:ed] = x_partner[:, :, st:ed]

        lam = 1 - (ed - st) / T
    elif mode == 'skeleton_mix':
        upper_bones = import_class(bone_type)
        T = x.size()[2]
        spine_id = 1

        if lam < 0.5:
            x, x_partner = x_partner, x
            rem_cnt = int((1-lam*2) * T)
        else:
            rem_cnt = int((lam*2-1) * T)

        mixed_x = x.clone()
        mixed_x[:, :, rem_cnt:, upper_bones] = x_partner[:, :, rem_cnt:, upper_bones] - x_partner[:, :, rem_cnt:, spine_id:spine_id+1] \
                                          + x[:, :, rem_cnt:, spine_id:spine_id+1]
    elif mode == 'HSke_mix':
        # x: [N, C, T, V, M]
        # attn_map: [N, V]
        from feeders.bone_pairs import ntu_groups, ske2group, group2ske
        x, x_partner = ske2group(x), ske2group(x_partner)
        mixed_x = x.clone()
        lam = torch.ones(x.size(0)).to(x.device)
        lam_partner = torch.zeros(x.size(0)).to(x.device)

        for item_id, (item_x, item_x_, item_attn, item_attn_) in enumerate(zip(x, x_partner, attn_map, attn_map_partner)):
            scores = []
            scores_ = []
            for group in ntu_groups:
                scores.append(torch.sum(item_attn[list(group)]).item())
                scores_.append(torch.sum(item_attn_[list(group)]).item())

            nums = [_ for _ in range(len(ntu_groups))]
            nums = sorted(nums, key=lambda x: scores_[x], reverse=True)

            for group_id in nums[:-1]:
                lam[item_id] -= scores[group_id]
                lam_partner[item_id] += scores_[group_id]
                mixed_x[item_id, :, :, ntu_groups[group_id], :] = x_partner[item_id, :, :, ntu_groups[group_id], :]
                if lam_partner[item_id] > .8:
                    break

        mixed_x = group2ske(mixed_x)

        return mixed_x, y, y_partner, lam, lam_partner
    elif mode == 'HSke_mix_demo':
        # x: [N, C, T, V, M]
        # attn_map: [N, V]
        from feeders.bone_pairs import ntu_groups, ske2group, group2ske
        x, x_partner = ske2group(x), ske2group(x_partner)
        mixed_x = x.clone()
        lam = torch.ones(x.size(0)).to(x.device)
        lam_partner = torch.zeros(x.size(0)).to(x.device)

        for item_id, (item_x, item_x_, item_attn, item_attn_) in enumerate(zip(x, x_partner, attn_map, attn_map_partner)):
            scores = []
            scores_ = []
            for group in ntu_groups:
                scores.append(torch.sum(item_attn[list(group)]).item())
                scores_.append(torch.sum(item_attn_[list(group)]).item())

            nums = [_ for _ in range(len(ntu_groups))]
            nums = sorted(nums, key=lambda x: scores_[x], reverse=True)

            threshold = np.random.uniform(0, 1)

            for group_id in nums[:-1]:
                lam[item_id] -= scores[group_id]
                lam_partner[item_id] += scores_[group_id]
                mixed_x[item_id, :, :, ntu_groups[group_id], :] = x_partner[item_id, :, :, ntu_groups[group_id], :]
                if lam_partner[item_id] > threshold:
                    break

        mixed_x = group2ske(mixed_x)

        return mixed_x, y, y_partner, lam, lam_partner

    elif mode == 'replace':
        lam = 0
        mixed_x = x_partner
    else:
        raise NotImplementedError

    return mixed_x, y, y_partner, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, lam_=None):
    if lam_ is None:
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    else:
        return lam * criterion(pred, y_a) + lam_ * criterion(pred, y_b)


class HardSampleMiner:
    def __init__(self, num_class=120, memory_size=10, alive_age=1000, use_fn=False, rank_reverse=False):
        self.num_class = num_class
        self.memory_size = memory_size
        self.alive_age = alive_age
        self.use_fn = use_fn
        self.rank_reverse = rank_reverse

        self.memory_bank = [[dict(step=-1, loss=-9e9)] for _ in range(self.num_class)]

    @torch.no_grad()
    def insert(self, x, y, scores, losses, step):
        if self.rank_reverse:
            losses = -losses.detach().clone()

        self.update(step)
        for item_data, item_label, item_score, item_loss in zip(x, y, scores, losses):
            pred_score, pred_label = torch.topk(item_score, 1)
            pred_score, pred_label = pred_score[0], pred_label[0]
            if not self.use_fn and pred_label != item_label:
                continue
            for index in range(len(self.memory_bank[item_label])):
                if self.memory_bank[item_label][index]['loss'] < item_loss:
                    self.memory_bank[item_label].insert(index, dict(data=item_data, loss=item_loss, step=step))
                    break
            while len(self.memory_bank[item_label]) > self.memory_size:  # maintain the size
                del self.memory_bank[item_label][-2]
                # self.memory_bank[item_label].pop(-2)

    @torch.no_grad()
    def query(self, x, y, scores):
        if self.rank_reverse:
            scores = -scores.detach().clone()

        x_partner = torch.zeros(x.size()).to(x.device)
        y_partner = torch.zeros(y.size()).long().to(y.device)
        for item_id, (item_data, item_label, item_score) in enumerate(zip(x, y, scores)):
            _, pred_label = torch.topk(item_score, 2)
            top_class = pred_label[0]
            if top_class == item_label:
                second_class = pred_label[1]
            else:
                second_class = top_class

            queue_size = len(self.memory_bank[second_class])
            if queue_size < 2:  # not enough queue
                x_partner[item_id] = item_data
                y_partner[item_id] = item_label
            else:
                select = np.random.randint(0, queue_size - 1)
                x_partner[item_id] = self.memory_bank[second_class][select]['data']
                y_partner[item_id] = second_class
        return x_partner, y_partner

    @torch.no_grad()
    def update(self, step):
        if self.alive_age == -1:
            return
        for class_id in range(self.num_class):
            tot = len(self.memory_bank[class_id])
            for index in range(tot - 2, -1, -1):  # reverse order, except the tail element
                if step - self.memory_bank[class_id][index]['step'] > self.alive_age:  # remove the old item
                    del self.memory_bank[class_id][index]
                    # self.memory_bank[class_id].pop(index)

    def __len__(self):
        ret = 0
        for _ in range(self.num_class):
            ret += len(self.memory_bank[_])
        return ret

    def hist(self):
        ret = []
        for _ in range(self.num_class):
            ret.append(len(self.memory_bank[_]))
        # print(f"Hist: {ret}")
        return ret


class HardSampleMinerBaseOnFeature(HardSampleMiner):
    def __init__(self, num_class=120, memory_size=10, alive_age=1000, use_fn=False, extend_class=False, rank_reverse=False):
        super(HardSampleMinerBaseOnFeature, self).__init__(num_class, memory_size, alive_age, use_fn, rank_reverse)

        self.memory_bank = [[dict(step=-1)] for _ in range(self.num_class)]
        self.extend_class = extend_class
        self.selected_sample_sim_mat = np.zeros((self.num_class, self.num_class))
        self.selected_sample_cls_mat = np.zeros((self.num_class, self.num_class))

    @torch.no_grad()
    def insert(self, x, y, scores, losses, step, feats=None):
        if feats is None:
            feats = scores
        if self.rank_reverse:
            scores = -scores.detach().clone()

        self.update(step)
        for item_data, item_label, item_score, item_feat in zip(x, y, scores, feats):
            pred_score, pred_label = torch.topk(item_score, 1)
            pred_score, pred_label = pred_score[0], pred_label[0]
            if not self.use_fn and pred_label != item_label:
                continue
            self.memory_bank[item_label].insert(0, dict(data=item_data, feat=item_feat, step=step))
            while len(self.memory_bank[item_label]) > self.memory_size:  # maintain the size
                del self.memory_bank[item_label][-2]
                # self.memory_bank[item_label].pop(-2)

    @torch.no_grad()
    def query(self, x, y, scores, feats=None):
        if feats is None:
            feats = scores

        x_partner = torch.zeros(x.size()).to(x.device)
        y_partner = torch.zeros(y.size()).long().to(y.device)
        for item_id, (item_data, item_label, item_score, item_feat) in enumerate(zip(x, y, scores, feats)):
            _, pred_label = torch.topk(item_score, 2)
            top_class = pred_label[0]
            if top_class == item_label:
                second_class = pred_label[1]
            else:
                second_class = top_class

            select, select_class, best_value = -1, -1, 0
            if not self.extend_class:
                select_class = second_class
                for index in range(len(self.memory_bank[second_class])-1):
                    feat = self.memory_bank[second_class][index]['feat']
                    sim_value = torch.sum(F.normalize(item_feat, dim=0) * F.normalize(feat.to(x.device), dim=0))
                    if select == -1 or sim_value > best_value:
                        best_value = sim_value
                        select = index
            else:
                for cls_id in range(self.num_class):
                    if cls_id == item_label:
                        continue
                    for index in range(len(self.memory_bank[cls_id])-1):
                        feat = self.memory_bank[cls_id][index]['feat']
                        sim_value = torch.sum(F.normalize(item_feat, dim=0) * F.normalize(feat.to(x.device), dim=0))
                        if select == -1 or sim_value > best_value:
                            best_value = sim_value
                            select = index
                            select_class = cls_id

            if select != -1:
                x_partner[item_id] = self.memory_bank[select_class][select]['data']
                y_partner[item_id] = select_class
            else:
                x_partner[item_id] = item_data
                y_partner[item_id] = item_label
        
            # record the feature similarity between x and x_partner
            # Note that y_cpu and y_partner_cpu can be the same, which means that there is no hard sample in second_class
            # This is allowed to see how many failed cases exsist for each class.
            y_cpu, y_partner_cpu = item_label.cpu().numpy(), y_partner[item_id].cpu().numpy()
            self.selected_sample_cls_mat[y_cpu, y_partner_cpu] += 1
            self.selected_sample_sim_mat[y_cpu, y_partner_cpu] += best_value
        return x_partner, y_partner
    
    @torch.no_grad()
    def get_sim_mat(self) -> np.ndarray:
        '''
        Print the similarity matrix of memory bank, calculate average similarity of each class
        '''
        sim_mat = torch.zeros(self.num_class, self.num_class)

        for cls_id in range(self.num_class):
            for index in range(len(self.memory_bank[cls_id])-1):
                feat = self.memory_bank[cls_id][index]['feat']
                for cls_id_ in range(self.num_class):
                    for index_ in range(len(self.memory_bank[cls_id_])-1):
                        feat_ = self.memory_bank[cls_id_][index_]['feat']
                        sim_value = torch.sum(F.normalize(feat.to(feat_.device), dim=0) * F.normalize(feat_, dim=0))
                        sim_mat[cls_id, cls_id_] += sim_value
            sim_mat[cls_id] /= len(self.memory_bank[cls_id]) - 1
        
        return sim_mat.cpu().numpy()

    
    @torch.no_grad()
    def get_selected_sample_info(self) -> Tuple[np.ndarray]:
        '''
        Return the selected sample similarity matrix and class matrix
        '''

        return self.selected_sample_sim_mat, self.selected_sample_cls_mat

    @torch.no_grad()
    def reset_selected_sample_info(self) -> None:
        '''
        Reset self.selected_sample_sim_mat and self.selected_sample_cls_mat
        '''
        self.selected_sample_sim_mat = np.zeros((self.num_class, self.num_class))
        self.selected_sample_cls_mat = np.zeros((self.num_class, self.num_class))
        


def get_knn_predict(embedding, anchor, metric='cosine', k=1):
    if metric == 'cosine':
        embedding = F.normalize(embedding, dim=-1)
        anchor = F.normalize(anchor, dim=-1)
        distance = torch.matmul(embedding, anchor.t())
        _, topk = distance.topk(k, dim=1)
        # return all topk index
        # print("[Debug] shape of topk:", topk.shape)
        # print("[Debug] shape of distance.max(dim=1)[1]: ", distance.max(dim=1)[1].shape)
        return topk
    elif metric == 'euclidean':
        distance = torch.cdist(embedding, anchor)
        _, topk = distance.topk(k, dim=1, largest=False)
        return topk
    elif metric == 'EMD':
        raise NotImplementedError
    
