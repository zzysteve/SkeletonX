from model.modules import *
from utils.cls_loss import *
from model.lib import ST_DecoupleNet, ST_FeatureAggrNet
from utils.data_utils import mixup_data


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):
    def __init__(self, gcn_model, loss_crit, aggr_mode, n_channel, n_frame, n_joint, n_person, **kwargs):
        super(Model, self).__init__()
        self.gcn_model = gcn_model
        self.loss_crit = loss_crit
        self.aggr_mode = aggr_mode
        self.n_channel = n_channel
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.n_person = n_person
        self.feat_aggr_net = ST_FeatureAggrNet(n_channel, n_frame, n_joint, n_person, aggr_mode, **kwargs)
        self.decouple_net = ST_DecoupleNet(n_channel, n_frame, n_joint, n_person, **kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        assert getattr(self, 'tb_writer', None) is not None, "tb_writer is None"
        self.weight_dict = getattr(self, 'weight_dict', None)
        self.abl_zeroout = getattr(self, 'abl_zeroout', None)
        self.eval_mode = getattr(self, 'eval_mode', None)

    def forward(self, x, x_dasp=None, x_sadp=None, label=None, label_dasp=None, label_sadp=None, epoch=None, eval=False, global_step=None, get_hidden_feat=False):
        '''
        A wrapper for forward function, including: a GCN model, a decouple module, a feature aggregation module.
        x: [N*M, C, T, V]
        x_dasp: [N*M, C, T, V]
        x_sadp: [N*M, C, T, V]

        return: [N, C]
        '''
        output = {}
        losses = {}

        label_dasp_mix = label_sadp_mix = None
        
        x = self.gcn_model.forward_to_hidden_feat(x)
        # spatial: [N*M, C, 1, V], temporal: [N*M, C, T, 1]
        subject_feat_x, action_feat_x = self.decouple_net(x)
        # normal forward
        x_feat = torch.cat((subject_feat_x.mean(3).squeeze(), action_feat_x.mean(2).squeeze()), dim=1)
        # transform from [N*M, C] -> [N, M, C] -> [N, C]
        x_feat = x_feat.view(-1, self.n_person, self.n_channel).mean(1)
        if self.abl_zeroout is not None:
            assert label_dasp is None and label_sadp is None, "label_dasp and label_sadp should be None when using abl_zeroout"
            if self.abl_zeroout == 'subject':
                x_feat[:, :self.n_channel//2] = 0
            elif self.abl_zeroout == 'action':
                x_feat[:, self.n_channel//2:] = 0
            else:
                raise ValueError("Unknown abl_zeroout: ", self.abl_zeroout)
        if get_hidden_feat:
            return x_feat
        x_out = self.gcn_model.forward_hidden_feat(x_feat)
        output['x'] = x_out
        
        if x_dasp is not None:
            x_dasp = self.gcn_model.forward_to_hidden_feat(x_dasp)
            subject_feat_x_dasp, action_feat_x_dasp = self.decouple_net(x_dasp)
            if self.DASP_mixup_ep != -1 and epoch >= self.DASP_mixup_ep:
                # mixup action feature
                action_feat_x_dasp_mix, _, _, lam = mixup_data(action_feat_x, label, action_feat_x_dasp, label_dasp, mode='value_mix')
                label_dasp_mix = lam * label + (1 - lam) * label_dasp
                # assemble feature
                assert self.aggr_mode == 'concat', "Only support concat mode"
                x_cross_mix = torch.cat((subject_feat_x.mean(3).squeeze(), action_feat_x_dasp_mix.mean(2).squeeze()), dim=1)
                x_cross_mix = x_cross_mix.view(-1, self.n_person, self.n_channel).mean(1)
                # output
                x_cross_mix_out = self.gcn_model.forward_hidden_feat(x_cross_mix)
                output['x_dasp_cross_mix'] = x_cross_mix_out
            # aggregation
            cross_a, cross_b = self.feat_aggr_net((subject_feat_x, action_feat_x), (subject_feat_x_dasp, action_feat_x_dasp))
            # original sample
            x_dasp_origin_feat = torch.cat((subject_feat_x_dasp.mean(3).squeeze(), action_feat_x_dasp.mean(2).squeeze()), dim=1)
            # transform from [N*M, C] -> [N, M, C] -> [N, C]
            x_dasp_origin_feat = x_dasp_origin_feat.view(-1, self.n_person, self.n_channel).mean(1)

            # output
            x_dasp_origin_out = self.gcn_model.forward_hidden_feat(x_dasp_origin_feat)
            x_cross_out_a = self.gcn_model.forward_hidden_feat(cross_a)
            x_cross_out_b = self.gcn_model.forward_hidden_feat(cross_b)
            output['x_dasp'] = x_dasp_origin_out
            output['x_dasp_cross_a'] = x_cross_out_a
            output['x_dasp_cross_b'] = x_cross_out_b
            if self.w_SP:
                if self.CA_mode == 'cosine':
                    # calculate cosine similarity, input shape [N, C], output is a scalar
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    person_cos_sim = cos(subject_feat_x_dasp, subject_feat_x)
                    losses['loss_SP'] = (self.w_SP * (1 - person_cos_sim)).mean()
                elif self.CA_mode == 'l2':
                    person_l2_dist = torch.norm(subject_feat_x_dasp - subject_feat_x, dim=1)
                    losses['loss_SP'] = (self.w_SP * person_l2_dist).mean()
                else:
                    raise ValueError("Unknown CA_mode: ", self.CA_mode)

        if x_sadp is not None:
            x_sadp = self.gcn_model.forward_to_hidden_feat(x_sadp)
            subject_feat_x_sadp, action_feat_x_sadp = self.decouple_net(x_sadp)
            if self.SADP_mixup_ep != -1 and epoch >= self.SADP_mixup_ep:
                # mixup subject feature, no label modified (same action)
                subject_feat_x_sadp, mixed_label_a, mixed_label_b,_ = mixup_data(subject_feat_x, label, subject_feat_x_sadp, label_sadp, mode='value_mix')
                label_sadp_mix = mixed_label_a
                # assemble feature
                assert self.aggr_mode == 'concat', "Only support concat mode"
                x_cross_mix = torch.cat((subject_feat_x_sadp.mean(3).squeeze(), action_feat_x.mean(2).squeeze()), dim=1)
                x_cross_mix = x_cross_mix.view(-1, self.n_person, self.n_channel).mean(1)
                # output
                x_cross_mix_out = self.gcn_model.forward_hidden_feat(x_cross_mix)
                output['x_sadp_cross_mix'] = x_cross_mix_out
            # aggregation
            cross_a, cross_b = self.feat_aggr_net((subject_feat_x, action_feat_x), (subject_feat_x_sadp, action_feat_x_sadp))
            # original sample
            x_sadp_origin_feat = torch.cat((subject_feat_x_sadp.mean(3).squeeze(), action_feat_x_sadp.mean(2).squeeze()), dim=1)
            # transform from [N*M, C] -> [N, M, C] -> [N, C]
            x_sadp_origin_feat = x_sadp_origin_feat.view(-1, self.n_person, self.n_channel).mean(1)
            
            # output
            x_sadp_origin_out = self.gcn_model.forward_hidden_feat(x_sadp_origin_feat)
            x_cross_out_a = self.gcn_model.forward_hidden_feat(cross_a)
            x_cross_out_b = self.gcn_model.forward_hidden_feat(cross_b)
            output['x_sadp'] = x_sadp_origin_out
            output['x_sadp_cross_a'] = x_cross_out_a
            output['x_sadp_cross_b'] = x_cross_out_b
            if self.w_SA:
                if self.CA_mode == 'cosine':
                    # calculate cosine similarity, input shape [N, C], output is a scalar
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    action_cos_sim = cos(action_feat_x, action_feat_x_sadp)
                    losses['loss_SA'] = (self.w_SA * (1 - action_cos_sim)).mean()
                elif self.CA_mode == 'l2':
                    action_l2_dist = torch.norm(action_feat_x - action_feat_x_sadp, dim=1)
                    losses['loss_SA'] = (self.w_SA * action_l2_dist).mean()
                else:
                    raise ValueError("Unknown CA_mode: ", self.CA_mode)
        if eval or self.eval_mode:
            return output['x']
        else:
            return self.loss(output, label, losses, label_dasp=label_dasp, label_sadp=label_sadp, label_dasp_mix=label_dasp_mix,
                             label_sadp_mix=label_sadp_mix, tb_writer=self.tb_writer, weight_dict=self.weight_dict, global_step=global_step)
    

    def loss(self, output, label, losses, label_dasp=None, label_sadp=None, label_dasp_mix=None, label_sadp_mix=None, tb_writer=None, weight_dict=None, global_step=None):
        def calc_acc(output, label):
            _, pred = output.max(dim=1)
            correct = pred.eq(label).sum()
            return correct.float() / label.size(0)

        acc = {}
        # calculate loss for training samples
        losses['loss_origin'] = self.loss_crit(output['x'], label).mean()
        acc['acc_origin'] = calc_acc(output['x'], label)

        if label_dasp is not None:
            losses['loss_DASP_x'] = self.loss_crit(output['x_dasp'], label_dasp).mean()
            losses['loss_DASP_xa'] = self.loss_crit(output['x_dasp_cross_a'], label).mean()
            losses['loss_DASP_xb'] = self.loss_crit(output['x_dasp_cross_b'], label_dasp).mean()

            # calculate accuracy
            acc['acc_DASP_x'] = calc_acc(output['x_dasp'], label_dasp)
            acc['acc_DASP_xa'] = calc_acc(output['x_dasp_cross_a'], label)
            acc['acc_DASP_xb'] = calc_acc(output['x_dasp_cross_b'], label_dasp)
        
        if label_sadp is not None:
            # print("[Debug] shape of output['x_sadp']:", output['x_sadp'].shape)
            # print("[Debug] shape of label_sadp:", label_sadp.shape)
            losses['loss_SADP_x'] = self.loss_crit(output['x_sadp'], label_sadp).mean()
            losses['loss_SADP_xa'] = self.loss_crit(output['x_sadp_cross_a'], label).mean()
            losses['loss_SADP_xb'] = self.loss_crit(output['x_sadp_cross_b'], label_sadp).mean()
            # assert label_sadp and label are the same
            assert (label_sadp == label).all(), "label_sadp and label are not the same"

            # calculate accuracy
            acc['acc_SADP_x'] = calc_acc(output['x_sadp'], label_sadp)
            acc['acc_SADP_xa'] = calc_acc(output['x_sadp_cross_a'], label)
            acc['acc_SADP_xb'] = calc_acc(output['x_sadp_cross_b'], label_sadp)
        
        if label_dasp_mix is not None:
            losses['loss_DASP_mix'] = self.loss_crit(output['x_dasp_cross_mix'], label_dasp_mix).mean()
            acc['acc_DASP_mix'] = calc_acc(output['x_dasp_cross_mix'], label_dasp_mix)
        
        if label_sadp_mix is not None:
            losses['loss_SADP_mix'] = self.loss_crit(output['x_sadp_cross_mix'], label_sadp_mix).mean()
            acc['acc_SADP_mix'] = calc_acc(output['x_sadp_cross_mix'], label_sadp_mix)

        if weight_dict is not None:
            for k, v in self.weight_dict.items():
                if k in losses:
                    losses[k] *= weight_dict[k]
        
        if tb_writer is not None:
            for k, v in losses.items():
                tb_writer.add_scalar(k, v, global_step)
            for k, v in acc.items():
                tb_writer.add_scalar(k, v, global_step)
        
        # add all losses
        # print("[Debug] losses:", losses)
        # print("[Debug] devices of all losses:", [v.device for v in losses.values()])

        total_loss = sum(losses.values())
        total_acc = sum(acc.values()) / len(acc)

        # print("[Debug] total_loss:", total_loss)

        # print("[Debug] Shape of total_loss:", total_loss.shape)
        # print("[Debug] Shape of total_acc:", total_acc.shape)

        # print("[Debug] type of total_loss:", type(total_loss))
        # print("[Debug] type of total_acc:", type(total_acc))

        # print("[Debug] Shape of total_loss:", total_loss.shape)
        # print("[Debug] Shape of total_acc:", total_acc.shape)

        # add one dimension and return
        return total_loss.unsqueeze(0), total_acc.float().unsqueeze(0)
    
    def train(self, mode=True):
        self.gcn_model.train(mode)
        self.decouple_net.train(mode)
        self.feat_aggr_net.train(mode)

    def eval(self):
        self.gcn_model.eval()
        self.decouple_net.eval()
        self.feat_aggr_net.eval()