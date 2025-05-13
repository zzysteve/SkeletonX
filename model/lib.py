import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_DecoupleNet(nn.Module):
    def __init__(self, n_channel, n_frame, n_joint, n_person, n_branch_channel=None, **kwargs):
        '''
        Accept input feature: [N*M, C, T, V]
        Output feature: [N, M, C]
        n_channel: input channel
        n_frame: input frame
        n_joint: input joint
        n_person: input person
        n_branch_channel: output channel for each branch, if set to None, `n_branch_channel` = `n_channel // 2`, and final output channel = `n_channel`
        '''
        super().__init__()
        self.n_channel = n_channel
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.n_person = n_person
        if n_branch_channel is None:
            self.n_branch_channel = n_channel // 2
        else:
            self.n_branch_channel = n_branch_channel
        print("[Debug] building ST_DecoupleNet, n_channel: {}, n_frame: {}, n_joint: {}, n_person: {}".format(
            n_channel, n_frame, n_joint, n_person))
        self.spatio_squeeze = nn.Sequential(nn.Conv2d(n_channel, self.n_branch_channel, kernel_size=1),
                                            nn.BatchNorm2d(self.n_branch_channel), nn.ReLU(True))
        self.tempor_squeeze = nn.Sequential(nn.Conv2d(n_channel, self.n_branch_channel, kernel_size=1),
                                            nn.BatchNorm2d(self.n_branch_channel), nn.ReLU(True))
    

    def forward(self, feat, **kwargs):
        # input feature: [N, M, C, T*V]
        # output feature: spatial: [N*M, C, 1, V], temporal: [N*M, C, T, 1]
        
        # [N, M, C, T*V] -> [N*M, C, T, V]
        feat = feat.view(-1, self.n_channel, self.n_frame, self.n_joint)

        spatial_feat = feat.mean(-2, keepdim=True)
        spatial_feat = self.spatio_squeeze(spatial_feat)

        temporal_feat = feat.mean(-1, keepdim=True)
        temporal_feat = self.tempor_squeeze(temporal_feat)

        # print("[Debug] spatial_feat shape: {}, temporal_feat shape: {}".format(spatial_feat.shape, temporal_feat.shape))
        
        return spatial_feat, temporal_feat


class ST_FeatureAggrNet(nn.Module):
    '''
    Accept two input features: [N, M, C, T*V]
    Output feature: [N, M, C]
    '''

    def __init__(self, n_channel, n_frame, n_joint, n_person, aggr_mode, n_branch_channel=None, **kwargs):
        super().__init__()
        self.n_channel = n_channel
        self.n_frame = n_frame
        self.n_joint = n_joint
        self.n_person = n_person
        self.aggr_mode = aggr_mode
        if aggr_mode == 'element_wise':
            self.conv1_1 = nn.Conv2d(n_channel // 2, n_channel, kernel_size=1)
        elif aggr_mode == 'cross_attn':
            self.w_q = nn.Linear(n_channel // 2, n_channel // 2)
            self.w_k = nn.Linear(n_channel // 2, n_channel // 2)
            self.w_v = nn.Linear(n_channel // 2, n_channel)
    
    def forward(self, feat_a, feat_b, **kwargs):
        '''
        feat_a: a tuple of (spatial_feat_a, temporal_feat_a), with shape [N, M, C, T*V]
        feat_b: a tuple of (spatial_feat_b, temporal_feat_b)

        output: cross_a, cross_b, with shape [N, C]
        '''
        # print("[Debug] Input feature shape: {}, {}".format(feat_a.shape, feat_b.shape))
        
        splitted_hidden_feat_a_subject, splitted_hidden_feat_a_action = feat_a
        splitted_hidden_feat_b_subject, splitted_hidden_feat_b_action = feat_b

        if self.aggr_mode == 'concat':
            output_channel = self.n_channel // 2
            # average pooling before cross up features, [N*M, C, 1, V] -> [N, M, C]
            splitted_hidden_feat_a_subject = splitted_hidden_feat_a_subject.mean(-1).squeeze(-1).view(-1, self.n_person, output_channel)
            splitted_hidden_feat_b_subject = splitted_hidden_feat_b_subject.mean(-1).squeeze(-1).view(-1, self.n_person, output_channel)

            # average pooling before cross up features, [N*M, C, T, 1] -> [N, M, C]
            splitted_hidden_feat_a_action = splitted_hidden_feat_a_action.squeeze(-1).mean(-1).view(-1, self.n_person, output_channel)
            splitted_hidden_feat_b_action = splitted_hidden_feat_b_action.squeeze(-1).mean(-1).view(-1, self.n_person, output_channel)

            # print("[Debug] splitted_hidden_feat_a_subject shape: {}, splitted_hidden_feat_a_action shape: {}"\
            #     .format(splitted_hidden_feat_a_subject.shape, splitted_hidden_feat_a_action.shape))

            # cross up features, input shape [N, M, C]
            cross_a = torch.cat((splitted_hidden_feat_b_subject, splitted_hidden_feat_a_action), dim=2)
            cross_b = torch.cat((splitted_hidden_feat_a_subject, splitted_hidden_feat_b_action), dim=2)
            # average pooling on M, [N, M, C] -> [N, C]
            cross_a = cross_a.mean(1)
            cross_b = cross_b.mean(1)
            return cross_a, cross_b
        elif self.aggr_mode == 'element_wise':
            # Aggregate using element-wise multiplication along channel dimension.

            # output spatial: [N*M, C, 1, V], temporal: [N*M, C, T, 1]
            # spatial_feat: [N*M, C, 1, V] -> [N*M, 1, V, C]
            splitted_hidden_feat_a_subject = splitted_hidden_feat_a_subject.permute(0, 2, 3, 1)
            splitted_hidden_feat_b_subject = splitted_hidden_feat_b_subject.permute(0, 2, 3, 1)
            # temporal_feat: [N*M, C, T, 1] -> [N*M, T, 1, C]
            splitted_hidden_feat_a_action = splitted_hidden_feat_a_action.permute(0, 2, 3, 1)
            splitted_hidden_feat_b_action = splitted_hidden_feat_b_action.permute(0, 2, 3, 1)
            
            # element-wise multiplication, output [N*M, T, V, C]
            cross_a = splitted_hidden_feat_a_action * splitted_hidden_feat_b_subject
            cross_b = splitted_hidden_feat_b_action * splitted_hidden_feat_a_subject

            cross_a = cross_a.permute(0, 3, 1, 2)
            cross_b = cross_b.permute(0, 3, 1, 2)
            
            # perform 1x1 convolution to reduce channel dimension, output [N*M, C, T, V]
            cross_a = self.conv1_1(cross_a)
            cross_b = self.conv1_1(cross_b)
            
            cross_a = cross_a.view(-1, self.n_person, self.n_channel, self.n_frame * self.n_joint).mean(-1).mean(1)
            cross_b = cross_b.view(-1, self.n_person, self.n_channel, self.n_frame * self.n_joint).mean(-1).mean(1)

            # print("[DEBUG] cross_a shape: {}, cross_b shape: {}".format(cross_a.shape, cross_b.shape))
            return cross_a, cross_b
        elif self.aggr_mode == 'original_s2a':
            # transpose channel to the last dimension
            # [N,M,C,T*V]->[N*M,C,T,V]->[N*M,T,V,C]
            feat_a = feat_a.view(-1, self.n_channel, self.n_frame, self.n_joint).permute(0, 2, 3, 1)
            feat_b = feat_b.view(-1, self.n_channel, self.n_frame, self.n_joint).permute(0, 2, 3, 1)

            # apply softmax to channel dimension of subject feature, with shape [N*M, 1, V, C]
            splitted_hidden_feat_a_subject = splitted_hidden_feat_a_subject.softmax(dim=-1).unsqueeze(-1)
            splitted_hidden_feat_b_subject = splitted_hidden_feat_b_subject.softmax(dim=-1).unsqueeze(-1)
            # multiply subject feature with feat1
            cross_a = feat_a * splitted_hidden_feat_a_subject
            cross_b = feat_b * splitted_hidden_feat_b_subject
            
            cross_a = cross_a.permute(0, 3, 1, 2).contiguous().view(-1, self.n_channel, self.n_frame * self.n_joint)
            cross_b = cross_b.permute(0, 3, 1, 2).contiguous().view(-1, self.n_channel, self.n_frame * self.n_joint)

            cross_a = cross_a.mean(-1)
            cross_b = cross_b.mean(-1)
            return cross_a, cross_b
        
        elif self.aggr_mode == 'cross_attn':
            qk_output_channel = self.n_channel // 2
            output_channel = self.n_channel
            # average pooling before cross up features, [N*M, C, 1, V] -> [N*M, C, V] -> [N*M, V, C]
            splitted_hidden_feat_a_subject = splitted_hidden_feat_a_subject.squeeze().permute(0, 2, 1)
            splitted_hidden_feat_b_subject = splitted_hidden_feat_b_subject.squeeze().permute(0, 2, 1)

            # average pooling before cross up features, [N*M, C, T, 1] -> [N*M, C, T] -> [N*M, T, C]
            splitted_hidden_feat_a_action = splitted_hidden_feat_a_action.squeeze().permute(0, 2, 1).contiguous()
            splitted_hidden_feat_b_action = splitted_hidden_feat_b_action.squeeze().permute(0, 2, 1).contiguous()

            # apply linear transformation to subject feature, with shape q:[N*M, T, C], k,v:[N*M, V, C]
            xa_q = self.w_q(splitted_hidden_feat_a_action)
            xa_k = self.w_k(splitted_hidden_feat_b_subject)
            xa_v = self.w_v(splitted_hidden_feat_b_subject)

            xb_q = self.w_q(splitted_hidden_feat_b_action)
            xb_k = self.w_k(splitted_hidden_feat_a_subject)
            xb_v = self.w_v(splitted_hidden_feat_a_subject)

            # calculate attention score, shape [N*M, T, V]
            attn_a = torch.matmul(xa_q, xa_k.transpose(-2, -1)) / math.sqrt(qk_output_channel)
            attn_b = torch.matmul(xb_q, xb_k.transpose(-2, -1)) / math.sqrt(qk_output_channel)

            # apply softmax to attention score
            attn_a = F.softmax(attn_a, dim=-1)
            attn_b = F.softmax(attn_b, dim=-1)

            # apply attention score to value, shape [N*M, T, C]
            cross_a = torch.matmul(attn_a, xa_v)
            cross_b = torch.matmul(attn_b, xb_v)
            
            # average pooling on M and T, [N*M, T, C] -> [N, C]
            cross_a = cross_a.mean(-2).view(-1, self.n_person, output_channel).mean(-2)
            cross_b = cross_b.mean(-2).view(-1, self.n_person, output_channel).mean(-2)
            return cross_a, cross_b
            
        else:
            raise NotImplementedError("aggr_mode {} not implemented".format(self.aggr_mode))