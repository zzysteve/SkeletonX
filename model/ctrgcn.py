import math
from model.modules import *
from utils.cls_loss import *


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Model(nn.Module):
    def build_basic_blocks(self):
        A = self.graph.A  # 3,25,25
        self.l1 = TCN_GCN_unit(self.in_channels, self.base_channel, A, residual=False, adaptive=self.adaptive)
        self.l2 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l3 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l4 = TCN_GCN_unit(self.base_channel, self.base_channel, A, adaptive=self.adaptive)
        self.l5 = TCN_GCN_unit(self.base_channel, self.base_channel * 2, A, stride=2, adaptive=self.adaptive)
        self.l6 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l7 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 2, A, adaptive=self.adaptive)
        self.l8 = TCN_GCN_unit(self.base_channel * 2, self.base_channel * 4, A, stride=2, adaptive=self.adaptive)
        self.l9 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)
        self.l10 = TCN_GCN_unit(self.base_channel * 4, self.base_channel * 4, A, adaptive=self.adaptive)


    def build_classifier(self):
        if self.metric_func is None:
            self.fc = nn.Linear(self.base_channel * 4, self.num_class)
        elif self.metric_func == 'ArcFace':
            self.fc = ArcMarginProduct(self.base_channel * 4, self.num_class)
        elif self.metric_func == 'CosFace':
            self.fc = AddMarginProduct(self.base_channel * 4, self.num_class)
        elif self.metric_func == 'SphereFace':
            self.fc = SphereProduct(self.base_channel * 4, self.num_class)
        else:
            raise KeyError(f"no such Metric Function {self.metric_func}")

    def __init__(self,
                 # Base Params
                 num_class=60, num_point=25, num_frame=64, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 base_channel=64, drop_out=0, adaptive=True,
                 # Module Params
                 metric_func=None, pred_threshold=0, use_p_map=True):
        super(Model, self).__init__()

        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = num_person
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        self.in_channels = in_channels
        self.base_channel = base_channel
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x
        self.adaptive = adaptive
        self.metric_func = metric_func
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map


        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.build_basic_blocks()
        self.build_classifier()

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def get_hidden_feat(self, x, pooling=True, raw=False):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # First stage
        x = self.l1(x)
        if self.attn_mode is not None:
            x = self.attn_low(x)

        # Second stage
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        if self.attn_mode is not None:
            x = self.attn_mid(x)

        # Third stage
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        if self.attn_mode is not None:
            x = self.attn_high(x)

        # Forth stage
        x = self.l9(x)
        x = self.l10(x)
        if self.attn_mode is not None:
            x = self.attn_fin(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)

        if raw:
            return x
        if pooling:
            return x.mean(3).mean(1)
        else:
            return x.mean(1)

    def forward(self, x, get_hidden_feat=False):
        if get_hidden_feat:
            return self.get_hidden_feat(x)

        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T*V
        c_new = x.size(1)
        # N, M, C, T*V
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x)



    def forward_hidden_feat(self, x, mean=False):
        if mean:
            x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)

    def forward_to_hidden_feat(self, x):
        # return the hidden feature of the last layer
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # First stage
        x = self.l1(x)

        # Second stage
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        # Third stage
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)

        # Forth stage
        x = self.l9(x)
        x = self.l10(x)
        return x