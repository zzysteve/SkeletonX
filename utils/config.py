import argparse
from argparse import ArgumentParser

from torchlight import DictAction


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser: ArgumentParser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')

    # basic configs
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml',
                        help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False,
                        help='if ture, the classification score will be stored')

    # visualize, debug & record
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--eval-feeder', default='feeder.feeder', help='data loader for evaluation')
    parser.add_argument('--num-worker', type=int, default=32, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for test')
    parser.add_argument('--aux-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for auxiliary set')
    parser.add_argument('--anchor-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for anchor set')
    parser.add_argument('--eval-feeder-args', action=DictAction, default=dict(),
                        help='the arguments of data loader for evaluation set')
    parser.add_argument('--one-shot-class-group', default=None, help='special selected class for one shot, starts with os, e.g. os50')

    # classification model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--loss-type', choices=['CE', 'Focal'], default='CE', help='type of loss')
    parser.add_argument('--metric-func', choices=['ArcFace', 'CosFace', 'SphereFace'], default=None,
                        help='type of metric function')
    parser.add_argument('--pred_threshold', type=float, default=0.0, help='threshold to define the confident sample')
    parser.add_argument('--use_p_map', type=str2bool, default=True,
                        help='whether to add (1 - p_{ik}) to constrain the auxiliary item')
    # optim
    parser.add_argument('--base-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--aug-base-lr', type=float, default=2e-3, help='initial learning rate for augment')
    parser.add_argument('--step', type=list, default=[20, 40, 60], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    # One-shot KNN setting
    parser.add_argument('--knn-metric', type=str, default='cosine', help='knn metric, can be [cosine, euclidean, EMD]')

    # Similarity matrix analysis
    parser.add_argument('--save-sim-mat', type=str2bool, default=False, help='save similarity matrix')

    # Feature disentanglement
    parser.add_argument('--w-xsample', type=float, default=0.1, help='weight of cross sample loss')
    parser.add_argument('--x-SADP', type=str2bool, default=False, help='use SADP(Same Action Different Person) or not')
    parser.add_argument('--x-DASP', type=str2bool, default=False, help='use DASP(Different Action Same Person) or not')
    parser.add_argument('--w-SA', type=float, default=None, help='weight of same action loss')
    parser.add_argument('--w-SP', type=float, default=None, help='weight of same person loss')
    parser.add_argument('--CA-mode', type=str, default=None, help='mode of cross action loss', choices=['l2', 'l1', 'cosine'])
    parser.add_argument('--feat-aggr-mode', type=str, default=None, help='mode of feature aggregation', choices=['concat', 'element_wise', 'original_s2a', 'cross_attn'])

    # Feature mixup
    parser.add_argument('--SADP-mixup-ep', type=int, default=-1, help='start epoch of SADP mixup')
    parser.add_argument('--DASP-mixup-ep', type=int, default=-1, help='start epoch of DASP mixup')
    parser.add_argument('--w-mixup', type=float, default=0.1, help='weight of mixup loss')

    # Ablations on evaluation
    parser.add_argument('--eval-mask-subject', type=str2bool, default=False, help='mask subject in evaluation')
    
    # Visualization
    parser.add_argument('--test-output-subject', type=str2bool, default=False, help='output subject in test')
    parser.add_argument('--abl-zeroout', type=str, default=None, help='abl zero out', choices=['action', 'subject'])

    # Other modalities
    parser.add_argument('--bone', action='store_true', help='use bone data')
    parser.add_argument('--vel', action='store_true', help='use velocity data')

    # Add meta-learning parameters.
    parser.add_argument('--num-way', type=int, default=5)
    parser.add_argument('--num-shot', type=int, default=1)
    parser.add_argument('--num-query', type=int, default=3)
    parser.add_argument('--num-episode', type=int, default=1000)
    parser.add_argument('--it-per-ep', type=int, default=100)
    parser.add_argument('--eval-mode', type=str, default='ntu120', choices=['ntu120', 'meta'])
    return parser
