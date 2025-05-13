#!/usr/bin/env python
from __future__ import print_function

import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
import itertools

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

import resource
from utils.config import get_parser
from utils.data_utils import *
from utils.visualize import wrong_analyze
from utils.cls_loss import build_loss

from model.protonet import prototypical_loss
from feeders.sampler_episode_train import init_dataloader as meta_init_dataloader

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

from utils.collect_env import get_env_str


def init_seed(seed):
    # fix random seeds
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    # Dynamic import
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Processor:
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':  # train = train + val
            if not arg.aux_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        else:
            self.test_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_data()
            self.load_optimizer()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        self.data_loader = dict()
        Feeder = import_class(self.arg.feeder)
        if self.arg.phase == 'train':
            aux_dataset = Feeder(**self.arg.aux_feeder_args, class_group=self.arg.one_shot_class_group)
            opt = dict(
                classes_per_it_tr=self.arg.num_way,
                num_support_tr=self.arg.num_shot,
                num_query_tr=self.arg.num_query,
                iterations=self.arg.it_per_ep,
            )
            self.data_loader['aux'] = meta_init_dataloader(opt, aux_dataset, mode='train')
            self.train_modality = dict(
                bone=self.arg.bone,
                vel=self.arg.vel
            )
        opt = dict(
            classes_per_it_val=self.arg.num_way,
            num_support_val=self.arg.num_shot,
            num_query_val=self.arg.num_query,
            iterations=self.arg.num_episode,
            )
        if self.arg.eval_mode == 'meta':
            eval_dataset = Feeder(**self.arg.eval_feeder_args)
            self.data_loader['eval'] = meta_init_dataloader(opt, eval_dataset, mode='val')
        elif self.arg.eval_mode == 'ntu120':
            self.data_loader['anchor'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.anchor_feeder_args),
                batch_size=20, shuffle=False)
            self.data_loader['eval'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.eval_feeder_args),
                batch_size=64,
                shuffle=False,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)

        self.test_modality = dict(
            bone=self.arg.bone,
            vel=self.arg.vel
        )
        if self.arg.bone:
            self.print_log('[Info] Using bone modality.')
        if self.arg.vel:
            self.print_log('[Info] Using velocity modality.')
        if self.arg.bone and self.arg.vel:
            self.print_log('[Info] Using bone velocity.')
        
    def load_model(self):
        self.print_log("System Info:\n{}\n".format(get_env_str()))
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # copy current file to work_dir
        shutil.copy2(sys.argv[0], self.arg.work_dir)
        print(Model)
        gcn_model = Model(**self.arg.model_args, metric_func=self.arg.metric_func)
        print(gcn_model)
        self.loss = build_loss(self.arg).cuda(output_device)

        self.rank_criterion = nn.MarginRankingLoss(margin=0.05).cuda(output_device)
        self.softmax_layer = nn.Softmax(dim=1).cuda(output_device)
        # construct weighted loss dict
        self.loss_w_dict = dict()
        self.loss_w_dict['loss_DASP_xa'] = self.loss_w_dict['loss_DASP_xb'] = \
            self.loss_w_dict['loss_SADP_xa'] = self.loss_w_dict['loss_SADP_xb'] = self.arg.w_xsample
        self.loss_w_dict['loss_DASP_mix'] = self.loss_w_dict['loss_SADP_mix'] = self.arg.w_mixup
        self.loss_w_dict['loss_SP'] = self.arg.w_SP
        self.loss_w_dict['loss_SA'] = self.arg.w_SA
        
        n_channel, n_frame, n_joint, n_person = gcn_model.base_channel * 4, gcn_model.num_frame // 4, gcn_model.num_point, gcn_model.num_person
        ST_Decouple = import_class("model.stdecouple.Model")
        if self.arg.phase == 'train':
            self.model = ST_Decouple(gcn_model, self.loss, self.arg.feat_aggr_mode, n_channel, n_frame, n_joint, n_person,
                                    w_SA=self.arg.w_SA, w_SP=self.arg.w_SP, CA_mode=self.arg.CA_mode, DASP_mixup_ep=self.arg.DASP_mixup_ep, 
                                    SADP_mixup_ep=self.arg.SADP_mixup_ep, tb_writer=self.train_writer, weight_dict=self.loss_w_dict)
        else:
            self.model = ST_Decouple(gcn_model, self.loss, self.arg.feat_aggr_mode, n_channel, n_frame, n_joint, n_person,
                                    w_SA=self.arg.w_SA, w_SP=self.arg.w_SP, CA_mode=self.arg.CA_mode, DASP_mixup_ep=self.arg.DASP_mixup_ep, 
                                    SADP_mixup_ep=self.arg.SADP_mixup_ep, tb_writer=self.test_writer, weight_dict=self.loss_w_dict)
        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_weight(self, weights, output_device):
        self.print_log('Load weights from {}.'.format(weights))
        if '.pkl' in weights:
            with open(weights, 'r') as f:
                weights = pickle.load(f)
        else:
            weights = torch.load(weights)

        weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

        keys = list(weights.keys())
        for w in self.arg.ignore_weights:
            for key in keys:
                if w in key:
                    if weights.pop(key, None) is not None:
                        self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                    else:
                        self.print_log('Can Not Remove Weights: {}.'.format(key))

        try:
            self.model.load_state_dict(weights)
        except:
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            print('Can not find these weights:')
            for d in diff:
                print('  ' + d)
            state.update(weights)
            self.model.load_state_dict(state)


    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['aux']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001, pairloader=0.001)
        process = tqdm(loader, ncols=40)

        # train model with real data
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            timer['dataloader'] += self.split_time()
            # setup optimizer
            self.optimizer.zero_grad()
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)

            embedding = self.model(calc_diff_modality(data, **self.test_modality), get_hidden_feat=True)
            tot_loss, tot_acc = prototypical_loss(embedding, label, self.arg.num_shot)

            # backward
            if len(self.arg.device) > 1:
                tot_loss = tot_loss.mean()
                tot_acc = tot_acc.mean()
            tot_loss.backward()
            self.optimizer.step()

            loss_value.append(tot_loss.mean().data.item())
            timer['model'] += self.split_time()

            self.train_writer.add_scalar('loss', tot_loss.mean().data.item(), self.global_step)
            self.train_writer.add_scalar('acc', tot_acc.mean().data.item(), self.global_step)
            
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
            acc_value.append(tot_acc.mean().data.item())

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value),
                                                                                np.mean(acc_value) * 100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}, [Pair Data]{pairloader}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

        
    def eval(self, epoch, save_score=False, wrong_file=None, result_file=None):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        acc_list = []
        step = 0
        process = tqdm(self.data_loader['eval'], ncols=40)
        for batch_idx, (data, label, episode_index) in enumerate(process):
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                embedding = self.model(calc_diff_modality(data, **self.test_modality), get_hidden_feat=True)
                _, acc = prototypical_loss(embedding, label, self.arg.num_shot)
                step += 1
                acc_list.append(acc.mean().data.item())

        accuracy = np.mean(acc_list)
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_acc_epoch = epoch + 1

        print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
        self.print_log('\tMean testing acc: {:.2f}%.'.format(np.mean(accuracy) * 100))
        if self.arg.phase == 'train':
            self.val_writer.add_scalar('acc', accuracy, self.global_step)

    def eval_oneshot(self, epoch, save_score=False, wrong_file=None, result_file=None):
        # evaluation following NTU120 protocol
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        for (data, label, index) in self.data_loader['anchor']:
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                anchor_label = label.long().cuda(self.output_device)
                anchor_embedding = self.model(calc_diff_modality(data, **self.test_modality), get_hidden_feat=True)
            break

        label_list = []
        pred_list = []
        step = 0
        process = tqdm(self.data_loader['eval'], ncols=40)
        for batch_idx, (data, label, index) in enumerate(process):
            label_list.append(label.data.cpu().numpy())
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                embedding = self.model(calc_diff_modality(data, **self.test_modality), get_hidden_feat=True)
                knn_indices = get_knn_predict(embedding, anchor_embedding, self.arg.knn_metric)
                most_frequent, _ = torch.mode(knn_indices, dim=1)
                predict_label = anchor_label[most_frequent]

                pred_list.append(predict_label.data.cpu().numpy())
                step += 1

            if wrong_file is not None or result_file is not None:
                predict = list(predict_label.cpu().numpy())
                true = list(label.data.cpu().numpy())
                for i, x in enumerate(predict):
                    if result_file is not None:
                        f_r.write(str(x) + ',' + str(true[i]) + '\n')
                    if x != true[i] and wrong_file is not None:
                        f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')


        label_list = np.concatenate(label_list)
        pred_list = np.concatenate(pred_list)
        accuracy = np.mean(pred_list == label_list)
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_acc_epoch = epoch + 1

        print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
        self.print_log('\tMean testing acc: {:.2f}%.'.format(np.mean(accuracy) * 100))
        if self.arg.phase == 'train':
            self.val_writer.add_scalar('acc', accuracy, self.global_step)

        # acc for each class:
        confusion = confusion_matrix(label_list, pred_list)
        list_diag = np.diag(confusion)
        list_raw_sum = np.sum(confusion, axis=1)
        each_acc = list_diag / list_raw_sum
        with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, 'eval'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(each_acc)
            writer.writerows(confusion)


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['aux']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            self.print_log(f'eval_interval: {self.arg.eval_interval}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch + 1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)

                if (epoch + 1) % self.arg.eval_interval == 0 or (epoch + 1) == self.arg.num_epoch:
                    self.print_log("Eval epoch: {}".format(epoch + 1))
                    if self.arg.eval_mode == 'ntu120':
                        self.eval_oneshot(epoch, save_score=self.arg.save_score)
                    elif self.arg.eval_mode == 'meta':
                        self.eval(epoch, save_score=self.arg.save_score)
                    else:
                        raise ValueError('Only support ntu120 and meta evaluation mode.')

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            if self.arg.eval_mode == 'ntu120':
                self.eval_oneshot(epoch=0, save_score=self.arg.save_score, wrong_file=wf, result_file=rf)
            else:
                self.eval(epoch=0, save_score=True, wrong_file=wf, result_file=rf)
            wrong_analyze(wf, rf)

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.load_weight(self.arg.weights, self.output_device)
            if self.arg.eval_mode == 'ntu120':
                self.print_log('Evaluation mode: ntu120')
                self.eval_oneshot(epoch=0, save_score=self.arg.save_score)
            elif self.arg.eval_mode == 'meta':
                self.print_log('Evaluation mode: meta')
                self.eval(epoch=0, save_score=self.arg.save_score)
            else:
                raise ValueError('Please appoint --eval-mode.')
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg from config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    try:
        processor.start()
    except:
        processor.print_log(str(traceback.format_exc()), False)