import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        print("[Debug] self.classes: ", self.classes)
        print("[Debug] self.counts: ", self.counts)
        print("[Debug] classes_per_it: ", classes_per_it)
        print("[Debug] num_samples: ", num_samples)
        

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            if len(ind) > 4:
                self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            #for i in range(1000):
            yield batch


def init_sampler(opt, labels, mode, sampler_type='PrototypicalBatchSampler'):
    if 'train' in mode:
        classes_per_it = opt['classes_per_it_tr']
        num_samples = opt['num_support_tr'] + opt['num_query_tr']
    else:
        classes_per_it = opt['classes_per_it_val']
        num_samples = opt['num_support_val'] + opt['num_query_val']
    if sampler_type == 'PrototypicalBatchSampler':
        return PrototypicalBatchSampler(labels=labels,
                                        classes_per_it=classes_per_it,
                                        num_samples=num_samples,
                                        iterations=opt['iterations'])
    elif sampler_type == 'CategoriesSampler':
        return CategoriesSampler(label=labels, n_batch=opt['iterations'], n_cls=classes_per_it, n_per=num_samples)

def init_dataloader(opt, dataset, mode, sampler_type='PrototypicalBatchSampler'):
    '''
    Keys of opt:
    - classes_per_it_tr: classes per iteration in training
    - num_support_tr: number of support sample in training for each class
    - num_query_tr: number of query sample in training for each class
    - classes_per_it_val
    - num_support_val
    - num_query_val
    - iterations: iterations per epoch
    '''
    sampler = init_sampler(opt, dataset.label, mode, sampler_type=sampler_type)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader