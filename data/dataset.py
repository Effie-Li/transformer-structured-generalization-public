import math
import numpy as np
import dill as pkl

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import pytorch_lightning as pl

class MultiTaskDataset(Dataset):

    '''
    load the raw sequence data and turn them into multihot tensors

    TODO: turn context into onehot vec
    '''

    def __init__(self,
                 config):

        self.raw_data = pkl.load(open(config['dataset']['fname'], 'rb'))
        self.config = config
        self.tasks = config['dataset']['use_tasks']
        self.item_meta_info = config['data']['item']
        self.max_seq_len = config['dataset']['max_len']+2 # counting <task> and <eos> tokens
    
    def __len__(self):
        # total length of dataset is n_sequence x n_task
        return len(self.raw_data['seq']) * len(self.tasks)

    def __getitem__(self, idx):
        '''
        returns
        -------
        tensor_values : dict
            containing keys ['context','seq','idx_label','random_label','out_seq','sort_idx','sorted_label']
        '''
        # extract task_idx and seq_idx from idx
        # 0*len(self.tasks), 1*len(self.tasks), 2*len(self.tasks), ..., reference the first task
        # 0*len(self.tasks)+1, 1*len(self.tasks)+1, 2*len(self.tasks)+1, ..., reference the second task
        # ...
        task_idx = idx % len(self.tasks)
        seq_idx = (idx-task_idx) // len(self.tasks)
        raw_values = self.get_raw_item(task=self.tasks[task_idx], seq_idx=seq_idx)
        tensor_values = self.raw_to_tensor(raw_values)
        return tensor_values

    def get_raw_item(self, task, seq_idx):
        return {
            'task': task,
            'context': self.raw_data[task]['context'][seq_idx],
            'seq': self.raw_data['seq'][seq_idx],
            'idx_label': self.raw_data['idx_label'][seq_idx],
            'random_label': self.raw_data['random_label'][seq_idx],
            'out_seq': self.raw_data[task]['out_seq'][seq_idx],
            'sort_idx': self.raw_data[task]['sort_idx'][seq_idx],
            'sorted_label': self.raw_data[task]['sorted_label'][seq_idx]
        }

    def raw_to_tensor(self, raw_values):
        result = {'task': raw_values['task'],
                  'context': torch.tensor(raw_values['context'])}
        for k in ['seq', 'out_seq']:
            result[k] = self.raw_sequence_to_tensor(items=raw_values[k], 
                                                    item_transform_func=self.item_to_multihot,
                                                    task=raw_values['task'])
        for k in ['idx_label', 'random_label', 'sort_idx', 'sorted_label']:
            result[k] = self.raw_sequence_to_tensor(items=raw_values[k], 
                                                    item_transform_func=self.label_to_onehot,
                                                    task=raw_values['task'])
        return result

    def raw_sequence_to_tensor(self, items, item_transform_func, task):
        '''
        transforms each item to a tensor
        then 1) pad 2) stack 3) add task/eos tokens,
        finally pad zeros after the true tokens to make max_seq_len

        args
        ----
        items : list 
            list of Item() or int labels
        item_transform_func : func
            a function to apply to each item in items
        task : str
            which task the current observation belongs to (used to generate the appropriate task token)

        returns
        -------
        torch.tensor
            shape (max_seq_len, dim) where max_seq_len=n_items+2+n_pad, dim=item_dim_after_transformation+n_task+1
        '''

        items = torch.stack([item_transform_func(x) for x in items]) # (n_items, transformed_dim)
        task_token = F.one_hot(torch.tensor(self.tasks.index(task)), num_classes=len(self.tasks)+1).unsqueeze(0) # (1, n_tasks+1)
        eos_token = F.one_hot(torch.tensor(len(self.tasks)), num_classes=len(self.tasks)+1).unsqueeze(0) # the last 'task'

        # pad and stack
        item_dim = items.shape[-1]
        task_dim = task_token.shape[-1]
        task_token = F.pad(task_token, (0, item_dim), value=0.0) # pad right
        eos_token = F.pad(eos_token, (0, item_dim), value=0.0) # pad right
        items = F.pad(items, (task_dim, 0), value=0.0) # pad left
        seq = torch.cat([task_token, items, eos_token], dim=0) # (n_items+2, transformed_dim+n_tasks+1)

        # add zero-pad items to make max_seq_len
        seq = F.pad(seq, (0, 0, 0, self.max_seq_len-seq.shape[0]), value=0.0) # pad down
        return seq
    
    def item_to_multihot(self, item):
        '''
        args
        ----
        item : Item
            an instance of self.Item namedtuple in task.ItemPool()
            e.g., Item(shape=0, color=6)

        returns
        -------
        a multihot tensor
            e.g., [1,0,0,0 | 0,0,0,0,0,0,1] if n_shape=4 and n_color=7
        '''
        # self.item_meta_info : dict of 'feature (str) -> n_classes (int)' pair for each field of Item()
        multihot_item = [F.one_hot(torch.tensor(getattr(item, f)), num_classes=self.item_meta_info[f])
                         for f in self.config['data']['feature_order']]
        multihot_item = torch.cat(multihot_item)
        return multihot_item

    def label_to_onehot(self, label):
        '''
        args
        ----
        label : int

        args
        ----
        a onehot tensor
            length = max label given in config
        '''
        return F.one_hot(torch.tensor(label), num_classes=self.config['dataset']['max_label'])

class MultiTaskDataModule(pl.LightningDataModule):

    '''
    wrapper class for MultiTaskDataset, handles train/validation split 
    and interfaces with trainer
    '''

    def __init__(self, 
                 dataset, 
                 batch_size, 
                 dataset_frac=1.0,
                 split_params={'mode': 'random', 'train_prop':0.75}, 
                 train_idx=None, 
                 val_idx=None):
        '''
        args
        ----
        dataset : MultiTaskDataset
            an instance of MultiTaskDataset
        dataset_frac : float, (0,1]
            fraction of dataset to use (shrinks train/val dataset size)
        split_params : dict
            dict of values used to split the dataset into train/val
        train_idx/val_idx : list of int
            used to subset MultiTaskDataset and recreate a previous dataset
        '''
        super().__init__()

        assert type(dataset) == MultiTaskDataset
        assert batch_size <= len(dataset)

        self.dataset = dataset
        self.dataset_frac = dataset_frac
        self.batch_size = batch_size
        self.split_params = split_params
        self.train_idx = train_idx
        self.val_idx = val_idx

        self.setup()

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:

            if self.train_idx is None and self.val_idx is None:
                train_idx, val_idx = self.split_dataset()
                self.train_idx = train_idx
                self.val_idx = val_idx
            self.data_train = Subset(self.dataset, self.train_idx)
            self.data_val = Subset(self.dataset, self.val_idx)

        if stage == 'test' or stage is None:
            self.data_test = self.dataset

    def split_dataset(self):

        '''
        generates train_idx and val_idx based on self.split_params

        returns
        -------
        train_idx : list of int
            indicates the indices of observations used for training
        val_idx : list of int
            indicates the indices of observations used in validation
        '''

        N_seq = math.ceil(len(self.dataset.raw_data['seq'])*self.dataset_frac)
        all_seq_idx = np.arange(N_seq) # take a subset of the dataset

        if self.split_params['mode'] == 'random':
            # randomly hold out some sequences
            n_train = math.floor(N_seq * self.split_params['train_prop'])
            train_seq_idx = np.random.choice(all_seq_idx, size=n_train, replace=False)
            val_seq_idx = all_seq_idx[~np.isin(all_seq_idx, train_seq_idx)]

        if self.split_params['mode'] == 'len':
            # hold out sequences of specific length
            seq_lens = [len(self.dataset.raw_data['seq'][i]) for i in all_seq_idx]
            train_seq_idx = []
            val_seq_idx = {}
            for i, l in enumerate(seq_lens):
                if l >= self.split_params['train_range'][0] and l <= self.split_params['train_range'][1]:
                    train_seq_idx.append(i)
                else:
                    if l not in val_seq_idx.keys(): 
                        val_seq_idx[l] = []
                    elif len(val_seq_idx[l]) < self.split_params['val_count']: 
                        val_seq_idx[l].append(i)

            train_seq_idx = np.array(train_seq_idx)
            val_seq_idx = np.array(sum(val_seq_idx.values(), [])) # concat seq indicies across all seq lengths

        # broadcast sequence idx to idx in dataset (accounting for multiple tasks)
        n_task = len(self.dataset.tasks)
        train_idx = np.concatenate([train_seq_idx*n_task+i for i in range(n_task)])
        val_idx = np.concatenate([val_seq_idx*n_task+i for i in range(n_task)])
        
        # turn into lists for wandb to record all values
        # the sort is for the human eye...
        return np.sort(train_idx).tolist(), np.sort(val_idx).tolist()
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)