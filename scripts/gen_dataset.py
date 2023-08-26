import yaml
import argparse
import numpy as np
import dill as pkl
from data.task import ItemPool, Solver

def generate_dataset(configf):
    config = yaml.safe_load(open(configf))
    
    pool = ItemPool(config['data']['item'])
    s = Solver()
    dataset = {task: {'context':[], 'out_seq':[], 'sort_idx':[], 'sorted_label':[]} 
                      for task in config['data']['task']}
    dataset['seq'] = []
    dataset['random_label'] = []
    dataset['idx_label'] = []

    for _ in range(config['dataset']['n_seq']):

        # sample sequence of itmes
        seq_len = np.random.choice(range(config['dataset']['min_len'], config['dataset']['max_len']+1))
        seq = pool.sample(n=seq_len, replace=False)
        dataset['seq'].append(seq)

        # add index labels and random labels
        dataset['idx_label'].append(list(range(seq_len)))
        labels = sorted(np.random.choice(range(config['dataset']['max_label']), seq_len, replace=False))
        dataset['random_label'].append(labels)

        # go through each task to generate the output seq, sort_idx, and sorted_label
        for task in config['data']['task']:
            # also sample a context indicator
            dataset[task]['context'].append(np.random.choice(range(config['dataset']['max_context'])))
            out_seq = s.solve(seq, task=task)
            sort_idx = [seq.index(x) for x in out_seq]
            dataset[task]['out_seq'].append(out_seq)
            dataset[task]['sort_idx'].append(sort_idx)
            dataset[task]['sorted_label'].append(np.array(labels)[sort_idx].tolist())
    
    pkl.dump(dataset, open(config['dataset']['fname'], 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-configf', default='')
    args = parser.parse_args()

    generate_dataset(configf=args.configf)