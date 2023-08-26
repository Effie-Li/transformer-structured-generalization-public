import os
os.environ["WANDB_SILENT"] = 'true'
import yaml
import argparse
import dill as pkl

from data.dataset import MultiTaskDataset, MultiTaskDataModule
from model.transformer import MultitaskModelModule
import misc.utils as utils

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm

api = wandb.Api()

class LitProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = tqdm(disable=True)
        return bar
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar
    def init_test_tqdm(self):
        bar = tqdm(disable=True)
        return bar

def setup_dataset(config):
    dataset = MultiTaskDataset(config)
    datamodule = MultiTaskDataModule(dataset=dataset, 
                                     dataset_frac=config['dataset']['dataset_frac'],
                                     batch_size=config['dataset']['batch_size'], 
                                     split_params=config['dataset']['split_params'])
    return datamodule

def setup_model(config):
    config['model']['n_task'] = len(config['dataset']['use_tasks'])
    config['model']['n_feature'] = len(config['data']['item'].keys())
    config['model']['item_dim'] = sum(config['data']['item'].values())+config['model']['n_task']+1
    config['model']['label_dim'] = config['dataset']['max_label']+config['model']['n_task']+1
    if config['model']['n_task'] > 1:
        assert config['model']['use_task_token'] or config['model']['add_task_embedding'], 'must use the task info when multiple tasks are being learned'
    model = MultitaskModelModule(**config['model'])
    return model

def setup_trainer(config, gpu):
    # wandb & trainer setup
    wandb_logger = WandbLogger(project=config['wandb']['project'],
                               save_dir=config['wandb']['save_dir'],
                               config=config)
    checkpoint_callback = ModelCheckpoint(**config['checkpoint'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    bar = LitProgressBar() # disable val progress bar
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, bar],
        gpus=[gpu],
        **config['trainer']
    )
    return trainer

def train(configf, gpu, mode, runid):

    assert configf != ''
    resume = True if runid != '' else False

    if not resume:
        config = yaml.safe_load(open(configf))

        datamodule = setup_dataset(config)
        # okay this should have really been config['dataset] but...
        config['data']['train_idx'] = datamodule.train_idx
        config['data']['val_idx'] = datamodule.val_idx
        model = setup_model(config)

        wandb.login()
        wandb.init(project=config['wandb']['project'],
                   dir=config['wandb']['save_dir'],
                   config=config,
                   mode=mode,
                   resume='allow',
                   settings=wandb.Settings(start_method='thread'))
    
    else: # resume a run
        config = yaml.safe_load(open(configf))
        wandb.login()
        run = wandb.init(project=config['wandb']['project'],
                         dir=config['wandb']['save_dir'],
                         mode=mode,
                         id=runid, 
                         resume='must',
                         settings=wandb.Settings(start_method='thread'))
        if run.resumed:
            config = run.config
            ckpt_root = config['wandb']['save_dir'] + '%s/'% config['wandb']['project']
            # TODO: should be able to cut this and let trainer.fit load model ckpt params
            model, datamodule = utils.reconstruct_model_and_dm(local_run_root=ckpt_root, 
                                                               runid=runid, 
                                                               config=config,
                                                               train_idx=config['data']['train_idx'],
                                                               val_idx=config['data']['val_idx'],
                                                               batch_size=config['dataset']['batch_size'])

    trainer = setup_trainer(config, gpu)
    # let it train, let it train, let it train
    if resume:
        ckpt_path = config['wandb']['save_dir']+'%s/'%config['wandb']['project']+'%s/checkpoints/last.ckpt'%runid
    else:
        ckpt_path = None
    trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    trainer.test(model, datamodule)
    wandb.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-config', default='')
    parser.add_argument('-gpu', type=int, default=4)
    parser.add_argument('-mode', type=str, default='online')
    parser.add_argument('-runid', default='')
    cl_args = parser.parse_args()

    train(cl_args.config, cl_args.gpu, cl_args.mode, cl_args.runid)