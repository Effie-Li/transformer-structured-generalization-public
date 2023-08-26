import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from .layers import *

class MultitaskTransformer(nn.Module):

    '''
    causal transformer
    '''

    def __init__(self,
                 item_dim,
                 label_dim,
                 embed_dim,
                 n_heads, 
                 depth,
                 mlp_dim,
                 pos_encoding='label',
                 use_task_token=False,
                 add_task_embedding=False,
                 use_task_context=False, # kept for now for backward compatibility
                 dropout=0.):

        '''
        args
        ----
        item_dim : int
            length of item vectors
        label_dim : int
            length of label vectors
        embed_dim : int
            model latent embedding length
        n_heads : int or list of int
            # of attention heads
        depth : int
            # of attention layers
        mlp_dim : int
            length of the hidden layer in mlp FF sublayer
        pos_encoding : str
            type of positional encoding, 'label' (to use random labels) or 'sinusoidal'/'learnable' (based on indices in seq)
        use_task_token : bool
            whether to use the task token included in item sequences in training
        add_task_embedding : bool
            whether to add the task embedding directly to item embeddings or leave it as a separate token
        dropout : float,
            proportion of weights to drop out during training
            (shared between input embedding, attention, and mlp)
        '''

        if type(n_heads)==list:
            assert len(n_heads)==depth

        super().__init__()

        self.item_embed = nn.Linear(item_dim, embed_dim)
        self.pos_encoding = pos_encoding
        if (self.pos_encoding == 'label') or (self.pos_encoding == 'learnable'):
            self.label_embed = nn.Linear(label_dim, embed_dim)
        elif self.pos_encoding == 'sinusoidal':
            self.label_embed = SinusoidPositionEncoding(50, embed_dim) # TODO: remove hardcoding
        else:
            raise Exception('unknown position embedding method')
        
        self.use_task_token = use_task_token
        self.add_task_embedding = add_task_embedding
        assert not (use_task_token and add_task_embedding) # distinct two methods, can't be both True (can be both False for single-task learning)
        if use_task_token or add_task_embedding:
            self.task_embed = nn.Linear(item_dim, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)

        if type(n_heads)==int:
            n_heads = [n_heads]*depth
        self.attn_layers = nn.ModuleList([
            AttentionLayer(embed_dim, n_heads[i], mlp_dim, dropout)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.item_head = nn.Linear(embed_dim, item_dim)
        self.label_head = nn.Linear(embed_dim, label_dim)

    def input_embed(self, input_dict):
        '''
        args
        ----
        input : dict,
            dict of input tensors, must contain keys ['item', 'label']
            - input['item'], shape (batch_size, max_seq_len, item_dim)
            - input['label'], shape (batch_size, max_seq_len, label_dim)
            - input['task'], shape (batch_size, 1, item_dim)
        '''
        if self.use_task_token:
            task_embed = self.task_embed(input_dict['item'][:,:1,:]) # embed task token (1st token) separately
            item_embed = self.item_embed(input_dict['item'][:,1:,:])
            item_embed = torch.cat([task_embed, item_embed], dim=1)
            if (self.pos_encoding == 'label') or (self.pos_encoding == 'learnable'):
                label_embed = self.label_embed(input_dict['label'][:,1:,:])
            else: # sinusoidal
                label_embed = self.label_embed(input_dict['label'][:,1:,-50:]) # TODO: remove assumption for max label
            label_embed = torch.cat([task_embed, label_embed], dim=1)
        else:
            item_embed = self.item_embed(input_dict['item'])
            if (self.pos_encoding == 'label') or (self.pos_encoding == 'learnable'):
                label_embed = self.label_embed(input_dict['label'])
            else: # sinusoidal
                label_embed = self.label_embed(input_dict['label'][:,:,-50:]) # TODO: remove assumption for max label
            if self.add_task_embedding:
                # add task embedding to items
                task_embed = self.task_embed(input_dict['task']) # embed task tokens separately, (batch, 1, embed_dim)
                item_embed = item_embed + repeat(task_embed, 'b n d -> b (repeat n) d', repeat=item_embed.shape[1])
        
        input_embed = item_embed + label_embed
        return input_embed

    def forward(self, input_dict, attn_mask=None):

        '''
        main model forward computation

        args
        ----
        input_dict : dict
            dict of input tensors, must contain keys ['item', 'label']
            - input['item'], shape (batch_size, max_seq_len, item_dim)
            - input['label'], shape (batch_size, max_seq_len, label_dim)
            - input['task'], shape (batch_size, 1, item_dim)
        attn_mask : torch.tensor
            shape `(n_item, n_item), used to mask self-attention, 
            True marks valid attention
        
        returns
        -------
        item_out/label_out : torch.tensor
            shape (batch, n_item, item_dim/label_dim)
        '''

        x = self.input_embed(input_dict)
        x = self.embed_dropout(x)

        for block in self.attn_layers:
            x = block(x, attn_mask=attn_mask) # (batch, n_items, embed_dim)

        item_out = self.item_head(x) # (batch, n_items, item_dim)
        label_out = self.label_head(x) # (batch, n_items, label_dim)

        return item_out, label_out

class MultitaskModelModule(pl.LightningModule):

    def __init__(self, 
                 n_task,
                 n_feature=2,
                 lr=1e-4,
                 teacher_forcing=0.5,
                 **model_kwargs):
        
        '''
        lightning module wrapper for the transformer nn.module above
        handles batch pre-processing, teacher forcing, loss/accuracy computation
        
        args
        ----
        n_task : int
            number of tasks being learned. used for loss/acc computation.
        n_feature : int
            number of features representing each item
        lr : float
            learning rate.
        teacher_forcing : float in [0,1]
            rate of teacher forcing during training.
        '''

        super().__init__()
        self.save_hyperparameters()

        self.transformer = MultitaskTransformer(**model_kwargs)
    
    def forward(self, batch):
        '''
        args
        ----
        batch : dict
            raw batch data from dataset.train_dataloader() or dataset.val_dataloader()
        
        returns
        -------
        dict
            {'pred': {'item': tensor, shape (batch, max_len, item_dim),
                        'label': tensor, shape (batch, max_len, label_dim)},
             'target': {'item': tensor, shape (batch, max_len, item_dim),
                          'label': tensor, shape (batch, max_len, label_dim)}
            }
        '''
        if torch.rand(1) < self.hparams.teacher_forcing:
            batch = self.process_batch(batch, teacher_forcing=True)
            item_pred, label_pred = self.forward_teacher_forcing(batch['src'])
        else:
            batch = self.process_batch(batch, teacher_forcing=False)
            item_pred, label_pred = self.forward_rollout(batch['src'])
        
        return {'pred': {'item': item_pred, 
                         'label': label_pred},
                'target': {'item': batch['target']['item'], 
                           'label': batch['target']['label']}}

    def process_batch(self, batch, teacher_forcing):

        '''
        prepare batch source and target data to feed into the body of transformer model
        args
        ----
        batch : dict
            raw batch data from dataset.train_dataloader() or dataset.val_dataloader()
        '''

        unsorted_items = batch['seq'].float()
        sorted_items = batch['out_seq'].float()
        task_tokens = unsorted_items[:,:1,:]

        if self.transformer.pos_encoding == 'label':
            unsorted_labels = batch['random_label'].float()
            sorted_labels = batch['sorted_label'].float()
        else: # self.transformer.pos_encoding == 'sinusoidal' or 'learnable'
            unsorted_labels = batch['idx_label'].float()
            sorted_labels = batch['sort_idx'].float()

        if not self.hparams.use_task_token: # truncate the task token (1st token in sequence)
            unsorted_items = unsorted_items[:,1:,:]
            unsorted_labels = unsorted_labels[:,1:,:]
            sorted_items = sorted_items[:,1:,:]
            sorted_labels = sorted_labels[:,1:,:]

        if teacher_forcing:
            seq_lens = [sum(seq.sum(-1)!=0) for seq in unsorted_items] # counting <eos>
            batch_size = unsorted_items.shape[0]
            # directly concat unsorted seq and sorted seq, move zero-padded items after unsorted seq to the end of sequence
            # so that the sequence comes out to be <...unsorted_items..., eos> <...sorted_items..., eos> <...padded items...>
            Xitems = torch.stack([torch.cat((unsorted_items[i][:seq_lens[i]], sorted_items[i], unsorted_items[i][seq_lens[i]:]))
                      for i in range(batch_size)]) # (batch_size, max_len*2, item_dim)
            Xlabels = torch.stack([torch.cat((unsorted_labels[i][:seq_lens[i]], sorted_labels[i], unsorted_labels[i][seq_lens[i]:])) 
                       for i in range(batch_size)]) # (batch_size, max_len*2, label_dim)
            return {'src': {'item': Xitems, 'label': Xlabels, 'task':task_tokens}, 
                    'target': {'item': sorted_items, 'label':sorted_labels}}
        else:
            return {'src': {'item': unsorted_items, 'label': unsorted_labels, 'task':task_tokens}, 
                    'target': {'item': sorted_items, 'label': sorted_labels}}

    def _calc_batch_seq_len(self, batch_items, teacher_forcing):
        '''
        computes batch sequence length (including <eos>)
        '''
        seq_lens = list(map(lambda seq: sum(seq.sum(-1)!=0), batch_items)) # counting the <eos>s
        if teacher_forcing:
            seq_lens = [int(l/2) for l in seq_lens] # divide by two cuz unsorted/sorted are concated
        return seq_lens

    def forward_teacher_forcing(self, batch):
        '''
        calls self.transformer.forward() while feeding ground-truth tokens

        args
        ----
        batch : dict
            input dict, must be pre-processed through self.process_batch(teacher_forcing=True)

        returns
        -------
        item_pred/label_pred : tensors
            shape (batch, max_len, item_dim/label_dim)
        '''

        # use mask to roll out all predictions at once while preventing attention to future items
        causal_mask = square_subsequent_mask(batch['item'].shape[1]).to(batch['item'].device)
        batch_item_pred, batch_label_pred = self.transformer.forward(batch, attn_mask=causal_mask) # (batch, max_len*2, dim)

        # extract the relevant predictions starting from <eos> in unsorted seq until last token in sorted seq
        seq_lens = self._calc_batch_seq_len(batch['item'], teacher_forcing=True)
        # first <eos> after unsorted items has index seq_len-1
        item_pred = [batch_item_pred[i][(l-1):(l-1)+l] for i, l in enumerate(seq_lens)]
        label_pred = [batch_label_pred[i][(l-1):(l-1)+l] for i, l in enumerate(seq_lens)]

        # pad zeros after predictions for each seq and stack
        max_len = int(batch['item'].shape[1] / 2)
        item_pred = [F.pad(item_pred[i], pad=(0,0,0,max_len-l), value=0.0) for i, l in enumerate(seq_lens)] # pad down
        item_pred = torch.stack(item_pred) # (batch, max_len, dim)
        label_pred = [F.pad(label_pred[i], pad=(0,0,0,max_len-l), value=0.0) for i, l in enumerate(seq_lens)]
        label_pred = torch.stack(label_pred)

        return item_pred, label_pred

    def forward_rollout(self, batch):
        '''
        rollout predictions for each sequence in batch by recycling the model's own predictions
        '''

        n_feature = self.hparams.n_feature
        n_task = self.hparams.n_task
        max_label = self.hparams.label_dim - n_task - 1

        def _task_logits_to_multihot(x):
            task = F.one_hot(x[..., :n_task+1].argmax(-1), num_classes=n_task+1) # (b,1,n_task+1)
            return F.pad(task, pad=(0, x.shape[-1]-n_task-1)) # (b,1,n_task+1)

        def _item_logits_to_multihot(x):
            feature_logits = rearrange(x[...,n_task+1:], 'b n (f d) -> b n f d', f=n_feature)# (b,1,3,5)
            features = F.one_hot(feature_logits.argmax(-1), num_classes=feature_logits.shape[-1]) # (b,1,3,5)
            item = rearrange(features, 'b n f d -> b n (f d)') # (b, 1, 15)
            return F.pad(item, pad=(n_task+1,0), value=0.0)

        def _label_logits_to_multihot(x):
            label = F.one_hot(x[..., n_task+1:].argmax(-1), num_classes=max_label) # (b,1,50)
            return F.pad(label, pad=(n_task+1,0), value=0.0)

        device = batch['item'].device
        batch_item_pred = torch.zeros_like(batch['item'], device=device) # (batch, 52, item_dim)
        batch_label_pred = torch.zeros_like(batch['label'], device=device) # (batch, 52, label_dim)

        # rollout separately for each sequence (TODO: any way to speed up?)
        seq_lens = self._calc_batch_seq_len(batch['item'], teacher_forcing=False) # counting <task> and <eos>
        for i, l in enumerate(seq_lens):
            src = {'item': batch['item'][i:i+1,:l], 
                'label': batch['label'][i:i+1,:l]} # remove all zero-padded items

            item_pred_logits = torch.zeros((1, l, src['item'].shape[-1]), device=device) # (1, n_step, item_dim)
            label_pred_logits = torch.zeros((1, l, src['label'].shape[-1]), device=device) # (1, n_step, label_dim)
            item_pred_multihot = torch.zeros((1, l, src['item'].shape[-1]), device=device)
            label_pred_multihot = torch.zeros((1, l, src['label'].shape[-1]), device=device)

            # ~rolling~ ~rolling~ ~rolling~
            for step in range(l):
                # get prediction, extract the last prediction
                step_src = {'item': torch.cat((src['item'], item_pred_multihot[:,:step]), dim=1), # (batch, n_items+n_predicted, item_dim)
                            'label': torch.cat((src['label'], label_pred_multihot[:,:step]), dim=1)} # (batch, n_items+n_predicted, label_dim)
                causal_mask = square_subsequent_mask(step_src['item'].shape[1]).to(step_src['item'].device)
                item_out, label_out = self.transformer.forward(step_src, attn_mask=causal_mask) # (batch, n_items+n_predicted, dim)
                item_out, label_out = item_out[:, -1:], label_out[:, -1:] # (batch, 1, dim)

                # keep track of predicted logits
                item_pred_logits[:,step] = item_out # (batch, 1, dim)
                label_pred_logits[:,step] = label_out # (batch, 1, dim)

                # keep track of multihot prediction (used for rollout)
                if (self.hparams.use_task_token and step==0) or (step==l-1):
                    # extract task and eos predictions
                    item = _task_logits_to_multihot(item_out)
                    label = F.pad(item, pad=(0,self.hparams.label_dim-self.hparams.item_dim), value=0.0)
                else:
                    item = _item_logits_to_multihot(item_out)
                    label = _label_logits_to_multihot(label_out)
                item_pred_multihot[:,step] = item
                label_pred_multihot[:,step] = label

            # fill predictions into batch prediction buffer
            batch_item_pred[i][:l] = item_pred_logits
            batch_label_pred[i][:l] = label_pred_logits

        return batch_item_pred, batch_label_pred

    def training_step(self, batch, batch_idx):
        result_dict = self.forward(batch)
        loss_dict, acc_dict = self._calc_loss_acc(result_dict)
        for loss in loss_dict.keys():
            self.log('train/%s'%loss, loss_dict[loss], on_epoch=True)
        for acc in acc_dict.keys():
            self.log('train/%s'%acc, acc_dict[acc], on_epoch=True)
        return sum(loss_dict.values())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            result_dict = self.forward(batch)
            loss_dict, acc_dict = self._calc_loss_acc(result_dict)
        for loss in loss_dict.keys():
            self.log('val/%s'%loss, loss_dict[loss], prog_bar=True)
        for acc in acc_dict.keys():
            self.log('val/%s'%acc, acc_dict[acc], prog_bar=True)
        return result_dict['pred']['label']

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            result_dict = self.forward(batch)
            loss_dict, acc_dict = self._calc_loss_acc(result_dict)
        for loss in loss_dict.keys():
            self.log('test/%s'%loss, loss_dict[loss])
        for acc in acc_dict.keys():
            self.log('test/%s'%acc, acc_dict[acc])
        return result_dict['pred']['label']

    def _calc_loss_acc(self, result_dict):
        use_task_token = self.hparams.use_task_token
        n_task = self.hparams.n_task
        item_pred, item_gt = result_dict['pred']['item'], result_dict['target']['item']
        label_pred, label_gt = result_dict['pred']['label'], result_dict['target']['label']
        if use_task_token:
            # loss/acc for all tasks combined
            task_loss_dict, task_acc_dict = self._calc_task_loss_acc(item_pred[:,0,:], item_gt[:,0,:])
            item_loss_dict, item_acc_dict = self._calc_item_loss_acc(item_pred[:,1:,:], item_gt[:,1:,:])
            label_loss_dict, label_acc_dict = self._calc_label_loss_acc(label_pred[:,1:,:], label_gt[:,1:,:])
            loss_dict = task_loss_dict | item_loss_dict | label_loss_dict # merge dict
            acc_dict = task_acc_dict | item_acc_dict | label_acc_dict
            # separate acc for each task
            batch_task = item_gt[:,0,:n_task+1].argmax(-1)
            for task in batch_task.unique():
                masked_item_pred, masked_item_gt = item_pred[batch_task==task], item_gt[batch_task==task]
                masked_label_pred, masked_label_gt = label_pred[batch_task==task], label_gt[batch_task==task]
                _, task_acc_dict = self._calc_task_loss_acc(masked_item_pred[:,0,:], masked_item_gt[:,0,:], 'task%d_'%task)
                _, item_acc_dict = self._calc_item_loss_acc(masked_item_pred[:,1:,:], masked_item_gt[:,1:,:], 'task%d_'%task)
                _, label_acc_dict = self._calc_label_loss_acc(masked_label_pred[:,1:,:], masked_label_gt[:,1:,:], 'task%d_'%task)
                acc_dict = acc_dict | task_acc_dict | item_acc_dict | label_acc_dict
        else:
            # TODO: compute per-task acc if self.hparams.add_task_embedding is True
            item_loss_dict, item_acc_dict = self._calc_item_loss_acc(item_pred, item_gt)
            label_loss_dict, label_acc_dict = self._calc_label_loss_acc(label_pred, label_gt)
            loss_dict = item_loss_dict | label_loss_dict # merge dict
            acc_dict = item_acc_dict | label_acc_dict
        return loss_dict, acc_dict

    def _calc_task_loss_acc(self, pred, gt, key_prefix=''):
        '''
        args
        ----
        pred/gt : tensor
            shape (batch, item_dim)
        '''
        n_task = self.hparams.n_task
        loss1 = nn.CrossEntropyLoss()(pred[:,:n_task+1], gt[:,:n_task+1].argmax(-1)) # the task token
        loss2 = nn.MSELoss()(pred[:,n_task+1:], gt[:,n_task+1:]) # should be all zeros
        loss = loss1 + loss2 # should have automatically averaged over batch dim
        
        acc = sum(pred[:,:n_task+1].argmax(-1)==gt[:,:n_task+1].argmax(-1))/pred.shape[0]

        return {'%stask_loss'%key_prefix: loss}, {'%stask_acc'%key_prefix: acc}

    def _calc_item_loss_acc(self, pred, gt, key_prefix=''):

        n_task = self.hparams.n_task
        # TODO: remove the assumption that each feature has equal dim
        n_feature = self.hparams.n_feature

        def _reshape_item_pred(t):
            return rearrange(t, 'n (f d) -> (n f) d', f=n_feature)

        seq_lens = list(map(lambda seq: sum(seq.sum(-1)!=0), gt)) # for each seq this is n_item+1 (including <EOS> token)
        n_seq = len(seq_lens) # batch dim

        '''
        first n_task+1 units: all zeros except for <eos> ([0,...,0,1])
        '''
        x = [pred[i,:seq_lens[i],:n_task+1] for i in range(n_seq)] # (n_seq, seq_len, n_task+1)
        y = [gt[i,:seq_lens[i],:n_task+1] for i in range(n_seq)] # (n_seq, seq_len, n_task+1)
        loss1 = [nn.MSELoss()(x[i][:-1], y[i][:-1]) for i in range(n_seq)]
        eos_loss = [nn.CrossEntropyLoss()(x[i][-1:], y[i][-1:].argmax(-1)) for i in range(n_seq)]
        loss1 = sum(loss1)/n_seq + sum(eos_loss)/n_seq

        '''
        n_task+1 onwards units: item feature prediction
        '''
        x = [pred[i,:seq_lens[i],n_task+1:] for i in range(n_seq)] # (n_seq, seq_len, n_feature x feature_dim)
        y = [gt[i,:seq_lens[i],n_task+1:] for i in range(n_seq)] # (n_seq, seq_len, n_feature x feature_dim)
        # compute loss for each seq and aggreagte: reshape multi-dim item prediction, then treat feature prediction as classification
        item_loss = [nn.CrossEntropyLoss()(_reshape_item_pred(x[i][:-1]), # (seq_len x n_feature, feature_dim)
                                           _reshape_item_pred(y[i][:-1]).argmax(-1))
                     for i in range(n_seq)] # excluding the last <eos> token
        eos_loss = [nn.MSELoss()(x[i][-1], y[i][-1]) for i in range(n_seq)] # should predict all 0s for <eos>
        loss2 = sum(item_loss)/n_seq + sum(eos_loss)/n_seq
        # compute onehot match accuracy (at the individual feature level)
        item_acc = [(_reshape_item_pred(x[i][:-1]).argmax(-1)==_reshape_item_pred(y[i][:-1]).argmax(-1)).sum()
                    for i in range(n_seq)]
        item_acc = [item_acc[i]/(n_feature*(seq_lens[i]-1)) for i in range(n_seq)] # denom -1 to not count the <eos> token
        item_acc = sum(item_acc)/n_seq

        return {'%sitem_loss'%key_prefix: loss1 + loss2}, {'%sitem_acc'%key_prefix: item_acc}

    def _calc_label_loss_acc(self, pred, gt, key_prefix=''):

        n_task = self.hparams.n_task
        seq_lens = list(map(lambda seq: sum(seq.sum(-1)!=0), gt)) # for each seq this is n_item+1 (including <EOS> token)
        n_seq = len(seq_lens) # batch dim

        '''
        first n_task+1 units: all zeros except for <eos> ([0,...,0,1])
        '''
        x = [pred[i,:seq_lens[i],:n_task+1] for i in range(n_seq)] # (n_seq, seq_len, n_task+1)
        y = [gt[i,:seq_lens[i],:n_task+1] for i in range(n_seq)] # (n_seq, seq_len, n_task+1)
        loss1 = [nn.MSELoss()(x[i][:-1], y[i][:-1]) for i in range(n_seq)]
        eos_loss = [nn.CrossEntropyLoss()(x[i][-1:], y[i][-1:].argmax(-1)) for i in range(n_seq)]
        loss1 = sum(loss1)/n_seq + sum(eos_loss)/n_seq

        '''
        n_task+1 onwards units: sorted label prediction
        '''
        x = [pred[i,:seq_lens[i],n_task+1:] for i in range(n_seq)] # (n_seq, seq_len, max_label)
        y = [gt[i,:seq_lens[i],n_task+1:] for i in range(n_seq)] # (n_seq, seq_len, max_label)
        # compute loss for each seq and aggreagte: sort label prediction as classification
        label_loss = [nn.CrossEntropyLoss()(x[i][:-1], y[i][:-1].argmax(-1)) for i in range(n_seq)] # excluding the last <eos> token
        eos_loss = [nn.MSELoss()(x[i][-1], y[i][-1]) for i in range(n_seq)] # should predict all 0s for <eos>
        loss2 = sum(label_loss)/n_seq + sum(eos_loss)/n_seq
        # compute onehot match accuracy (at individual label level)
        label_acc = [(x[i][:-1].argmax(-1)==y[i][:-1].argmax(-1)).sum() for i in range(n_seq)]
        label_acc = [label_acc[i]/(seq_lens[i]-1) for i in range(n_seq)] # denom -1 to not count the <eos> token
        label_acc = sum(label_acc)/n_seq
        
        return {'%slabel_loss'%key_prefix: loss1 + loss2}, {'%slabel_acc'%key_prefix: label_acc}

    @torch.no_grad()
    def get_attention_maps(self, batch, batch_processed, teacher_forcing):
        '''
        extract the attention matrices of the whole model for a single batch
        code modified from model forward pass

        returns
        -------
        attention_maps : list
            list of (batch, n_head, n_item, n_item) tensors, length = depth
        '''

        self.transformer.eval()

        if not batch_processed:
            batch = self.process_batch(batch, teacher_forcing=teacher_forcing)
        batch = batch['src']

        if teacher_forcing:
            x = self.transformer.input_embed(batch)
            causal_mask = square_subsequent_mask(batch['item'].shape[1]).to(batch['item'].device)

            attn_maps = []
            for block in self.transformer.attn_layers:
                _, attn_map = block.self_attn(x, x, x, attn_mask=causal_mask, return_attn=True)
                attn_maps.append(attn_map)
                x = block(x, attn_mask=causal_mask)
            return attn_maps
        
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def get_encoded_reps(self, batch, batch_processed, teacher_forcing):
        '''
        extract the encoder output at input embedding layer and each encoder layer

        returns
        -------
        dict with keys 'input_embed', 'attn_embed'.
            dict['input_embed']: torch.tensor, shape (batch, n_items, embed_dim)
            dict['attn_embed'] : torch.tensor, shape (depth, batch, n_items, embed_dim)
        '''

        self.transformer.eval()

        if not batch_processed:
            batch = self.process_batch(batch, teacher_forcing=teacher_forcing)
        batch = batch['src']

        if teacher_forcing:
            encoded_reps = {}
            x = self.transformer.input_embed(batch)
            encoded_reps['input_embed'] = x

            causal_mask = square_subsequent_mask(batch['item'].shape[1]).to(batch['item'].device)
            encoded_reps['attn_embed'] = []
            for block in self.transformer.attn_layers:
                x = block(x, attn_mask=causal_mask)
                encoded_reps['attn_embed'].append(x)
            encoded_reps['attn_embed'] = torch.stack(encoded_reps['attn_embed'])
            return encoded_reps

        else:
            raise NotImplementedError()

    @torch.no_grad()
    def ablated_forward(self, ablate_head_list, batch, teacher_forcing, 
                        keep_attn_to_self=None, keep_attn_to_task=None, keep_attn_to_eos=None,
                        keep_attn_to_same_shape=None, keep_attn_to_same_color=None, keep_attn_to_next=None):

        '''
        modified forward computation by ablating the given attention heads (setting weights to zero) in the model

        args
        ----
        ablate_head_dict : list of attention heads to ablate
            each sublist denotes the head indices at that layer to be ablated
        '''

        # check ablate_head_dict specification
        assert len(ablate_head_list) == self.hparams.depth
        for l, heads in enumerate(ablate_head_list):
            if len(heads)==0: continue
            elif type(self.hparams.n_heads)==int: assert max(heads)<self.hparams.n_heads
            else: assert max(heads)<self.hparams.n_heads[l]
        # set default attn mask bools
        if keep_attn_to_self is None: keep_attn_to_self = [False]*self.hparams.depth
        if keep_attn_to_task is None: keep_attn_to_task = [False]*self.hparams.depth
        if keep_attn_to_eos is None: keep_attn_to_eos = [False]*self.hparams.depth
        if keep_attn_to_same_shape is None: keep_attn_to_same_shape = [False]*self.hparams.depth
        if keep_attn_to_same_color is None: keep_attn_to_same_color = [False]*self.hparams.depth
        if keep_attn_to_next is None: keep_attn_to_next = [False]*self.hparams.depth

        self.transformer.eval()

        tf_batch = self.process_batch(batch, teacher_forcing=True)
        batch = self.process_batch(batch, teacher_forcing=teacher_forcing)
        bsz, n_token, _ = tf_batch['src']['item'].shape
        ablate_masks = {'L%d'%l: self.get_attn_ablation_mask(bsz=bsz, n_token=n_token, batch=tf_batch, 
                                                             n_heads=self.transformer.attn_layers[l].self_attn.n_heads,
                                                             ablate_heads=ablate_head_list[l],
                                                             keep_attn_to_self=keep_attn_to_self[l], 
                                                             keep_attn_to_task=keep_attn_to_task[l], 
                                                             keep_attn_to_eos=keep_attn_to_eos[l],
                                                             keep_attn_to_same_shape=keep_attn_to_same_shape[l], 
                                                             keep_attn_to_same_color=keep_attn_to_same_color[l],
                                                             keep_attn_to_next=keep_attn_to_next[l]) 
                        for l in range(self.hparams.depth)}

        def _ablated_forward(batch, attn_mask, ablate_masks):
            '''
            modified from transformer.MultitaskTransformer.forward(), masks attention by the given ablation specs
            '''
            x = self.transformer.input_embed(batch)

            attn_maps = []
            for l, layer in enumerate(self.transformer.attn_layers):

                # modified from layers.Attention.forward()
                q, k, v = x.clone(), x.clone(), x.clone()

                bsz, q_len, _ = q.shape
                _, k_len, _ = k.shape

                # check attention mask
                if attn_mask is not None:
                    assert attn_mask.dtype == torch.bool
                    assert attn_mask.shape == (q_len, k_len) or attn_mask.shape == (bsz, q_len, k_len)

                # project q/k/v
                q = layer.self_attn.q_project(q) # (batch, n_items, embed_dim)
                k = layer.self_attn.k_project(k)
                v = layer.self_attn.v_project(v)
                # (batch, n_items, n_heads x dim_head) -> (batch, n_heads, n_items, dim_head)
                q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=layer.self_attn.n_heads), (q,k,v))

                attn = torch.matmul(q, k.transpose(-1,-2)) * layer.self_attn.scale # (batch, n_heads, n_target_items, n_source_items)
                if attn_mask is not None:
                    if attn_mask.dim()==3:
                        attn_mask = attn_mask.unsqueeze(1) # add an n_head dim for broadcast add
                    attn_mask = attn_mask.to(attn.device)
                    # mark -inf where mask==False
                    ninf_mask = torch.zeros_like(attn_mask, dtype=q.dtype, device=attn.device)
                    ninf_mask.masked_fill_(attn_mask==False, float('-inf'))
                    attn += ninf_mask
                attn = F.softmax(attn, dim=-1)

                # modified part: mask attention weights beyond future masking (masked head will have original weight where ablate_mask==0 and 0. otherwise)
                attn.masked_fill_(ablate_masks['L%d'%l].to(attn.device).bool(), value=0.0)
                
                out = torch.matmul(layer.self_attn.attn_dropout(attn), v) # (batch, n_heads, n_target_items, n_source_items) x (batch, n_heads, n_source_item, dim_head)
                out = rearrange(out, 'b h n d -> b n (h d)') # (batch, n_target_items, n_heads x dim_head)
                out = layer.self_attn.to_out(out) # (batch, n_target_items, embed_dim)

                # modified from layers.AttentionLayer.forward()
                attn_maps.append(attn)
                x = layer.norm1(x + out)
                x = layer.norm2(x + layer.mlp(x))

            item_out = self.transformer.item_head(x) # (batch, n_items, item_dim)
            label_out = self.transformer.label_head(x) # (batch, n_items, label_dim)

            return item_out, label_out, attn_maps

        if teacher_forcing:
            causal_mask = square_subsequent_mask(n_token).to(batch['src']['item'].device)
            item_out, label_out, attn_maps = _ablated_forward(batch['src'], attn_mask=causal_mask, ablate_masks=ablate_masks)

            # extract the relevant predictions starting from <eos> in unsorted seq until last token in sorted seq
            seq_lens = self._calc_batch_seq_len(batch['src']['item'], teacher_forcing=True)
            # first <eos> after unsorted items has index seq_len-1
            item_pred = [item_out[i][(l-1):(l-1)+l] for i, l in enumerate(seq_lens)]
            label_pred = [label_out[i][(l-1):(l-1)+l] for i, l in enumerate(seq_lens)]

            # pad zeros after predictions for each seq and stack
            max_len = int(batch['src']['item'].shape[1] / 2)
            item_pred = [F.pad(item_pred[i], pad=(0,0,0,max_len-l), value=0.0) for i, l in enumerate(seq_lens)] # pad down
            item_pred = torch.stack(item_pred) # (batch, max_len, dim)
            label_pred = [F.pad(label_pred[i], pad=(0,0,0,max_len-l), value=0.0) for i, l in enumerate(seq_lens)]
            label_pred = torch.stack(label_pred)

            result_dict = {'pred': {'item': item_pred, 
                                    'label': label_pred},
                           'target': {'item': batch['target']['item'], 
                                      'label': batch['target']['label']}}
            return result_dict, attn_maps
            
        else:
            n_feature = self.hparams.n_feature
            n_task = self.hparams.n_task
            max_label = self.hparams.label_dim - n_task - 1

            def _task_logits_to_multihot(x):
                task = F.one_hot(x[..., :n_task+1].argmax(-1), num_classes=n_task+1) # (b,1,n_task+1)
                return F.pad(task, pad=(0, x.shape[-1]-n_task-1)) # (b,1,n_task+1)

            def _item_logits_to_multihot(x):
                feature_logits = rearrange(x[...,n_task+1:], 'b n (f d) -> b n f d', f=n_feature)# (b,1,3,5)
                features = F.one_hot(feature_logits.argmax(-1), num_classes=feature_logits.shape[-1]) # (b,1,3,5)
                item = rearrange(features, 'b n f d -> b n (f d)') # (b, 1, 15)
                return F.pad(item, pad=(n_task+1,0), value=0.0)

            def _label_logits_to_multihot(x):
                label = F.one_hot(x[..., n_task+1:].argmax(-1), num_classes=max_label) # (b,1,50)
                return F.pad(label, pad=(n_task+1,0), value=0.0)

            device = batch['src']['item'].device
            batch_item_pred = torch.zeros_like(batch['src']['item'], device=device) # (batch, 52, item_dim)
            batch_label_pred = torch.zeros_like(batch['src']['label'], device=device) # (batch, 52, label_dim)

            # rollout separately for each sequence (TODO: any way to speed up?)
            seq_lens = self._calc_batch_seq_len(batch['src']['item'], teacher_forcing=False) # counting <task> and <eos>
            for i, l in enumerate(seq_lens):
                src = {'item': batch['src']['item'][i:i+1,:l], 
                       'label': batch['src']['label'][i:i+1,:l]} # remove all zero-padded items

                item_pred_logits = torch.zeros((1, l, src['item'].shape[-1]), device=device) # (1, n_step, item_dim)
                label_pred_logits = torch.zeros((1, l, src['label'].shape[-1]), device=device) # (1, n_step, label_dim)
                item_pred_multihot = torch.zeros((1, l, src['item'].shape[-1]), device=device)
                label_pred_multihot = torch.zeros((1, l, src['label'].shape[-1]), device=device)

                # ~rolling~ ~rolling~ ~rolling~
                for step in range(l):
                    # get prediction, extract the last prediction
                    step_src = {'item': torch.cat((src['item'], item_pred_multihot[:,:step]), dim=1), # (batch, n_items+n_predicted, item_dim)
                                'label': torch.cat((src['label'], label_pred_multihot[:,:step]), dim=1)} # (batch, n_items+n_predicted, label_dim)
                    causal_mask = square_subsequent_mask(step_src['item'].shape[1]).to(step_src['item'].device)
                    step_ablate_masks = {k: ablate_masks[k][i:i+1, :, :l+step, :l+step] for k in ablate_masks}
                    item_out, label_out, _ = _ablated_forward(step_src, attn_mask=causal_mask, ablate_masks=step_ablate_masks) # (batch, n_items+n_predicted, dim)
                    item_out, label_out = item_out[:, -1:], label_out[:, -1:] # (batch, 1, dim)

                    # keep track of predicted logits
                    item_pred_logits[:,step] = item_out # (batch, 1, dim)
                    label_pred_logits[:,step] = label_out # (batch, 1, dim)

                    # keep track of multihot prediction (used for rollout)
                    if (self.hparams.use_task_token and step==0) or (step==l-1):
                        # extract task and eos predictions
                        item = _task_logits_to_multihot(item_out)
                        label = F.pad(item, pad=(0,self.hparams.label_dim-self.hparams.item_dim), value=0.0)
                    else:
                        item = _item_logits_to_multihot(item_out)
                        label = _label_logits_to_multihot(label_out)
                    item_pred_multihot[:,step] = item
                    label_pred_multihot[:,step] = label

                batch_item_pred[i][:l] = item_pred_logits
                batch_label_pred[i][:l] = label_pred_logits

            result_dict = {'pred': {'item': batch_item_pred, 
                                    'label': batch_label_pred},
                           'target': {'item': batch['target']['item'], 
                                      'label': batch['target']['label']}}
            return result_dict

    @torch.no_grad()
    def get_attn_ablation_mask(self, bsz, n_token, batch, n_heads, ablate_heads, 
                               keep_attn_to_self, keep_attn_to_task, keep_attn_to_eos,
                               keep_attn_to_same_shape, keep_attn_to_same_color, keep_attn_to_next):

        # create the mask for a single ablated head
        # by default it is fully ablated; all attention weights will be masked to 0.0
        ablate_head_mask = torch.ones((bsz, n_token, n_token))
        
        if keep_attn_to_self:
            # set diagnal units in mask back to 0 (allowing attention to self)
            diag_mask = repeat(torch.diag(torch.ones(n_token)).int(), 
                                'n1 n2 -> b n1 n2', b=bsz)
            ablate_head_mask.masked_fill_(diag_mask.bool(), value=0.0)

        if keep_attn_to_task:
            # allow attention to task tokens
            for i in range(bsz):
                seq = batch['src']['item'][i]
                task_tokens = torch.where(seq[:,:self.hparams.n_task].sum(-1) != 0)[0]
                for t in task_tokens:
                    ablate_head_mask[i,:,t] = torch.zeros(n_token) # set attention to task tokens in mask back to 0
        
        if keep_attn_to_eos:
            # allow attention to eos
            for i in range(bsz):
                seq = batch['src']['item'][i]
                eos_tokens = torch.where(seq[:,self.hparams.n_task]==1)[0]
                for e in eos_tokens:
                    ablate_head_mask[i,:,e] = torch.zeros(n_token) # set attention to eos tokens in mask back to 0

        if keep_attn_to_same_shape or keep_attn_to_same_color:
            # allow attention to other items of same shape/color
            same_feature_mask = torch.zeros_like(ablate_head_mask) # (bsz, n_token, n_token)
            for i in range(bsz):
                seq = batch['src']['item'][i]
                token_features = seq[:,self.hparams.n_task+1:].view((n_token,3,5)).argmax(-1)
                items = (seq[:,:self.hparams.n_task+1].sum(-1)==0) & (seq[:,:].sum(-1)!=0) # items marked with True, task/eos/pad with False
                for q, query in enumerate(seq):
                    if not items[q]: continue
                    if keep_attn_to_same_shape and not keep_attn_to_same_color:
                        same_feature_mask[i][q] = (token_features[:,0]==token_features[q,0]) & items
                    elif keep_attn_to_same_color and not keep_attn_to_same_shape:
                        same_feature_mask[i][q] = (token_features[:,1]==token_features[q,1]) & items
                    else:
                        same_feature_mask[i][q] = ((token_features[:,0]==token_features[q,0]) | (token_features[:,1]==token_features[q,1])) & items
            ablate_head_mask.masked_fill_(same_feature_mask.bool(), value=0.0) # for each item query set attention to other items with the same shape/color back to 0

        if keep_attn_to_next:
            # allow attention to the next token
            next_item_mask = torch.zeros_like(ablate_head_mask) # (bsz, n_token, n_token)
            seq_lens = self._calc_batch_seq_len(batch['src']['item'], teacher_forcing=True) # counts <task> or <eos>
            for i in range(bsz):
                items = (batch['src']['item'][i,:,:self.hparams.n_task+1].sum(-1)==0) & (batch['src']['item'][i].sum(-1)!=0) # items marked with True, task/eos/pad with False
                labels = batch['src']['label'][i,:,self.hparams.n_task+1:].argmax(-1)[:seq_lens[i]*2] # labels for all tokens in source+query
                if self.hparams.use_task_token:
                    labels[0] = -1 # set <task> and <eos> positions to -1
                    labels[seq_lens[i]-1] = -1
                    labels[seq_lens[i]] = -1
                    labels[seq_lens[i]*2-1] = -1
                    sorted_labels = batch['target']['label'][i,:,self.hparams.n_task+1:].argmax(-1)[:seq_lens[i]][1:-1] # labels for output items only
                else:
                    labels[seq_lens[i]-1] = -1
                    labels[seq_lens[i]*2-1] = -1
                    sorted_labels = batch['target']['label'][i,:,self.hparams.n_task+1:].argmax(-1)[:seq_lens[i]][:-1] # labels for output items only
                for q, cur_label in enumerate(labels):
                    if not items[q]: continue # next token undefined
                    output_ind = (sorted_labels==cur_label).nonzero().item()
                    if output_ind+1 < len(sorted_labels): # this item is not the last item in out seq
                        next_item_ind = (labels==sorted_labels[output_ind+1]).nonzero()[0].item() # find first occurrance among source items
                        next_item_mask[i,q,next_item_ind] = 1.0
            ablate_head_mask.masked_fill_(next_item_mask.bool(), value=0.0)

        # set mask for each head in ablate_heads, 0s for all other non-ablate heads
        ablate_mask = torch.zeros((bsz, n_heads, n_token, n_token))
        for h in ablate_heads:
            ablate_mask[:,h] = ablate_head_mask

        return ablate_mask