from encodecmae import load_model
from encodecmae.tasks.models.transformers import TransformerEncoder, MultiHeadAttention, SinusoidalPositionalEmbeddings

import pytorch_lightning as pl
import torch

from functools import partial

class TransformerNet(torch.nn.Module):
    def __init__(self, model_dim, embedding_dim, num_decoder_layers=4, num_decoder_heads=12, apply_positional_encodings=True):
        super().__init__()
        self.posenc = SinusoidalPositionalEmbeddings(model_dim)
        self.trans = TransformerEncoder(model_dim=model_dim, num_layers=num_decoder_layers, attention_layer=partial(MultiHeadAttention,model_dim=model_dim, num_heads=num_decoder_heads),compile=False)
        self.lin = torch.nn.Linear(model_dim, embedding_dim)
        self.apply_positional_encodings = apply_positional_encodings

    def forward(self, x, padding_mask = None):
        if self.apply_positional_encodings:
            y = self.posenc(x)
        else:
            y = x
        y = self.trans(y, padding_mask=padding_mask)
        return self.lin(y)

class EnCodecDecoder(pl.LightningModule):
    def __init__(self, encodecmae_model='base', decoder_model=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.encoder = load_model(encodecmae_model)
        self.encoder.downsample_factor = 75
        self.encoder.visible_encoder.compile = False
        model_dim = self.encoder.visible_encoder.model_dim
        self.decoder = decoder_model(model_dim=model_dim, embedding_dim=128)
        self.optimizer=optimizer
        self.lr_scheduler = lr_scheduler
        
    def encode(self, x):
        with torch.no_grad():
            self.encoder.encode_wav(x)
            self.encoder.mask(x, ignore_mask=True)
            self.encoder.encode_visible(x)

    def decode(self, x):
        x['reconstruction'] = self.decoder(x['visible_embeddings'],padding_mask=x['feature_padding_mask'])
        return x

    def forward(self, x):
        self.encode(x)
        self.decode(x)
        return x

    def calculate_loss(self, x):
        loss = {'loss': torch.nn.functional.l1_loss(x['wav_features'], x['reconstruction'])}
        return loss
        
    def training_step(self,x, batch_idx):
        x = self(x)
        losses = self.calculate_loss(x)
        self.log_results(x,losses,'train')

        return losses['loss']

    def validation_step(self,x, batch_idx):
        x = self(x)
        losses = self.calculate_loss(x)
        self.log_results(x,losses,'val')

    def log_results(self,x,losses,prefix):
        self.log_dict({'{}_{}'.format(prefix,k): v for k,v in losses.items()})

    def configure_optimizers(self):
        opt = self.optimizer(self.trainer.model.parameters())
        if self.lr_scheduler is not None:
            if self.lr_scheduler.__name__ == 'SequentialLR':
                binds = gin.get_bindings('torch.optim.lr_scheduler.SequentialLR')
                lr_scheduler = self.lr_scheduler(opt, schedulers=[s(opt) for s in binds['schedulers']])
            else:
                lr_scheduler = self.lr_scheduler(opt) if self.lr_scheduler is not None else None
        else:
            lr_scheduler = None
        del self.optimizer
        del self.lr_scheduler
        opt_config = {'optimizer': opt}
        if lr_scheduler is not None:
            opt_config['lr_scheduler'] = {'scheduler': lr_scheduler,
                                          'interval': 'step',
                                          'frequency': 1}
        return opt_config

    def set_optimizer_state(self, state):
        pass
