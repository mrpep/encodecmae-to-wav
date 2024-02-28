import pytorch_lightning as pl
from .encoders import TransformerCLSEncoder
from encodecmae.tasks.models.transformers import MultiHeadAttention, SinusoidalPositionalEmbeddings, TransformerEncoder
from functools import partial
from math import pi
import torch

class TransformerAEDiffusion(pl.LightningModule):
    def __init__(self, encodecmae_model='base', num_cls_tokens=5, num_encoder_layers=4, 
                       num_denoiser_layers=4, num_heads=8,
                       optimizer=None, lr_scheduler=None,
                       signal_to_generate='visible_embeddings',
                       signal_dim=768,
                       unconditional_probability=0.1,
                       conditioning_method='concat'):

        super().__init__()
        self.encoder = TransformerCLSEncoder(num_cls_tokens=num_cls_tokens, 
                                             num_encoder_layers=num_encoder_layers, 
                                             encodecmae_model=encodecmae_model,
                                             num_heads=num_heads)
        self.denoiser = TransformerEncoder(self.encoder.out_channels,
                                          attention_layer=partial(MultiHeadAttention,
                                                                  model_dim=self.encoder.out_channels,
                                                                  num_heads=num_heads),
                                          num_layers=num_denoiser_layers, compile=False)
        self.in_adapter = torch.nn.Linear(signal_dim, self.encoder.out_channels)
        self.out_adapter = torch.nn.Linear(self.encoder.out_channels, signal_dim)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.time_embedding = torch.nn.Sequential(torch.nn.Linear(1,self.encoder.out_channels), torch.nn.ReLU(), torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels))
        self.signal_to_generate = signal_to_generate
        self.signal_dim=signal_dim
        self.unconditional_probability = unconditional_probability
        self.conditioning_method = conditioning_method
        self.num_denoiser_layers = num_denoiser_layers

    def add_diffusion_noise(self, x):
        xin = x[self.signal_to_generate]
        noise = torch.randn_like(xin)
        sigmas = torch.rand(xin.shape[0], device=xin.device)[:,None,None]
        angle = sigmas * pi / 2
        alphas, betas = torch.cos(angle), torch.sin(angle)
        x_noisy = alphas * xin + betas * noise
        v_target = alphas * noise - betas * xin

        x['noisy'] = x_noisy
        x['target'] = v_target
        x['sigmas'] = sigmas

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.add_diffusion_noise(x)
        if self.conditioning_method == 'concat':
            dec_in = torch.cat([x['cls_token'], self.time_embedding(x['sigmas']), self.encoder.pos_encoder(self.in_adapter(x['noisy']))], dim=1)
            padding_mask = torch.cat([torch.zeros((x[self.signal_to_generate].shape[0], self.encoder.num_cls_tokens+1), device=x[self.signal_to_generate].device), x['feature_padding_mask']], dim=1)
        else:
            raise Exception('Unrecognized conditioning method')

        dec_out = self.denoiser(dec_in, padding_mask=padding_mask)
        x['denoised'] = self.out_adapter(dec_out[:,self.encoder.num_cls_tokens+1:,:])

        return x

    def encode(self, x):
        self.encoder(x)
        mask = torch.rand(size=(x['cls_token'].shape[0],), device=x['cls_token'].device)>self.unconditional_probability
        x['cls_token'] = x['cls_token']*mask[:,None,None]
        return x

    def sample(self, code, steps=10, length=75, guidance_strength=0):
        with torch.no_grad():
            sigmas = torch.linspace(1,0,steps+1).to(code.device)
            ts = self.time_embedding(sigmas[:,None,None])
            angle = sigmas * pi / 2
            alphas, betas = torch.cos(angle), torch.sin(angle)
            x_noisy = torch.randn((1,length,self.signal_dim), device=code.device)
            noise_sequence = []
            for i in range(steps):
                denoiser_in = torch.cat([code, ts[i,:,:].unsqueeze(0), self.encoder.pos_encoder(self.in_adapter(x_noisy))], dim=1)
                v_pred = self.out_adapter(self.denoiser(denoiser_in, None)[:,self.encoder.num_cls_tokens+1:])

                if guidance_strength > 0:
                    denoiser_in_unc = torch.cat([torch.zeros_like(code), ts[i,:,:].unsqueeze(0), self.encoder.pos_encoder(self.in_adapter(x_noisy))], dim=1)
                    v_pred_unc = self.out_adapter(self.denoiser(denoiser_in_unc, None)[:,self.encoder.num_cls_tokens+1:])
                    v_pred += guidance_strength*(v_pred - v_pred_unc)
                    
                x_pred = alphas[i] * x_noisy - betas[i] * v_pred
                noise_pred = betas[i] * x_noisy + alphas[i] * v_pred
                x_noisy = alphas[i + 1] * x_pred + betas[i + 1] * noise_pred
                noise_sequence.append(x_noisy)
        return noise_sequence

    def calculate_loss(self, x):
        loss = {'loss': torch.nn.functional.mse_loss(x['target'], x['denoised'])}
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