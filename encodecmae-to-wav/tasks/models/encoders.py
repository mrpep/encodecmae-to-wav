from encodecmae import load_model
from encodecmae.tasks.models.transformers import TransformerEncoder, MultiHeadAttention, SinusoidalPositionalEmbeddings

import torch

from functools import partial

class TransformerCLSEncoder(torch.nn.Module):
    def __init__(self, encodecmae_model='base',num_heads=12, num_encoder_layers=2,num_cls_tokens=1, device='cpu', downsample_factor=75):
        super().__init__()
        self.encodecmae_model = load_model(encodecmae_model, device=device)
        self.encodecmae_model.visible_encoder.compile=False
        model_dim = self.encodecmae_model.visible_encoder.model_dim
        self.encoder = TransformerEncoder(model_dim,attention_layer=partial(MultiHeadAttention,model_dim=model_dim,num_heads=num_heads),num_layers=num_encoder_layers, compile=False)
        self.cls_tokens = torch.nn.Embedding(num_cls_tokens,model_dim)
        self.pos_encoder = SinusoidalPositionalEmbeddings(model_dim)
        self.num_cls_tokens = num_cls_tokens
        self.out_channels = model_dim
        self.downsample_factor = downsample_factor

    def forward(self, x):
        with torch.no_grad():
            self.encodecmae_model.encode_wav(x)
            self.encodecmae_model.mask(x, ignore_mask=True)
            self.encodecmae_model.encode_visible(x)
        cls_tokens = torch.tile(self.cls_tokens(torch.arange(self.num_cls_tokens, device=x['visible_embeddings'].device).unsqueeze(0)),(x['visible_embeddings'].shape[0],1,1))
        enc_in = torch.cat([cls_tokens, self.pos_encoder(x['visible_embeddings'])],dim=1)
        padding_mask = torch.cat([torch.zeros((x['visible_embeddings'].shape[0], self.num_cls_tokens), device=x['visible_embeddings'].device), x['feature_padding_mask']], dim=1)
        enc_out = self.encoder(enc_in, padding_mask)
        cls = enc_out[:,:self.num_cls_tokens]
        x['cls_token'] = cls
        return x