from huggingface_hub import HfFileSystem, hf_hub_download
import gin
from ginpipe.core import gin_configure_externals
from encodecmae.hub import get_model
import torch
from tqdm import tqdm
from encodec import EncodecModel
import numpy as np
from pathlib import Path
from .helpers import identify_state_dict_version, v1_to_v2_state_dict

class ChunkDiffusionSampler:
  def __init__(self, model):
    self.model = model
    self.model.unconditional_probability = 0
    self.model.eval()
    self.ec = EncodecModel.encodec_model_24khz()
    self.ec.to(self.model.device)

  def encode(self, x):
    codes = []
    win_size = int(self.model.chunk_size*self.model.fs)
    with torch.no_grad():
      for i in range(0,x.shape[0],win_size):
        X = {'wav': torch.from_numpy(x[i:i+win_size]).unsqueeze(0).to(self.model.device), 'wav_lens': torch.tensor([win_size,], device=self.model.device)}
        self.model.encode(X)
        codes.append(X['cls_token'][0,0])
    return torch.stack(codes)

  def sample(self, codes, steps=100, guidance_strength=4.0):
    out_length=int(codes.shape[0]*self.model.chunk_size*self.model.fs)
    win_size = int(self.model.chunk_size*self.model.fs)
    outs = np.zeros(out_length,)
    with torch.no_grad():
      for i,c in enumerate(tqdm(codes)):    
        rec = self.model.sample(c[None,None,:].to(self.model.device),steps=steps,guidance_strength=guidance_strength, length=int(self.model.chunk_size*self.model.rate))
        ecmae_features = rec[-1]
        reconstruction = self.ec.decoder(ecmae_features.transpose(1,2))
        outs[i*win_size:(i+1)*win_size] += reconstruction[0,0].detach().cpu().numpy()
    return outs

class ChunkSampler:
  def __init__(self, model):
    self.model = model
    self.model.eval()
    self.ec = EncodecModel.encodec_model_24khz()
    self.ec.to(self.model.device)

  def encode(self, x):
    codes = []
    win_size = int(self.model.chunk_size*self.model.fs)
    with torch.no_grad():
      for i in range(0,x.shape[0],win_size):
        X = {'wav': torch.from_numpy(x[i:i+win_size]).unsqueeze(0).to(self.model.device), 'wav_lens': torch.tensor([win_size,], device=self.model.device)}
        self.model.encode(X)
        codes.append(X['visible_embeddings'][0])
    return codes

  def sample(self, codes):
    out_length=int(len(codes)*self.model.chunk_size*self.model.fs)
    win_size = int(self.model.chunk_size*self.model.fs)
    out = np.zeros((out_length,))
    with torch.no_grad():
      for i, c in enumerate(tqdm(codes)):
        X = {'visible_embeddings': c.unsqueeze(0), 'feature_padding_mask': torch.zeros_like(c[:,0]).unsqueeze(0)}
        self.model.decode(X)
        wave_i = self.ec.decoder(X['reconstruction'].transpose(1,2))[0,0]
        out[i*win_size:i*win_size+wave_i.shape[0]] += wave_i.detach().cpu().numpy()
    return out

def get_available_models():
    fs = HfFileSystem()
    available_models = [x['name'].split('/')[-1] for x in fs.ls('lpepino/encodecmae2wav/models', refresh=True)]
    return available_models

def load_model(name, device='cuda:0'):
    config_str = gin.config_str()
    gin.clear_config()
    available_models = get_available_models()
    if name not in available_models:
      if Path(name).exists():
        ckpt_file = Path(name, 'weights.pt')
        config_file = Path(name, 'config.gin')
      else:
        raise Exception('{} is not a valid model name. Valid names are: {} or local paths.'.format(name,available_models))
    else:
        ckpt_file = hf_hub_download(repo_id='lpepino/encodecmae2wav',filename='models/{}/weights.pt'.format(name))
        config_file = hf_hub_download(repo_id='lpepino/encodecmae2wav',filename='models/{}/config.gin'.format(name))
    gin_configure_externals({'module_list_str': 'encodecmae_to_wav.tasks.models:models', 'module_list':[]})
    gin.parse_config_file(config_file)
    model = get_model()()
    ckpt = torch.load(ckpt_file, map_location='cpu')
    #sd_version = identify_state_dict_version(ckpt['state_dict'])
    #if sd_version == '1':
    #  ckpt['state_dict'] = v1_to_v2_state_dict(ckpt['state_dict'])
    model.load_state_dict(ckpt['state_dict'])
    gin.clear_config()
    gin.parse_config(config_str)

    with open(config_file,'r') as f:
      lines = f.readlines()
      max_audio_duration = [float(xi.split('=')[1].strip()) for xi in lines if 'MAX_AUDIO_DURATION' in xi][0]
      meta_model = [xi.split('=')[1].strip().replace("'","").replace("\"","") for xi in lines if 'META_MODEL' in xi][0]
    model.chunk_size=max_audio_duration
    model.fs=24000
    model.rate = model.encoder.downsample_factor
    model.to(device)
    
    if meta_model == 'ChunkDiffusionSampler':
      return ChunkDiffusionSampler(model)
    elif meta_model == 'ChunkSampler':
      return ChunkSampler(model)
