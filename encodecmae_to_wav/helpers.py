def identify_state_dict_version(sd):
    version = None
    for k,v in sd.items():
        if 'wav_encoder.model' in k:
            version = '1'
        elif 'wav_encoder.encoder' in k:
            version = '2'
    return version

def v1_to_v2_state_dict(v1_sd):
    v2_sd = {}
    is_nested = False
    for k,v in v1_sd.items():
        if k.startswith('encoder'):
            is_nested = True
    for k,v in v1_sd.items():
        if is_nested:
            k2 = 'encoder.'.join(k.split('encoder.')[1:])
        else:
            k2 = k
        if k2.startswith('wav_encoder'):
            v2_sd[k.replace('wav_encoder','wav_encoder.encoder')] = v
        elif k2.startswith('decoder_projector'):
            v2_sd[k.replace('decoder_projector','visible_encoder.post_net')] = v
        elif k2.startswith('feat_projector'):
            v2_sd[k.replace('feat_projector','wav_encoder.post_net')] = v
        elif k2.startswith('positional_encoder'):
            if is_nested:
                v2_sd['encoder.masker.positional_encoder.scale'] = v
                v2_sd['encoder.decoder.positional_encoder.scale'] = v
            else:
                v2_sd['masker.positional_encoder.scale'] = v
                v2_sd['decoder.positional_encoder.scale'] = v
        else:
            v2_sd[k] = v
    return v2_sd