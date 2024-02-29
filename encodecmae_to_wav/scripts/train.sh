#TransformerAE diffusion 1xCLS 2L 8L
#ginpipe configs/base/decode_encodecmae.gin \
#	configs/models/encodecmae-ae-diffusion.gin \
#	configs/datasets/fma-large-24k.gin \
#	configs/features/wav_only.gin \
#	--module_list configs/imports \
#	--project_name encodecmae_cls \
#	--experiment_name transformer_ae_2L8Lclsx1_diffusion_encodec_guidance01 \
#	--mods "NUM_CLS_TOKENS=1" "NUM_DENOISER_LAYERS=8" "NUM_ENCODER_LAYERS=2" "DEVICE=[0]"

#TransformerAE diffusion 1xCLS 2L 8L 10s FMA+Jamendo
# ginpipe configs/base/decode_encodecmae.gin \
# 	configs/models/encodecmae-ae-diffusion.gin \
# 	configs/datasets/fma-large-24k.gin \
# 	configs/datasets/jamendo-24k.gin \
# 	configs/features/wav_only.gin \
# 	--module_list configs/imports \
# 	--project_name encodecmae_cls \
# 	--experiment_name transformer_ae_2L8Lclsx1_diffusion_encodec_guidance01_jamendofma_10s \
# 	--mods "NUM_CLS_TOKENS=1" "NUM_DENOISER_LAYERS=8" "DEVICE=[0]" "NUM_ENCODER_LAYERS=2" "MAX_AUDIO_DURATION=10" "TRAIN_BATCH_SIZE=32"

#Base model to waveform
ginpipe configs/base/decode_encodecmae.gin \
 		configs/models/encodecmae-to-ec.gin \
 		configs/datasets/audioset-unbalanced-24k.gin \
 		configs/datasets/fma-large-24k.gin \
 		configs/datasets/librilight-6k-24k.gin \
 		configs/features/wav_only.gin \
 		--module_list configs/imports \
 		--project_name ecmae2ec \
 		--experiment_name ecmae_base_transformer1L