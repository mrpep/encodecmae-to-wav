<p align="center">
  <img src="https://github.com/mrpep/encodecmae-to-wav/blob/main/back2waveform.png" title="Logo saying 'Back to the waveform'">
</p>
<p align="center">
  Code with implementation of models aiming to invert <a href="https://github.com/habla-liaa/encodecmae">EnCodecMAE</a> features back to the waveform domain.
</p>



### Inference
We provide pretrained weights for many of our models, and [this colab](https://colab.research.google.com/drive/1vxAvLuzSe2QJkcSTzck96GBM35wIka_a?usp=sharing) demonstrates how to play around with them.

| Model Name  | Upstream | Summary | Training Data | Model Type |
| ----------- | -------- | ------- | ------------- | ---------- |
| ecmae2ec-base-1LTransformer | EnCodecMAE Base | None | AS + LL + FMA | Regressor |
| DiffTransformerAE2L8L1CLS-10s | EnCodecMAE Base | 10s | FMA + Jamendo | Diffusion |
| DiffTransformerAE2L8L1CLS-4s | EnCodecMAE Base | 4s | FMA | Diffusion |

### Training
For training follow these steps:
1) Gather training datasets and put them in a folder. The datasets should have a sampling rate of 24 kHz.
2) Install docker and docker-compose.
3) Clone this repository and also [this one](https://github.com/habla-liaa/encodecmae)
4) Edit the [docker-compose file](https://github.com/mrpep/encodecmae-to-wav/blob/main/encodecmae-to-wav/docker-compose.yml). Modify the paths in volumes so that they point to: encodecmae repository, this repository, and the folder with the datasets. These folders will appear in the docker container inside the /workspace folder. Update the device_ids according to the gpus that you want to use inside the container for training.
5) Update the paths in the configs/datasets as needed
6) Inside this repository folder run:
  ```
docker compose up -d
docker attach encodecmae-to-wav-train
  ```
7) An interactive shell will open. Run
```
cd /workspace/encodecmae
pip install -e .
cd /workspace/encodecmae-to-wav
pip install -e .
```
8) Check that the datasets appear in /workspace/datasets
9) Navigate to /workspace/encodecmae-to-wav/encodecmae-to-wav
10) Run
   ```chmod +x scripts/train.sh```
11) In scripts/train.sh you will find a list of commands, each corresponding to a different experiment. Comment everything except the experiment to be ran. The batch size and other parameters can be modified in the --mods argument or by editing [this config](https://github.com/mrpep/encodecmae-to-wav/blob/main/encodecmae-to-wav/configs/base/decode_encodecmae.gin)
12) Run scripts/train.sh and it should start training.

### Citation

If you use this code or results in your paper, please cite our work as:
```
@article{alonso2024leveraging,
  title={Leveraging pre-trained autoencoders for interpretable prototype learning of music audio},
  author={Alonso Jim{\'e}nez, Pablo and Pepino, Leonardo and Batlle-Roca, Roser and Zinemanas, Pablo and Serra, Xavier and Rocamora, Mart{\'\i}n},
  year={2024},
  publisher={Institute of Electrical and Electronics Engineers (IEEE)}
}
```
```
@article{pepino2023encodecmae,
  title={EnCodecMAE: Leveraging neural codecs for universal audio representation learning},
  author={Pepino, Leonardo and Riera, Pablo and Ferrer, Luciana},
  journal={arXiv preprint arXiv:2309.07391},
  year={2023}
}
```
