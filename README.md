Code with implementation of models aiming to invert [EnCodecMAE](https://github.com/habla-liaa/encodecmae) features back to the waveform domain.

### Inference
We provide pretrained weights for many of our models, and [this colab] so that you can play around with them.

### Training
For training follow these steps:
1) Gather training datasets and put them in a folder.
2) Install docker and docker-compose.
3) Clone this repository and also [this one](https://github.com/habla-liaa/encodecmae)
4) Edit the [docker-compose file](https://github.com/mrpep/encodecmae-to-wav/blob/main/encodecmae-to-wav/docker-compose.yml). Modify the paths in volumes so that they point to: encodecmae repository, this repository, and the folder with the datasets. These folders will appear in the docker container inside the /workspace folder. Update the device_ids according to the gpus that you want to use inside the container for training.
5) Update the paths in the configs/datasets as needed
6) Run
  ```docker compose up -d
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

If you found the diffusion models useful please cite:
PAPER PECMAE
EnCodecMAE
