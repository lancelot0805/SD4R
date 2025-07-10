## Training Details

We train our SD4R for totally 24 epochs on 4 NVIDIA 4090 GPUs, with a batch size of 4. Specifically, we divide the training of SD4R into two stages: pretraining and training. (1) Pretraining Stage: In this stage, the radar and image branches are initialized with the pretrained RadarPillarNet and MVX-Faster-RCNN, respectively. The goal is to train the model's ability to estimate depth in the image branch and to fuse multimodal information in the BEV perspective effectively. (2) Training Stage: Using the pretrained checkpoint obtained from the pretraining stage, the model is further initialized and trained for 3D object detection tasks. 
## Train

```
tmux new -s your_tmux_name
conda activate SD4R
bash ./tools_det3d/dist_train.sh config_path 4
# modified detailed settings in dist_train.sh
```

The training logs and checkpoints will be saved under the log_folder、

## Evaluation

Downloading the checkpoints from the model zoo and putting them under the projects/KD4R/checkpoints.
```
bash test_VoD.sh # for evaluating the FINAL_VoD.pth on View-of-delft (VoD) dataset.
```