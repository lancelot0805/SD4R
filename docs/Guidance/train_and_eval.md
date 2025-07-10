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
bash test_VoD.sh 
```