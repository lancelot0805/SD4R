## View-of-delft (VoD)

Download the [View-of-delft](https://github.com/tudelft-iv/view-of-delft-dataset). Then we generate perspective view foreground segmentation results from detectron2 as one of the ground-truths. All our ablation experiments are conducted on the VoD dataset.

### preparation
```bash
ln -s /your_path/view_of_delft_PUBLIC/ ./data/VoD
python projects/SD4R/preprocess/gen_panoptic_seg_vod.py
python projects/SD4R/preprocess/png2npy_vod.py
python tools_det3d/create_data_VODradar.py # creat radar_5frames data as radar data
python tools_det3d/create_data_VODlidar.py # creat lidar data, for lidar detection
```

### Folder structure

The data is organized in the following format:

```
View-of-Delft-Dataset (root)
    ├── lidar (VoD dataset where velodyne contains the LiDAR point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── radar (VoD dataset where velodyne contains the radar point clouds)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    | 
    ├── radar_3_scans (VoD dataset where velodyne contains the accumulated radar point clouds of 3 scans)
    │   │── ImageSets
    │   │── training
    │   │   ├──calib & velodyne & image_2 & label_2
    │   │── testing
    │       ├──calib & velodyne & image_2
    |
    ├── radar_5_scans (VoD dataset where velodyne contains the radar point clouds of 5 scans)
        │── ImageSets
        │── training
        │   ├──calib & velodyne & image_2 & label_2
        │── testing
            ├──calib & velodyne & image_2
          
```

