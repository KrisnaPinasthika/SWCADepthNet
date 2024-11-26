# SWCA DepthNet

Official implementation of [SWCA DepthNet: Leveraging Filtered Local and Global Features for Efficient Monocular Depth Estimation and Enhanced Depth Map Reconstruction]()

For further information, please refer to previous research: [AdaBins](https://github.com/shariqfarooq123/AdaBins) and [BTS](https://github.com/cleinc/bts/tree/master/pytorch)

## Download links

-   You can download all of the pretrained models from [here](https://drive.google.com/drive/u/3/folders/1GjKzqeaOwnJ5pTQ-6i23HJpla8WOFvxl)

## Note

This folder contains the PyTorch implementation of SWCA DepthNet.
We examined this code under Python 3.10.4, Pytorch 1.13.1, CUDA 11.6 on Ubuntu 20.04

## Training preparation

## NYU Depth V2

Download the dataset we used in this work.
In this work, we used the dataset provided by the previous research [BTS](https://github.com/cleinc/bts/tree/master/pytorch).

You can download it from following link: https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing Please make sure to locate the downloaded file to "~/workspace/SWCADepthNet/dataset/nyu_depth_v2/sync.zip".

### Prepare NYU Depth V2 Test set

These codes are based on previous research [BTS](https://github.com/cleinc/bts/tree/master/pytorch). <br>
Please refer to their GitHub repository for more details.

```
$ cd ~/workspace/SWCADepthNet/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/

```

## KITTI

You can also train SWCADepthNet with KITTI dataset by following procedures.
First, make sure that you have prepared the ground truth depthmaps from [KITTI](https://www.cvlibs.net/download.php?file=data_depth_annotated.zip).

Furthermore, please locate the dataset in the following folder "~/workspace/SWCADepthNet/dataset/kitti_dataset".

## Training & Testing

Once the dataset is ready, you can train or evaluate the network using following command.
Ensure you're using the <b>correct</b> pre-trained model from ~/workspace/SWCADepthNet/checkpoints during testing.

-   Train on NYU Depth V2 dataset

```
$ python train.py args_train_nyu.txt
```

-   Evaluate on NYU Depth V2 dataset

```
$ python evaluate.py args_test_nyu.txt
```

---

-   Train on KITTI dataset

```
$ python train.py args_train_kitti_eigen.txt
```

-   Evaluate on KITTI dataset

```
$ python evaluate.py args_test_kitti_eigen.txt
```

---
