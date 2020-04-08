# Flow Attack

This is an official repository of

*Anurag Ranjan, Joel Janai, Andreas Geiger, Michael J. Black*. **Attacking Optical Flow.** ICCV 2019.

[[Project Page](http://flowattack.is.tue.mpg.de/)] [[Arxiv](arxiv.org)]

### Known Issues
- To obtain the batch, use the learning rate of `1e3` and `1e4`. For each learning rate, run at least five different trials for 30 epochs.
- The best patch for FlowNetC was obtained with LR of `1e3` and for FlowNet2 was obtained with LR of `1e4`. 

## Prerequisites
Python3 and pytorch are required. Third party libraries can be installed (in a `python3 ` virtualenv) using:

```bash
pip3 install -r requirements.txt
```
Install custom cuda layers for FlowNet2 using

```bash
bash install_flownet2_deps.sh
```
### Preparing training data

Download the [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php) dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command.

```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti' --dump-root /path/to/resulting/formatted/data/ --width 1280 --height 384 --num-threads 1 --with-gt
```

For testing optical flow ground truths on KITTI, download [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) dataset.

### Pretrained Models
Download the pretrained models for [FlowNetC](https://drive.google.com/file/d/1BFT6b7KgKJC8rA59RmOVAXRM_S7aSfKE/view), [FlowNet2](https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view), [PWC-Net](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/pwc_net_chairs.pth.tar). The pretrained models for SPyNet and Back2Future are provided with this repository.

## Generating Adversarial Patches

### White-Box Attacks
In the White-Box attacks we optimize a patch for a single network. We use gradient descent as described in the paper.
Use the following command to generate an adversarial patch for a specific network architecture using the prepared dataset:

```bash
python3 main.py --data [Path to prepared dataset] --kitti-data [Path to KITTI 2015 test set] --flownet [FlowNetS|FlowNetC|FlowNet2|PWCNet|Back2Future|SpyNet] --patch-size 0.10 --name [Name of the experiment]
```

The patch size is specified in percentage of the training image size (default: 256).
All other arguments such as the learning rate, epoch size, etc are set to the values used in our experiments. For details please check main.py

## Acknowledgements
- We thank several github users for their contributions which are used in this repository.
  - The code for generating randomized patches and augmentation comes from [jhayes14/adversarial-patch](https://github.com/jhayes14/adversarial-patch).
  - Data preprocessing and KITTI dataloaders code is taken from [ClementPinard/SfmLearner-Pytorch/](https://github.com/ClementPinard/SfmLearner-Pytorch/). Optical flow evalution code is taken from [anuragranj/cc](https://github.com/anuragranj/cc).
  - FlowNet and FlowNet2 models have been taken from [NVIDIA/flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch). PWCNet is taken from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net).
  - SPyNet implementation is taken from [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet).
  - Back2Future implementation is taken from [anuragranj/back2future.pytorch](https://github.com/anuragranj/back2future.pytorch).
  - Correlation module is taken from [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
