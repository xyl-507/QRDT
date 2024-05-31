# Handling Occlusion in UAV Visual Tracking with Query-Guided Re-Detection

This is an official pytorch implementation of the 2024 IEEE Transactions on Instrumentation and Measurement paper: 
```
Handling Occlusion in UAV Visual Tracking with Query-Guided Re-Detection
(accepted by IEEE Transactions on Instrumentation and Measurement)
```
The paper can be downloaded from [IEEE Xplore]()

The models and raw results can be downloaded from [BaiduYun](https://pan.baidu.com/s/10G2rx4--6vWgGCHjKhWpHw?pwd=1234). 

The tracking demos are displayed on the [Bilibili](https://www.bilibili.com/video/BV1kN411n78y/) or [GitHub](https://github.com/xyl-507/QRDT/releases/tag/demo)

The real-world tests are displayed on the [GitHub](https://github.com/xyl-507/QRDT/releases/tag/demos)

### Proposed modules
- `query update (QU)` in [Tracker](https://github.com/xyl-507/QRDT/blob/master/siamban/tracker/siambanlt_tracker_template_KF.py)

- `Cross Fusion Layer (CFL)` in [model](https://github.com/xyl-507/QRDT/blob/master/siamban/models/cam.py)
  
- `Trajectory prediction (TP)` in [Tracker](https://github.com/xyl-507/QRDT/blob/master/siamban/tracker/siambanlt_tracker_template_KF.py)

### UAV Tracking

| Datasets | qrdt_r50_l234| 
| :--------------------: | :----------------: | 
| UAV20L - Full Occlusion (Suc./Pre.) | 0.396/0.613| 
| UAVDT - Large Occlusion (Suc./Pre.) | 0.474/0.604 |
| DTB70 - Occlusion (Suc./Pre.) | 0.553/0.768 |
| VisDrone2019-SOT-test-dev - Full Occlusion (Suc./Pre.) |0.591/0.812 |
| HOB (Suc./Pre.) | 0.363/0.244 |


Note:

-  `r50_lxyz` denotes the outputs of stage x, y, and z in [ResNet-50](https://arxiv.org/abs/1512.03385).

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using QRDT

### Add SmallTrack to your PYTHONPATH

```bash
export PYTHONPATH=/path/to/qrdt:$PYTHONPATH
```


### demo

```bash
python tools/demo.py \
    --config experiments/siamban_r50_l234/config.yaml \
    --snapshot experiments/siamban_r50_l234/QRDT.pth
    --video demo/bag.avi
```

### Download testing datasets

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [Google Drive](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI) or [BaiduYun](https://pan.baidu.com/s/1js0Qhykqqur7_lNRtle1tA#list/path=%2F). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`. 

### Test tracker

```bash
cd experiments/siamban_r50_l234
python -u ../../tools/test.py 	\
	--snapshot QRDT.pth 	\ # model path
	--dataset UAV20L 	\ # dataset name
	--config config.yaml	  # config file
```

The testing results will in the current directory(results/dataset/model_name/)

### Eval tracker

assume still in experiments/ssiamban_r50_l234

``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset UAV20L        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'ch*'   # tracker_name
```

###  Training :wrench:

See [TRAIN.md](TRAIN.md) for detailed instruction.


### Acknowledgement
The code based on the [PySOT](https://github.com/STVIR/pysot) , [SiamBAN](https://github.com/hqucv/siamban) ,
[CAM](https://dl.acm.org/doi/10.5555/3454287.3454647) , [CFME](https://ieeexplore.ieee.org/document/8880656) and [DROL](https://aaai.org/papers/13017-discriminative-and-robust-online-learning-for-siamese-visual-tracking/)

We would like to express our sincere thanks to the contributors.
