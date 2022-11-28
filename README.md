# Object Centric Open Vocabulary Detection (NeurIPS 2022)
Official repository of project "[Open-Vocabulary Object Detection with Barlow Twins]".

<hr />

![main figure](docs/overall_architecture.pdf)
> **<p align="justify"> Abstract:** *Open-vocabulary object detection, which focuses on figuring out how to find novel categories given by text queries from the user, is getting more and more attention from the community. Existing open-vocabulary object detectors often increase the size of their vocabulary by utilizing various weak supervision techniques. Based on object-centric OVD, which was the first work to identify the misalignment between image-centric and object-centric in the previous approach and solve this limitation using region-based knowledge distillation (RKD), we investigate the bottleneck of RKD and propose using contrastive loss to achieve better representation alignment. To this end, our proposed method with contrastive loss consistency outperforms Object-Centric OVD across all settings for novel classes by a large margin while slightly increasing the performance of base classes.* </p>

<hr />

## Installation
The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, follow the below steps in [INSTALL.md](docs/INSTALL.md).
All of our models are trained using 8 A100 GPUs. 
<hr />

## Demo: Create your own custom detector
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Object_Centric_OVD_Demo.ipynb) Checkout our demo using our interactive colab notebook. Create your own custom detector with your own class names. 


## Results
We present performance of Object-centric Open Vocabulary object detector that demonstrates state-of-the-art results on Open Vocabulary COCO benchmark dataset.


### Open-vocabulary COCO 

| Name                                                                                        | APnovel | APbase |  AP  | Train-time | Download                                                                                                                            |
|:--------------------------------------------------------------------------------------------|:-------:|:------:|:----:|:----------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Base-OVD-RCNN-C4](configs/coco/Base-OVD-RCNN-C4.yaml)                                      |   1.7   |  53.2  | 39.6 |     8h     |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_base.pth) |
| [COCO_OVD_Base_RKD](configs/coco/COCO_OVD_Base_RKD.yaml)                                    |  21.2   |  54.7  | 45.9 |     8h     |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd.pth) |
| [COCO_OVD_Base_PIS](configs/coco/COCO_OVD_Base_PIS.yaml)                                    |  30.4   |  52.6  | 46.8 |    8.5h    |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_pis.pth) |
| [COCO_OVD_RKD_PIS](configs/coco/COCO_OVD_RKD_PIS.yaml)                                      |  31.5   |  52.8  | 47.2 |    8.5h    |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd_pis.pth) |
| [COCO_OVD_RKD_PIS_WeightTransfer](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer.yaml)        |  36.6   |  54.0  | 49.4 |    8.5h    |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd_pis_weighttransfer.pth) |
| [COCO_OVD_RKD_PIS_WeightTransfer_8x](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer_8x.yaml)  |  36.9   |  56.6  | 51.5 |  2.5 days  |[model](https://github.com/hanoonaR/object-centric-ovd/releases/download/v1.0/coco_ovd_rkd_pis_weighttransfer_8x.pth) |

### New LVIS Baseline
Our Mask R-CNN based LVIS Baseline ([mask_rcnn_R50FPN_CLIP_sigmoid](configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)) 
achieves 12.2 rare class and 20.9 overall AP and trains in only 4.5 hours on 8 A100 GPUs. 
We believe this could be a good baseline to be considered for the future research work in LVIS OVD setting.

| Name                                                                 | APr  | APc  | APf  |  AP  | Epochs |
|----------------------------------------------------------------------|:----:|:----:|:----:|:----:|:------:|
| [PromptDet Baseline](https://arxiv.org/abs/2203.16513)               | 7.4  | 17.2 | 26.1 | 19.0 |   12   |
| [ViLD-text](https://arxiv.org/abs/2104.13921)                        | 10.1 | 23.9 | 32.5 | 24.9 |  384   |
| [Ours Baseline](configs/lvis/mask_rcnn_R50FPN_CLIP_sigmoid.yaml)     | 12.2 | 19.4 | 26.4 | 20.9 |   12   |
<hr />

## Training and Evaluation

To train or evaluate, first prepare the required [datasets](docs/DATASETS.md).

To train a model, run the below command with the corresponding config file.

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml
```

Note: Some trainings are initialized from Supervised-base or RKD models. Download the corresponding pretrained models
and place them under `$object-centric-ovd/saved_models/`.

To evaluate a pretrained model, run 

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
```
<hr />

## References
Our RKD and PIS methods utilize the MViT model Multiscale Attention ViT with Late fusion (MAVL) proposed in the work [Class-agnostic Object Detection with Multi-modal Transformer (ECCV 2022)](https://github.com/mmaaz60/mvits_for_class_agnostic_od).
Our code is based on [Detic](https://github.com/facebookresearch/Detic) and [Centric] repositories. We thank them for releasing their code.
