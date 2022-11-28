# Object-centric Open-Vocabulary Object Detection with Barlow Twins
<!-- Official repository of project "[Open-Vocabulary Object Detection with Barlow Twins]". -->

<hr />

![main figure](docs/overall_architecture.pdf)
> **<p align="justify"> Abstract:** *Open-vocabulary object detection, which focuses on figuring out how to find novel categories given by text queries from the user, is getting more and more attention from the community. Existing open-vocabulary object detectors often increase the size of their vocabulary by utilizing various weak supervision techniques. Based on object-centric OVD, which was the first work to identify the misalignment between image-centric and object-centric in the previous approach and solve this limitation using region-based knowledge distillation (RKD), we investigate the bottleneck of RKD and propose using contrastive loss to achieve better representation alignment. To this end, our proposed method with contrastive loss consistency outperforms Object-Centric OVD across all settings for novel classes by a large margin while slightly increasing the performance of base classes.* </p>

<hr />

## Installation
The code is tested with PyTorch 1.10.0 and CUDA 11.3. After cloning the repository, follow the below steps in [INSTALL.md](docs/INSTALL.md).
All of our models are trained using 4 NVIDIA GTX3090 GPUs. 
<hr />

## Results
We present performance of our approach that demonstrates state-of-the-art results on Open Vocabulary COCO benchmark dataset.


### Open-vocabulary COCO 

| Name                                                                                        | APnovel | APbase |  AP  | Train-time | Download                                                                                                                            |
|:--------------------------------------------------------------------------------------------|:-------:|:------:|:----:|:----------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| [Base_OVD_RCNN_C4_repro](configs/coco/Base-OVD-RCNN-C4.yaml)                                      |   0.7 | 53.4 | 39.6 |     8h     |[model](https://drive.google.com/file/d/1yol1rZRCCCDGlRbX5ydDWtiqTKsIY4uR/view?usp=share_link) |
| [COCO_OVD_Base_RKD_repro](configs/coco/COCO_OVD_Base_RKD.yaml)                                    |  21.3 | 54.5 | 45.8 |     8h     |[model](https://drive.google.com/file/d/1VQpI4BAfjb9vGZO2K9XuvCcXFyxRZ8Mf/view?usp=share_link) |
| [COCO_OVD_Base_PIS-repro](configs/coco/COCO_OVD_Base_PIS.yaml)                                    |  33.2 | 50.1 | 45.7 |    8.5h    |[model](https://drive.google.com/file/d/1AkW0Y14VWiJY_JxANIiG2Av1SAOCjVwp/view?usp=share_link) |
| [COCO_OVD_RKD_PIS_repro](configs/coco/COCO_OVD_RKD_PIS.yaml)                                      |  34.7 | 49.7 | 45.8 |    8.5h    |[model](https://drive.google.com/file/d/18bRrmqVs1c5-s5yk1NxYTIL0naNDJWHT/view?usp=share_link) |
| [COCO_OVD_RKD_PIS_WeightTransfer_repro](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer.yaml)        |  39.0 | 49.9 | 47.0 |    8.5h    |[model](https://drive.google.com/file/d/11dy8F8JujIZyk80IIf3qpR5iUT4dlqLP/view?usp=share_link) |
| [COCO_OVD_RKD_PIS_WeightTransfer_8x_repro](configs/coco/COCO_OVD_RKD_PIS_WeightTransfer_8x.yaml)  |  40.1 | 53.9 | 50.3 |  2.5 days  |[model](https://drive.google.com/file/d/19USlM-w5u956W7jVtGIqWa_OXA1CE8Ko/view?usp=share_link) |

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
Our code is based on [Detic](https://github.com/facebookresearch/Detic) and [Object-centric OVD](https://github.com/hanoonaR/object-centric-ovd) repositories. We thank them for releasing their code.
