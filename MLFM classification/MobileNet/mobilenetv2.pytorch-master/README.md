# PyTorch Implemention of MobileNet V2


Reproduction of MobileNet V2 architecture as described in [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) by Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov and Liang-Chieh Chen on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.


# Requirements
## Dependencies
* PyTorch 1.0+
* [NVIDIA-DALI](https://github.com/NVIDIA/DALI) (in development, not recommended)

# Usage
## Training
Configuration to reproduce our strong results efficiently, consuming around 2 days on 4x TiTan XP GPUs with [non-distributed DataParallel](https://pytorch.org/docs/master/nn.html#torch.nn.DataParallel) and [PyTorch dataloader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader).
* *batch size* 256
* *epoch* 150
* *learning rate* 0.05
* *LR decay strategy* cosine
* *weight decay* 0.00004

The [newly released model](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/pretrained/mobilenetv2-c5e733a8.pth) achieves even higher accuracy, with larger bacth size (1024) on 8 GPUs, higher initial learning rate (0.4) and longer training epochs (250). In addition, a dropout layer with the dropout rate of 0.2 is inserted before the final FC layer, no weight decay is imposed on biases and BN layers and the learning rate ramps up from 0.1 to 0.4 in the first five training epochs.

```shell
python imagenet.py \
    -a mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --epochs 150 \
    --lr-decay cos \
    --lr 0.05 \
    --wd 4e-5 \
    -c <path-to-save-checkpoints> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -j <num-workers>
```

## Test
```shell
python imagenet.py \
    -a mobilenetv2 \
    -d <path-to-ILSVRC2012-data> \
    --weight <pretrained-pth-file> \
    --width-mult <width-multiplier> \
    --input-size <input-resolution> \
    -e
```

# Citations
The following is a [BibTeX](www.bibtex.org) entry for the MobileNet V2 paper that you should cite if you use this model.
```
@InProceedings{Sandler_2018_CVPR,
author = {Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
title = {MobileNetV2: Inverted Residuals and Linear Bottlenecks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
If you find this implementation helpful in your research, please also consider citing:
```
@InProceedings{Li_2019_ICCV,
author = {Li, Duo and Zhou, Aojun and Yao, Anbang},
title = {HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2019}
}
```
## Acknowledgments
The foundational code for our project is derived from the repository "mobilenetv2.pytorch" by Duo Li. We appreciate their contribution to the open-source community. Our adaptation involves the integration of our own LFUM unit, resulting in the creation of the MLFM_MobileNet network.

To learn more about the original "mobilenetv2.pytorch" repository by hysts(https://github.com/d-li14), please visit here(https://github.com/d-li14/mobilenetv2.pytorch).