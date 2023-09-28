## This is the PyTorch implement of SENet (train on ImageNet dataset)

Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)
# Usage

### Train

* If you want to train from scratch, you can run as follows:

```
python train.py --network se_resnext_50 --batch-size 256 --gpus 0,1,2,3
```

parameter `--network` can be `se_resnet_18` or `se_resnet_34` or `se_resnet_50` or `se_resnet_101` or `se_resnet_152` or `se_resnext_50` or `se_resnext_101` or `se_resnext_152`.

* If you want to train from one checkpoint, you can run as follows(for example train from `epoch_4.pth.tar`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
python train.py --network se_resnext_50 --batch-size 256 --gpus 0,1,2,3 --resume output/epoch_4.pth.tar --start-epoch 4
```

## Acknowledgments
The foundational code for this project is derived from the repository "SENet-PyTorch" by Kaifeng Wei. We appreciate their contribution to the open-source community. Our adaptation involves the integration of our own LFUM unit, resulting in the creation of the MLFM_SENet network.

To learn more about the original "SENet-PyTorch" repository by hysts(https://github.com/miraclewkf), please visit here(https://github.com/miraclewkf/SENet-PyTorch).