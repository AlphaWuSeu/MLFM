# PyTorch Image Classification

Following papers are implemented using PyTorch.

* ResNet ([1512.03385](https://arxiv.org/abs/1512.03385))

## Requirements

* Ubuntu 
* Python >= 3.7
* PyTorch >= 1.4.0
* torchvision
* [NVIDIA Apex](https://github.com/NVIDIA/apex)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py --config config/resnet10.yaml
```