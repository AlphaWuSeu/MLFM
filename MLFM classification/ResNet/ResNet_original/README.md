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

Acknowledgments
The foundational code for our project is derived from the repository "pytorch_image_classification" by hysts. We appreciate their contribution to the open-source community. Our adaptation involves the integration of our own LFUM unit, resulting in the creation of the MLFM_ResNet network.

To learn more about the original "pytorch_image_classification" repository by hysts, please visit here.