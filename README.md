# Multiscale Low-Frequency Memory (MLFM) Network for Enhanced CNNs

Deep learning, particularly Convolutional Neural Networks (CNNs), have driven monumental advancements across various research arenas. However, their inherent shortcomings in handling low-frequency information often pose challenges, especially in tasks like deciphering global structures or managing smoothly transitioning images. While transformers exhibit commendable performance across tasks, their intricate optimization intricacies underscore an ongoing necessity for sophisticated CNN enhancements with constrained resources.

**MLFM** emerges as a solution to these intricacies.

## Introduction

The **Multiscale Low-Frequency Memory (MLFM) Network** is a revolutionary framework crafted with an intent to harness the untapped prowess of CNNs without tampering with their intrinsic complexity. Central to its design is the **Low-Frequency Memory Unit (LFMU)**, a unique component adept at retaining diverse low-frequency information, thus boosting performance in designated computer vision undertakings. One of MLFM's standout features is its impeccable compatibility with a plethora of leading-edge networks, sans the need to modify their foundational structures.

## Key Features
- Efficient preservation of low-frequency details.
- Seamless integration with renowned networks like ResNet, MobileNet, EfficientNet, and ConvNeXt.
- Demonstrated efficacy beyond image classification - adaptable to image-to-image translation endeavors such as semantic segmentation networks like FCN and U-Net.

## Networks Integrated with MLFM

### ResNet
Derived and adapted from "pytorch_image_classification" by hysts.  
[Original Repository](https://github.com/hysts/pytorch_image_classification)

### SeNet
Adapted from "SENet-PyTorch" by Kaifeng Wei.  
[Original Repository](https://github.com/miraclewkf/SENet-PyTorch)

### MobileNetV2
Sourced from "mobilenetv2.pytorch" by Duo Li.  
[Original Repository](https://github.com/d-li14/mobilenetv2.pytorch)

### ConvNeXt & inceptionnext
Both these networks are adapted from the "inceptionnext" repository by Sea AI Lab.  
[Original Repository](https://github.com/sail-sg/inceptionnext)

## Dataset Utilized
**ImageNet100**: A subset of ImageNet-1k Dataset from the ImageNet Large Scale Visual Recognition Challenge 2012. It encapsulates 100 random classes as detailed in the `Labels.json` file.  
[Download Dataset](https://www.kaggle.com/datasets/ambityga/imagenet100)

## Train && test
Please train and validate in the manner provided in the original catalogue of the network.

## Concluding Remarks
This endeavor underscores a monumental leap in optimizing CNNs' potential within resource constraints, building on existing CNN paradigms and setting the stage for imminent breakthroughs in computer vision.

The accuracies of our network compared to the original CNN network on ImageNet100 is shown below.
| Network              | Baseline Accuracy | MLFM Enhanced Accuracy |
|----------------------|-------------------|------------------------|
|ResNet10|77.58%|78.64%|
|ResNet18|77.86%|81.22%|
|ResNet34|79.82%|81.50%|
|ResNet50|80.16%|81.80%|
|MobileNetV2_0.1|58.38%|62.86%|
|MobileNetV2_0.35|76.82%|79.24%|
|MobileNetV2_0.5|79.64%|80.36%|
|MobileNetV2_0.75|81.82%|82.60%|
|MobileNetV2_1.0|82.52%|83.06%|
|RegNetX_200M|77.59%|79.02%|
|RegNetX_400M|78.92%|81.68%|
|RegNetX_600M|81.34%|81.98%|
|RegNetX_800M|82.70%|83.36%|
|RegNetY_200M|77.90%|78.94%|
|RegNetY_400M|79.32%|81.48%|
|RegNetY_600M|80.54%|82.48%|
|RegNetY_800M|82.70%|82.98%|
|EfficientNet_B0|83.84%|83.44%|
|EfficientNet_B1|83.94%|84.64%|
|EfficientNet_B2|84.16%|84.54%|
|EfficientNet_B3|84.28%|85.30%|
|EfficientNet_B4|84.68%|85.96%|
|SeNet10|74.98%|75.40%|
|SeNet18|76.34%|77.86%|
|SeNet34|78.80%|78.98%|
|SeNet50|79.56%|80.36%|
|ConvNeXt_K3_par1_8|88.03%|88.30%|
|ConvNeXt_K3_par1_4|88.02%|88.46%|
|ConvNeXt_K3_par1_2|88.15%|88.33%|
|ConvNeXt_K3|88.11%|88.32%|
|ConvNeXt_K5|88.18%|88.42%|
|ConvNeXt_tiny|88.12%|88.60%|
|ConvNeXt_small|88.08%|88.34%|
|InceptionNeXt|87.58%|88.06%|
