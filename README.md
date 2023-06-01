# Deformable Convolutions for Semantic Segmentation

This project focuses on implementing deformable convolutions for the task of semantic segmentation, using the ADE20k dataset as an example. It is carried out as part of the PA228 course, "Machine Learning in Image Processing."

## Objectives

The main objectives of this project are as follows:

-   Study state-of-the-art approaches related to semantic segmentation and cite at least three ML IP papers on this topic. Each method will be briefly introduced, highlighting its basic ideas.
-   Implement one of the selected methods. Any improvements made to the existing implementation will be documented, and all resources and collaborations will be properly acknowledged.
-   Train a model using the implemented method and evaluate its performance. The results will be compared to those of another non-trivial model to demonstrate the effectiveness of the implemented approach. The training process will be thoroughly validated, and any improvements to standard techniques will be validated for their benefits.
-   Write a report documenting the experiences and findings throughout the project. Difficulties encountered will be listed, and suggestions for future work and improvements will be provided.

## Implementation

The project utilizes the PyTorch Lightning and Albumentations libraries for implementation. 
The pretrained DilatedResNet50 model serves as the backbone, and a Deformable Convolutions layer with a Modulation mechanism has been implemented.

## Results

Presentation of the results can be found in the file [Deformable Convolutions.pdf](Deformable%20Convolutions.pdf)

All metrics, logs, and configurations are available on the public WANDB project: [https://wandb.ai/tesskyrim000/pa228-project](https://wandb.ai/tesskyrim000/pa228-project)

Example datasets and models are provided in the repository.

## References

The following resources have been referenced in this project:

-   Paper: [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
-   Paper: [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168)
-   Video: [ICCV17 | 133 | Deformable Convolutional Networks](https://www.youtube.com/watch?v=HRLMSrxw2To)
-   Paper: [Semantic Understanding of Scenes through the ADE20K Dataset](https://arxiv.org/abs/1608.05442)
-   Blog: [Behind the Scenes: Setting a Baseline for Image Segmentation Speedups](https://www.mosaicml.com/blog/behind-the-scenes)
-   Paper: [Accurate, large minibatch sgd: Training ImageNet in 1 hour](https://arxiv.org/abs/1706.02677)
