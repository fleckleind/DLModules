# HRNet

[Deep High-Resolution Representation Learning for Human Pose Estimation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)  

High-Resoluiton Net (HRNet): consists of parallel high-to-low resolution layers with repeated information exchange across multi-resolution layers, with horizontal and vertical directions corresponding to the depth of the network and the scale of the feature maps.  

## Parallel Multi-Resolution Subnetworks
$N_{sr}$ is the subnetwork in the $s-th$ stage and $r$ resolution index, with resolution as $1/(2^r-1)$ of the first-subnetwork resolution.  

The resolution of first stage in HRNet is $4\times$ downsampled compared with the original input images. The resolutions for the parallel subnetworks of later stages consists of the resolutions from the previous stage and an extra lower one.  

## Multi-Scale Fusion
HRNet introduces exchange units across parallel subnetworks to receive the information from other parallel subnetworks. To unify the different resolution of previous parallel feature maps, $3\times3$ convolution with stride as 2 is used to downsample, and $1\times1$ bilinear interpolation followed by a $1\times1$ convolution is used to upsample. The output representation is the sum of transformed input feature maps.

## Representation Head
The representation heads of HRNet includes 3 categories:
1. only output the representation from the highest-resolution convolution stream.
2. concatenate the feature maps from all stream with resolution unified with the highest one via bilinear interpolation, and use $1\times1$ convolution to adjust channel size.
3. construct multi-level representations (feature pyramid) via downsampling the concatenated high-resolution representation.



## Reference
[official implementation](https://github.com/HRNet)  
[Deep High-Resolution Representation Learning
 for Visual Recognition](https://arxiv.org/pdf/1908.07919)
