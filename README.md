# UCMCTrack

[![arXiv](https://img.shields.io/badge/arXiv-2312.08952-<COLOR>.svg)](https://arxiv.org/abs/2312.08952)

[AAAI 2024] UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation

*The source code will be released soon.*


## Estimated Camera Parameters

This directory provides camera parameters we have estimated:

└─cam_para
    ├─DanceTrack
    ├─MOT17
    └─MOT20

We have provided the camera parameter files estimated on the datasets MOT17, MOT20 and DanceTrack. The specific format of the camera parameter file consists of following three parts. Among them, $IntrinsicMatrix$ represents the intrinsic parameters Ki of the camera, the first and second columns represent the focal lengths of the camera in the x and y directions, and the third column is the offset when the origin of the physical imaging plane moves to the pixel plane. $RotationMatrices$ and $TranslationVectors$ represent key components of the camera’s extrinsic parameters Ko. Among them, $RotationMatrices$ represents the rotation of the camera relative to the ground plane, while $TranslationVectors$ represents the offset of the camera relative to the ground plane, in millimeters.

#### Sample

$RotationMatrices$
0.00000 -1.00000 0.00000
-0.05234 0.00000 -0.99863
0.99863 0.00000 -0.05234

$TranslationVectors$
0 1391 3968 

$IntrinsicMatrix$
1213 0 960 
0 1213 540 
0 0 1 



## Benchmark Performance



## Get Started



## Demo



## Acknowledgement and Citation

```bibtex
@inproceedings{yi2024ucmc,
  title={UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation},
  author={Kefu Yi, Kai Luo, Xiaolei Luo, Jiangui Huang, Hao Wu, Rongdong Hu, Wei Hao},
  booktitle={AAAI},
  year={2024}
}
```

