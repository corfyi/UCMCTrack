# UCMCTrack

[![arXiv](https://img.shields.io/badge/arXiv-2312.08952-<COLOR>.svg)](https://arxiv.org/abs/2312.08952) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=ucmctrack-multi-object-tracking-with-uniform)

[AAAI 2024] UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation


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

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=ucmctrack-multi-object-tracking-with-uniform)

| Dataset         | HOTA | AssA | IDF1 | MOTA | FP     | FN     | IDs   | Frag  |
| --------------- | ---- | ---- | ---- | ---- | ------ | ------ | ----- | ----- |
| MOT17 (private) | 65.5 | 66.4 | 80.9 | 80.1 | 34,584 | 75,846 | 1,647 | 2,298 |

*This result was obtained using the Python version of the code. In the paper, the C++ version of the code was used, and the Python version had slightly lower performance metrics than the C++ version (due to some subtle differences in implementation). We will address this issue in the future.*

## Get Started

```bash
pip install -r requirements.txt
```



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

