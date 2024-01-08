# UCMCTrack

> **[AAAI 2024] UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation**.
> UCMCTrack is a simple pure motion based tracker that achieves state-of-the-art performance on multiple datasets. In particular, **it achieves the first place on MOT17 without using any appearance cues**, making it highly applicable for real-time object tracking on end devices.

[![arXiv](https://img.shields.io/badge/arXiv-2312.08952-<COLOR>.svg)](https://arxiv.org/abs/2312.08952) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=ucmctrack-multi-object-tracking-with-uniform) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=ucmctrack-multi-object-tracking-with-uniform)


## 🚗 Tracking Vehicles with Moving Camera
![](demo/demo.gif)


## 📷 Estimating Camera Parameter from a Single Image
![](docs/cam_para.gif)

## 📰 News
* [12/29/2023]  **Open-Sourcing a Tool for Estimating Camera Parameters from a Single Image！** For specific steps, refer to Get Started.
* [01/02/2024]  **Usage Guide Now Available for the Camera Parameter Estimation Tool!** 
* [01/02/2024]  **Add head padding (HP) post-processing trick as OC-SORT.** Now the performance gap between the Python version of the code and the C++ version in the paper has been eliminated.

## 📈 Star Rising
[![Star History Chart](https://api.star-history.com/svg?repos=corfyi/UCMCTrack&type=Timeline)](https://star-history.com/#corfyi/UCMCTrack&Timeline)

## ⭐ Stargazers
[![Stargazers repo roster for @corfyi/UCMCTrack](http://reporoster.com/stars/corfyi/UCMCTrack)](https://github.com/corfyi/UCMCTrack/stargazers) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fcorfyi%2FUCMCTrack&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## 🚩 Demo
This demo demonstrates the use of YOLOv8x as the detector and UCMCTrack as the tracker for real-time vehicle detection and tracking from a video file. The demo processes the video file `demo.mp4` to detect and track vehicles, saving the tracking results in the `output` folder. **In the case of significant camera shake**, UCMCTrack still has good performance without using any appearance information.

#### Environment
Before you begin, ensure you have the following prerequisites installed on your system:
- Python (3.8 or later)
- PyTorch with CUDA support
- Ultralytics Library
- Download weight file [yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) to folder `pretrained`

#### Run the demo

```bash
python demo.py --cam_para demo/cam_para.txt --video demo/demo.mp4
```
The file `demo/cam_para.txt` is the camera parameters estimated from a single image. The code of this tool is released.  For specific steps, please refer to the Get Started.

## 🗼 Pipeline of UCMCTrack
First, the detection boxes are mapped onto the ground plane using homography transformation. Subsequently, the Correlated Measurement Distribution (CMD) of the target is computed. This distribution is then fed into a Kalman filter equipped with the Constant Velocity (CV) motion model and Process Noise Compensation (PNC). Next, the mapped measurement and the predicted track state are utilized as inputs to compute the Mapped Mahalanobis Distance (MMD). Finally, the Hungarian algorithm is applied to associate the mapped measurements with tracklets, thereby obtaining complete tracklets.

![](docs/pipeline.png)

## 🖼️ Visualization of Different Distances
(a) Visualization of IoU on the image plane. IoU fails as there is no intersection between bounding boxes. (b) Visualization of Mapped Mahalanobis Distance (MMD) without Correlated Measurement Distribution (CMD). Incorrect associations occur due to insufficient utilization of distribution information. (c) Visualization of MMD with CMD. Correct associations after using the correlated probability distribution, undergoing a rotation on the ground plane.

![](docs/distance_measure.png)

## 🏃 Benchmark Performance
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=ucmctrack-multi-object-tracking-with-uniform) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=ucmctrack-multi-object-tracking-with-uniform)

| Dataset    | HOTA | AssA | IDF1 | MOTA | FP     | FN     | IDs   | Frag  |
| ---------- | ---- | ---- | ---- | ---- | ------ | ------ | ----- | ----- |
| MOT17 test | 65.8 | 66.4 | 81.0 | 80.6 | 36,213 | 71,454 | 1,689 | 2,220 |
| MOT20 test | 62.8 | 63.5 | 77.4 | 75.6 | 28,678 | 96,199 | 1,335 | 1,370 |


## 💁 Get Started
- Install the required dependency packages 

```bash
pip install -r requirements.txt
```

- Run UCMCTrack on the MOT17 test dataset, and the tracking results are saved in the folder `output/mot17/test`

```bash
. run_mot17_test.bat
```

- Run UCMCTrack on the MOT17 validation dataset and evaluate performance metrics such as IDF1, HOTA, and MOTA locally

```bash
. run_mot17_val.bat
```

- Run UCMCTrack on the MOT20 test dataset, and the tracking results are saved in the folder `output/mot20/test`

```bash
. run_mot20_test.bat
```

- Estimating camera parameters from a single image

```bash
python util/estimate_cam_para.py --img demo/demo.jpg --cam_para demo/cam_para_test.txt
```

press 'q' on the Image UI window to quit and save camera parameters.




## 🔧 Camera Parameter Estimation Tool

This tool is designed to estimate unknown camera parameters from a single image. It's particularly useful for extracting camera parameters from images extracted from video sequences, where the exact camera settings might be unknown.

### Usage
To estimate camera parameters from an image, use the following command:

```
python util/estimate_cam_para.py --img demo/demo.jpg --cam_para demo/cam_para_test.txt
```

Here, `demo/demo.jpg` is an image extracted from a video sequence, and `demo/cam_para_test.txt` is the file where you will input and receive the camera parameters.

### Step 1: Modifying the Camera Parameters File

The first step involves adjusting the initial camera parameters in `demo/cam_para_test.txt`. Follow these steps:

1. **Set the Center Point**: Based on the resolution of your image, set the center point in the `IntrinsicMatrix` section of the `cam_para_test.txt` file. This should be half of the image's width and height.
2. **Set the Focal Length**: Assign a default value of 1000 to the focal length in the `IntrinsicMatrix`.

### Step 2: Running the Tool

After modifying the `cam_para_test.txt` file, run the command provided above. The tool will process the image and estimate the camera parameters based on the initial values you've set.

By adjusting parameters like `theta_x`, `theta_y`, `theta_z`, `focal`, and `t_x`, you can align a virtual ground plane grid with the ground in the image. This process will provide you with an estimation of the camera parameters. While this estimation might not be extremely precise, it is sufficiently accurate for use with UCMCTrack.



## 📷 Estimated Camera Parameters

This directory provides camera parameters we have estimated:
```
├─cam_para
    ├─DanceTrack
    ├─MOT17
    └─MOT20
```
We have provided the camera parameter files estimated on the datasets MOT17, MOT20 and DanceTrack. The specific format of the camera parameter file consists of following three parts. Among them, $IntrinsicMatrix$ represents the intrinsic parameters Ki of the camera, the first and second columns represent the focal lengths of the camera in the x and y directions, and the third column is the offset when the origin of the physical imaging plane moves to the pixel plane. $RotationMatrices$ and $TranslationVectors$ represent key components of the camera’s extrinsic parameters Ko. Among them, $RotationMatrices$ represents the rotation of the camera relative to the ground plane, while $TranslationVectors$ represents the offset of the camera relative to the ground plane, in millimeters.

#### Sample
```txt
RotationMatrices
0.00000 -1.00000 0.00000
-0.05234 0.00000 -0.99863
0.99863 0.00000 -0.05234

TranslationVectors
0 1391 3968 

IntrinsicMatrix
1213 0 960 
0 1213 540 
0 0 1 
```



## 🗺 Roadmap

We are continuously updating UCMCTrack and warmly welcome contributions to enhance its value for the community. Our current high-priority tasks are as follows:

- [x] ~~Release code for replicating results on MOT17 dataset.~~
- [x] ~~Release code for replicating results on MOT20 dataset.~~
- [x] ~~Implement a demo of UCMCTrack based on YOLOv8.~~
- [x] ~~Release code for estimating camera parameters from a single picture.~~
- [ ] Release code for replicating results on DanceTrack dataset.
- [ ] Release code for replicating results on Kitti dataset.

## 📋 Citation
```bibtex
@inproceedings{yi2024ucmc,
  title={UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation},
  author={Kefu Yi, Kai Luo, Xiaolei Luo, Jiangui Huang, Hao Wu, Rongdong Hu, Wei Hao},
  booktitle={arXiv preprint arXiv:2312.08952},
  year={2023}
}
```

