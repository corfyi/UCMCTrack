# UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation

**Abstract: ** Multi-object tracking (MOT) in video sequences remains a challenging task, especially in scenarios with significant camera movements. This is because targets can drift considerably on the image plane, leading to erroneous tracking outcomes. Addressing such challenges typically requires supplementary appearance cues or Camera Motion Compensation (CMC). While these strategies are effective, they also introduce a considerable computational burden, posing challenges for real-time MOT. In response to this, we introduce UCMCTrack, a novel motion model-based tracker robust to camera movements. Unlike conventional CMC that computes compensation parameters frame-by-frame, UCMCTrack consistently applies the same compensation parameters throughout a video sequence. It employs a Kalman filter on the ground plane and introduces the Mapped Mahalanobis Distance (MMD) as an alternative to the traditional Intersection over Union (IoU) distance measure. By leveraging projected probability distributions on the ground plane, our approach efficiently captures motion patterns and adeptly manages uncertainties introduced by homography projections. Remarkably, UCMCTrack, relying solely on motion cues, achieves state-of-the-art performance across a variety of challenging datasets, including MOT17, MOT20, DanceTrack and KITTI, with an exceptional speed of over 1000 FPS on a single CPU.

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

