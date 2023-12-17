from detector.detector import Detector, Detection
import os

det_path = "det_results/mot17/yolox_x_ablation"
cam_path = "cam_para/mot17"
seq_name = "MOT17-02"
det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")

print(det_file)
print(cam_para)

detector = Detector(30, 1080, 1920)
detector.load(cam_para, det_file)

dets = detector.get_dets(2, 0.92)
for det in dets:
    print(det)

