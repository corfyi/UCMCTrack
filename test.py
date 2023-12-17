from detector.detector import Detector, Detection
from tracker.ucmc import UCMCTrack
from tracker.kalman import TrackStatus
import os,time

det_path = "det_results/mot17/yolox_x_ablation"
cam_path = "cam_para/mot17"
seq_name = "MOT17-02"
det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")

print(det_file)
print(cam_para)

detector = Detector()
detector.load(cam_para, det_file)
print(f"seq_length = {detector.seq_length}")

a1 = 45.0
a2 = 20.0
high_score = 0.5
conf_thresh = 0.1
fps = 30.0
cdt = 20
wx = 0.1
wy = 0.1
vmax = 3.0
tracker = UCMCTrack(a1, a2, wx,wx,vmax, cdt, fps, "MOT17", high_score)

t1 = time.time()
for frame_id in range(1, detector.seq_length + 1):
    dets = detector.get_dets(frame_id, conf_thresh)
    tracker.update(dets)
    # print(f"Frame {frame_id}, num of dets = {len(dets)}")
    # print(f"Cofirmed:{len(tracker.confirmed_idx)}, Coasted:{len(tracker.coasted_idx)}, Tentative:{len(tracker.tentative_idx)}")

out_str = ""
for trk in tracker.trackers:
    if trk.status == TrackStatus.Confirmed:
        out_str += f"t{trk.id},d{trk.detidx} "
print(out_str)
print(f"Time cost: {time.time() - t1:.2f}s")
