from detector.detector import Detector, Detection
from tracker.ucmc import UCMCTrack
from tracker.kalman import TrackStatus
import os,time
from eval.eval import eval

det_path = "det_results/mot17/yolox_x_ablation"
dataset_path = "E:/dataset/MOT17/train"
cam_path = "cam_para/mot17"
out_path = "output/mot17"
exp_name = "UCMC"
seq_name = "MOT17-02"
det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")
result_file = os.path.join(out_path,exp_name,f"{seq_name}-SDP.txt")

print(det_file)
print(cam_para)

detector = Detector()
detector.load(cam_para, det_file)
print(f"seq_length = {detector.seq_length}")

a1 = 10.0
a2 = 10.0
high_score = 0.7
conf_thresh = 0.4
fps = 30.0
cdt = 30
wx = 0.1
wy = 0.2
vmax = 0.5
tracker = UCMCTrack(a1, a2, wx,wx,vmax, cdt, fps, "MOT17", high_score)

t1 = time.time()

with open(result_file,"w") as f:
    for frame_id in range(1, detector.seq_length + 1):
        dets = detector.get_dets(frame_id, conf_thresh)
        tracker.update(dets)
        for i in tracker.confirmed_idx:
            t = tracker.trackers[i] 
            if(t.detidx < 0 or t.detidx >= len(dets)):
                continue
            d = dets[t.detidx]
            f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")
            
seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")
HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)
print(f"Time cost: {time.time() - t1:.2f}s")
