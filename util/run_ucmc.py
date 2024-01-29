from detector.detector import Detector, Detection
from tracker.ucmc import UCMCTrack
from tracker.kalman import TrackStatus
from eval.interpolation import interpolate
import os,time

import argparse


class Tracklet():
    def __init__(self,frame_id,box):
        self.is_active = False
        self.boxes = dict()
        self.boxes[frame_id] = box

    def add_box(self, frame_id, box):
        self.boxes[frame_id] = box

    def activate(self):
        self.is_active = True


def make_args():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--seq', type=str, default = "MOT17-02", help='seq name')
    parser.add_argument('--fps', type=float, default=30.0, help='fps')
    parser.add_argument('--wx', type=float, default=0.1, help='wx')
    parser.add_argument('--wy', type=float, default=0.1, help='wy')
    parser.add_argument('--vmax', type=float, default=0.5, help='vmax')
    parser.add_argument('--a', type=float, default=10.0, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=30.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.6, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='detection confidence threshold')
    parser.add_argument("--cmc", action="store_true", help="use cmc or not.")
    parser.add_argument("--hp", action="store_true", help="use head padding or not.")
    parser.add_argument('--u_ratio', type=float, default=0.05, help='assignment threshold')
    parser.add_argument('--v_ratio', type=float, default=0.05, help='assignment threshold')
    parser.add_argument('--u_max', type=float, default=13, help='assignment threshold')
    parser.add_argument('--v_max', type=float, default=10, help='assignment threshold')
    parser.add_argument("--add_cam_noise", action="store_true", help="add noise to camera parameter.")
    args = parser.parse_args()


    g_u_ratio = args.u_ratio
    g_v_ratio = args.v_ratio
    g_u_max = args.u_max
    g_v_max = args.v_max

    print(f"u_ratio = {g_u_ratio}, v_ratio = {g_v_ratio}, u_max = {g_u_max}, v_max = {g_v_max}")


    return args

def run_ucmc(args, det_path = "det_results/mot17/yolox_x_ablation",
                   cam_path = "cam_para/mot17",
                   gmc_path = "gmc/mot17",
                   out_path = "output/mot17",
                   exp_name = "val",
                   dataset = "MOT17"):

    seq_name = args.seq

    eval_path = os.path.join(out_path,exp_name)
    orig_save_path = os.path.join(eval_path,seq_name)
    if not os.path.exists(orig_save_path):
        os.makedirs(orig_save_path)


    if dataset == "MOT17":
        det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}-SDP.txt")
    elif dataset == "MOT20":
        det_file = os.path.join(det_path, f"{seq_name}.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}.txt")

    gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")

    print(det_file)
    print(cam_para)

    detector = Detector(args.add_cam_noise)
    detector.load(cam_para, det_file,gmc_file)
    print(f"seq_length = {detector.seq_length}")

    a1 = args.a
    a2 = args.a
    high_score = args.high_score
    conf_thresh = args.conf_thresh
    fps = args.fps
    cdt = args.cdt
    wx = args.wx
    wy = args.wy
    vmax = args.vmax
    
    tracker = UCMCTrack(a1, a2, wx,wy,vmax, cdt, fps, dataset, high_score,args.cmc,detector)

    t1 = time.time()

    tracklets = dict()

    with open(result_file,"w") as f:
        for frame_id in range(1, detector.seq_length + 1):
            dets = detector.get_dets(frame_id, conf_thresh)
            tracker.update(dets,frame_id)
            if args.hp:
                for i in tracker.tentative_idx:
                    t = tracker.trackers[i]
                    if(t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    if t.id not in tracklets:
                        tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                    else:
                        tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                for i in tracker.confirmed_idx:
                    t = tracker.trackers[i]
                    if(t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    if t.id not in tracklets:
                        tracklets[t.id] = Tracklet(frame_id, dets[t.detidx].get_box())
                    else:
                        tracklets[t.id].add_box(frame_id, dets[t.detidx].get_box())
                    tracklets[t.id].activate()
            else:
                for i in tracker.confirmed_idx:
                    t = tracker.trackers[i] 
                    if(t.detidx < 0 or t.detidx >= len(dets)):
                        continue
                    d = dets[t.detidx]
                    f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")

        if args.hp:
            for frame_id in range(1, detector.seq_length + 1):
                for id in tracklets:
                    if tracklets[id].is_active:
                        if frame_id in tracklets[id].boxes:
                            box = tracklets[id].boxes[frame_id]
                            f.write(f"{frame_id},{id},{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f},-1,-1,-1,-1\n")

    interpolate(orig_save_path, eval_path, n_min=3, n_dti=cdt, is_enable = True)

    print(f"Time cost: {time.time() - t1:.2f}s")

