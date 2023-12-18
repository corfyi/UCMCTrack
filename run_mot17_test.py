from detector.detector import Detector, Detection
from tracker.ucmc import UCMCTrack
from tracker.kalman import TrackStatus
from eval.interpolation import interpolate
import os,time

import argparse


def main(args):
    det_path = "det_results/mot17/bytetrack_x_mot17"
    cam_path = "cam_para/mot17"
    gmc_path = "gmc/mot17"
    out_path = "output/mot17"
    exp_name = "test"

    seq_name = args.seq

    eval_path = os.path.join(out_path,exp_name)
    orig_save_path = os.path.join(eval_path,seq_name)
    if not os.path.exists(orig_save_path):
        os.makedirs(orig_save_path)

    

    det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
    cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")
    gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")
    result_file = os.path.join(orig_save_path,f"{seq_name}-SDP.txt")

    print(det_file)
    print(cam_para)

    detector = Detector()
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
    
    tracker = UCMCTrack(a1, a2, wx,wy,vmax, cdt, fps, "MOT17", high_score,detector)

    t1 = time.time()

    with open(result_file,"w") as f:
        for frame_id in range(1, detector.seq_length + 1):
            dets = detector.get_dets(frame_id, conf_thresh)
            tracker.update(dets,frame_id)
            for i in tracker.confirmed_idx:
                t = tracker.trackers[i] 
                if(t.detidx < 0 or t.detidx >= len(dets)):
                    continue
                d = dets[t.detidx]
                f.write(f"{frame_id},{t.id},{d.bb_left:.1f},{d.bb_top:.1f},{d.bb_width:.1f},{d.bb_height:.1f},{d.conf:.2f},-1,-1,-1\n")

    interpolate(orig_save_path, eval_path, n_min=3, n_dti=cdt, is_enable = True)

    print(f"Time cost: {time.time() - t1:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--seq', type=str, default = "MOT17-01", help='seq name')
    parser.add_argument('--fps', type=float, default=30.0, help='fps')
    parser.add_argument('--wx', type=float, default=0.1, help='wx')
    parser.add_argument('--wy', type=float, default=0.1, help='wy')
    parser.add_argument('--vmax', type=float, default=0.5, help='vmax')
    parser.add_argument('--a', type=float, default=10.0, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=30.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.6, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='detection confidence threshold')

    args = parser.parse_args()

    main(args)
