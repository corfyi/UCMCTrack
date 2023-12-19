from util.run_ucmc import run_ucmc, make_args

if __name__ == '__main__':

    det_path = "det_results/mot17/bytetrack_x_mot17"
    cam_path = "cam_para/mot17"
    gmc_path = "gmc/mot17"
    out_path = "output/mot17"
    exp_name = "test"
    dataset  = "MOT17"
    args = make_args()

    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)