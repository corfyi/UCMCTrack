import os
from eval.eval import eval


dataset_path = "E:/dataset/MOT17/train"
out_path = "output/mot17"
exp_name = "val"

seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

HOTA,IDF1,MOTA,AssA = eval(dataset_path,out_path, seqmap, exp_name,1,False)