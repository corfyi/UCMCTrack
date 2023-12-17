from .mapper import Mapper
import numpy as np

# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从文本文件读取任意一帧中的目标检测的结果
class Detector:
    def __init__(self, fps,img_height,img_width):
        self.fps = fps
        self.img_height = img_height
        self.img_width = img_width

    def load(self,cam_para_file, det_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        self.load_detfile(det_file)

    def load_detfile(self, filename):
        self.dets = dict()
        # 打开文本文件filename
        with open(filename, 'r') as f:
            # 读取文件中的每一行
            for line in f.readlines():
                # 将每一行的内容按照空格分开
                line = line.strip().split(',')
                frame_id = int(line[0])
                det_id = int(line[1])
                # 新建一个Detection对象
                det = Detection(det_id)
                det.bb_left = float(line[2])
                det.bb_top = float(line[3])
                det.bb_width = float(line[4])
                det.bb_height = float(line[5])
                det.conf = float(line[6])
                det.det_class = int(line[7])
                if det.det_class == -1:
                    det.det_class = 0
                
                det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])

                # 将det添加到字典中
                if frame_id not in self.dets:
                    self.dets[frame_id] = []
                self.dets[frame_id].append(det)

    def get_dets(self, frame_id,conf_thresh = 0,det_class = 0):
        dets = self.dets[frame_id]
        dets = [det for det in dets if det.det_class == det_class and det.conf >= conf_thresh]
        return dets


