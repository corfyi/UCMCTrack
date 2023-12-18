import numpy as np

# 定义一个GMCLoader类，用于加载GMC文件
class GMCLoader:
    def __init__(self, gmc_file):
        self.gmc_file = gmc_file
        self.affines = dict()
        self.load_gmc()

    def load_gmc(self):
        # 打开文件self.gmc_file
        with open(self.gmc_file, 'r') as f:
            # 读取文件中的每一行
            for line in f.readlines():
                # 将每一行的内容按照空格或者tab分开
                line = line.strip().split()
                frame_id = int(line[0]) + 1

                # 新建一个2x3的矩阵
                affine = np.zeros((2, 3))
                # 将每一行的内容转换为float类型
                affine[0, 0] = float(line[1])
                affine[0, 1] = float(line[2])
                affine[0, 2] = float(line[3])
                affine[1, 0] = float(line[4])
                affine[1, 1] = float(line[5])
                affine[1, 2] = float(line[6])

                self.affines[frame_id] = affine


    def get_affine(self,frame_id):
        return self.affines[frame_id]