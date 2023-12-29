import argparse
import cv2
import numpy as np
import copy

class CameraPara:

    def open(self,file):

        # 读取文件
        with open(file, 'r') as f:
            data = f.readlines()

        # 从文件中读取旋转矩阵
        R = []
        for i in range(1, 4):
            row = data[i].split()
            R.append([float(x) for x in row])
        R = np.array(R)

        # 从文件中读取平移向量
        T = []
        for i in range(6, 7):
            row = data[i].split()
            T.append([float(x)/1000.0 for x in row])
        T = np.array(T)

        self.Ko = np.column_stack((R, T.T))
        self.Ko = np.row_stack((self.Ko, np.array([0,0,0,1])))

        # 从文件中读取内参矩阵
        Ki = []
        for i in range(9, 12):
            row = data[i].split()
            Ki.append([float(x) for x in row])
        self.Ki =  np.array(Ki)
        # 在self.Ki中添加一列[0,0,0]
        self.Ki = np.column_stack((self.Ki, np.array([0,0,0])))

    def xy2uv(self,x,y):
        # 计算uv
        uv = np.dot(self.Ki, np.dot(self.Ko, np.array([x,y,0,1])))
        # 归一化
        uv = uv/uv[2]
        return int(uv[0]), int(uv[1])
    

def xy2uv(x,y,Ki,Ko):
    # 计算uv
    uv = np.dot(Ki, np.dot(Ko, np.array([x,y,0,1])))
    # 归一化
    uv = uv/uv[2]
    return int(uv[0]), int(uv[1])


# 定义转换函数
def get_real_theta(value):
    return (value - 250) / 10.0

def get_real_theta_z(value):
    return (value - 500) / 5.0

def get_real_focal(value):
    return (value - 100) * 5

def get_real_transition(value):
    return (value-30) * 0.04

# 定义滑动条回调函数

def update_value_display():
    global value_display,g_theta_x,g_theta_y,g_theta_z,g_focal,g_tz
    value_display.fill(0)  # 清空图像
    text = f"theta_x: {g_theta_x:.2f}"
    cv2.putText(value_display, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"theta_y: {g_theta_y:.2f}"
    cv2.putText(value_display, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"theta_z: {g_theta_z:.2f}"
    cv2.putText(value_display, text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"focal: {g_focal}"
    cv2.putText(value_display, text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    text = f"Tz: {g_tz:.2f}"
    cv2.putText(value_display, text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow('Values', value_display)

def on_theta_x_change(value):
    global g_theta_x
    g_theta_x = get_real_theta(value)
    update_value_display()
    
def on_theta_y_change(value):
    global g_theta_y
    g_theta_y = get_real_theta(value)
    update_value_display()

def on_theta_z_change(value):
    global g_theta_z
    g_theta_z = get_real_theta_z(value)
    update_value_display()

def on_focal_change(value):
    global g_focal
    g_focal = get_real_focal(value)
    update_value_display()

def on_tz_change(value):
    global g_tz
    g_tz = get_real_transition(value)
    update_value_display()

cv2.namedWindow('Values')
# 初始化一个空白图像来显示实际值
value_display = np.zeros((400, 300), dtype=np.uint8)
g_theta_x = 0
g_theta_y = 0
g_theta_z = 0
g_focal = 0
g_tz = 0

def main(args):

    cam_para = CameraPara()
    cam_para.open(args.cam_para)

    img = cv2.imread(args.img)
    # 获取img的大小
    height, width = img.shape[:2]

    cv2.namedWindow('CamParaSettings')
    # 添加ui界面来修改theta_x,theta_y,theta_z, 调节访问是-10到10，间隔0.2
    cv2.createTrackbar('theta_x', 'CamParaSettings', 250,500, on_theta_x_change)
    cv2.createTrackbar('theta_y', 'CamParaSettings', 250,500, on_theta_y_change)
    cv2.createTrackbar('theta_z', 'CamParaSettings', 500,1000, on_theta_z_change)
    cv2.createTrackbar('focal', 'CamParaSettings', 100,500, on_focal_change)
    cv2.createTrackbar('Tz', 'CamParaSettings', 30,500, on_tz_change)

    global g_theta_x,g_theta_y,g_theta_z

    # 循环一直到按下q键
    while True:
        theta_x = g_theta_x/180.0*np.pi
        theta_y = g_theta_y/180.0*np.pi
        theta_z = g_theta_z/180.0*np.pi
        Ki = copy.copy(cam_para.Ki)
        Ko = copy.copy(cam_para.Ko)
        R = Ko[0:3,0:3]
        Rx = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
        Ry = np.array([[np.cos(theta_y),0,np.sin(theta_y)],[0,1,0],[-np.sin(theta_y),0,np.cos(theta_y)]])
        Rz = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z),np.cos(theta_z),0],[0,0,1]])
        R = np.dot(R, np.dot(Rx, np.dot(Ry,Rz)))
        Ko[0:3,0:3] = R
        Ko[2,3] += g_tz
        Ki[0,0] += g_focal
        Ki[1,1] += g_focal

        img = cv2.imread(args.img)
        # x取值范围0-10，间隔0.1
        for x in np.arange(0,10,0.5):
            for y in np.arange(-5,5,0.5):
                u,v = xy2uv(x,y,Ki,Ko)
                cv2.circle(img, (u,v), 3, (0,255,0), -1)

        # 修改img的大小
        img = cv2.resize(img, (int(width*0.5),int(height*0.5)))
        cv2.imshow('img', img)
        key = cv2.waitKey(50)
        if key == ord('q'):
            break

    with open(args.cam_para, 'w') as f:
        f.write("RotationMatrices\n")
        for i in range(3):
            for j in range(3):
                f.write(str(R[i,j])+" ")
            f.write("\n")
        f.write("\nTranslationVectors\n")
        for i in range(3):
            f.write(str(int(Ko[i,3]*1000))+" ")
        f.write("\n\nIntrinsicMatrix\n")
        for i in range(3):
            for j in range(3):
                f.write(str(int(Ki[i,j]))+" ")
            f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--img', type=str, required= True,help='The image file ')
    parser.add_argument('--cam_para', type=str, required= True,help='The estimated camera parameters file ')
    args = parser.parse_args()
    main(args)
