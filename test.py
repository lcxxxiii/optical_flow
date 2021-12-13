import numpy as np
import cv2
import matplotlib.pyplot as plt #直接用cv2中的imshow也可以画图，这里用plt绘图方便显示

#读取视频-----------------------------------------------------
path = "jiaochakou1.MP4"
cap = cv2.VideoCapture(path)
print("Is this video captured correctly? :",cap.isOpened()) #返回True表示视频读取成功

#计算视频总帧数---------------------------------------------------
frame_count = 0
all_frames = []
while(True):
    ret, frame = cap.read()
    if ret is False: #逐帧读取到最后一帧的下一帧时，ret=false
        break
    all_frames.append(frame)
    frame_count = frame_count + 1
print("The total number of frames:",frame_count)

#计算视频fps---------------------------------------------------
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') #返回cv2的版本，根据不同版本查阅fps的命令不同。我的版本是4.4.0
if int(major_ver) < 3:
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second: {0}".format(fps))
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second: {0}".format(fps))

#设置角点检测及光流法的参数-------------------------------------------

# 构建角点检测所需参数
feature_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 2, blockSize = 7,useHarrisDetector = True)
# maxCorners:角点的最大个数,
# qualityLevel:角点品质,
# minDistance:两个角点间最小间距，以像素为单位
# blockSize：计算协方差矩阵时窗口大小
# useHarrisDetector：是否使用Harris角点检测，为false，则使用Shi-Tomasi算子

# lucas kanade参数
lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))# lucas kanade光流法参数
# winSize表示选择多少个点进行u和v的求解，maxLevel表示空间金字塔的层数

# 轨迹颜色
# color = np.random.randint(0,255,(100,3))# 创建随机颜色
color = (0,255,0)

#获取第一帧的灰度图像及其角点-----------------------------------------
cap = cv2.VideoCapture(path) #这里必要要重新捕获视频，否则.read()不能正确获取第一帧

# 拿到第一帧图像并灰度化作为前一帧图片
ret, old_frame = cap.read() #获取第一帧
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) #找到原始灰度图
plt.figure("old_gray")
plt.imshow(old_gray)
plt.title('old_gray')
plt.show()
#cv2.imshow("old_gray",old_gray) #用cv2中的模块画图
#cv2.waitKey()

#获取第一帧图像old_frame的灰度图像old_gray中的角点p0
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params) 
print(p0)

#对每帧图像计算光流并绘制光流轨迹-----------------------------------------

#创建一个蒙版用来画轨迹,i.e.和每帧图像大小相同的全0张量
mask = np.zeros_like(old_frame) 

while(1):
     
     # 读取图片灰度化作为后一张图片的输入
    ret,frame = cap.read()
    try:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except:
        print("This video has been processed.")
        break

    # 计算每帧的光流
    # 进行光流检测需要输入前一帧和当前图像及前一帧检测到的角点
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    '''
    pl表示光流检测后的角点位置，st表示是否是运动的角点，err表示是否出错，
    old_gray表示输入前一帧图片，frame_gray表示后一帧图片，p0表示需要检测的角点，
    lk_params：winSize表示选择多少个点进行u和v的求解，maxLevel表示空间金字塔的层数
    '''

    
    # 读取运动了的角点，st == 1表示检测到的运动物体，即v和u表示为0
    good_new = p1[st==1]
    good_old = p0[st==1]

    # 绘制轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2) #添加了该帧光流的轨迹图
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color, 2)
        # frame = cv2.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
        frame = cv2.circle(frame,(int(a),int(b)),5,color,-1)
    #将该图和轨迹图合并
    img = cv2.add(frame,mask) 
    cv2.imshow('frame',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    

    # 更新上一帧的图像和追踪点（角点）位置
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)


cv2.destroyAllWindows()
cap.release()

