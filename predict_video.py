import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import datetime
import cv2
from PIL import Image
import tkinter as tk
import sys
import json
import base64
import os
import cv2
import base64
import time
##############################
# 保证兼容python2以及python3
IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode
# 防止https证书校验不正确
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

API_KEY = 'GmhC18eVP1Fo1ECX911dtOzw'

SECRET_KEY = 'PQ2ukO4Aec2PTsgQU9UkiEKYciavlZk8'


OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"


"""  TOKEN start """
TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'

"""
    获取token
"""
def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': 'yiqtCVnEPZytrsKCXGnU6XBw',
              'client_secret': 'dcChZs8DLPiafYVHGxk4GyxulSCUVYs6'}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    if (IS_PY3):
        result_str = result_str.decode()


    result = json.loads(result_str)

    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print ('please ensure has check the  ability')
            exit()
        return result['access_token']
    else:
        print ('please overwrite the correct API_KEY and SECRET_KEY')
        exit()

"""
    读取文件
"""
def read_file(image_path):
    f = None
    try:
        f = open(image_path, 'rb')
        return f.read()
    except:
        print('read image file fail')
        return None
    finally:
        if f:
            f.close()
"""
    调用远程服务
"""
def request(url, data):
    req = Request(url, data.encode('utf-8'))
    has_error = False
    try:
        f = urlopen(req)
        result_str = f.read()
        if (IS_PY3):
            result_str = result_str.decode()
        return result_str
    except  URLError as err:
        print(err)
"""
    OCR识别
    预期作用：   输入一个img
                输出识别出来的str文档
"""  
def OCR(img_frame):
    # 读取图像数据并进行颜色空间转换 
    # 将图像编码为JPEG格式的二进制数据并进行base64编码
    retval, buffer = cv2.imencode('.jpg', img_frame)
    data = base64.b64encode(buffer).decode('utf-8')
    # 获取access token
    token = fetch_token()
    # 拼接通用文字识别高精度url
    image_url = OCR_URL + "?access_token=" + token
    result = request(image_url, urlencode({'image': data}))
    result_json = json.loads(result)
    text = ""
    for words_result in result_json["words_result"]:
        text += words_result["words"]
    return text
def update_text():
    # 获取输入框的文本
    text = entry.get()
    # 在文本框末尾插入新的文本
    text_box.insert(tk.END, text + "\n")
    # 清空输入框
    entry.delete(0, tk.END)

# 创建窗口
window = tk.Tk()
# 设置窗口标题
window.title("实时输出")
# 设置窗口大小
window.geometry("400x300")

# 创建文本框，用于显示文本
text_box = tk.Text(window)
text_box.pack()
window.update()

def process1(result):
    boxes = result[0].boxes
    cls= boxes.cls.cpu().numpy()
    length = cls.shape[0]
    button1 = np.where(cls == 1) # 复选框按下
    button3 = np.where(cls == 3) # 单选按钮按下
    button4 = np.where(cls == 4) # 工具栏
    button5 = np.where(cls == 5) # 工具栏名称
    
    return button1,button3,button4,button5

def process2(result):
    boxes = result[0].boxes
    cls= boxes.cls.cpu().numpy()
    button1 = np.where(cls == 8)
    button2 = np.where(cls == 9)
    return button1,button2

'''
def Ocr(img):
    img_fp = img
    ocr = CnOcr(lang_config='densenet-lite-fc',gpu = 'True', ocr_threshold=0.5)  # 所有参数都使用默认值
    img = Image.fromarray(img_fp)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 调整亮度
    # brightness = 5  # 亮度增加50
    # img = cv2.add(img, brightness)
    # 调整对比度
    # contrast = 0.5  # 对比度增加1.5倍
    # img = cv2.multiply(img, contrast)
    # 锐化图像
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])  # 锐化核
    gaussian = cv2.getGaussianKernel(5, 0.8) # 创建高斯模糊核
    gaussian_2d = np.outer(gaussian, gaussian.transpose()) # 将高斯模糊核转化为二维矩阵
    kernel = np.array([[0,0,-1,0,0],
                    [0,-1,-2,-1,0],
                    [-1,-2,16,-2,-1],
                    [0,-1,-2,-1,0],
                    [0,0,-1,0,0]])*gaussian_2d # 将高斯模糊核与Laplacian锐化核相乘得到最终的锐化核
    img = cv2.filter2D(img, -1, kernel)
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    out = ocr.ocr(thresh)
    return out
'''

model1 = YOLO('./BCD.pt',task='source')# 单选按钮 复选框 工具栏 BCD
model2 = YOLO('./button.pt',task='source')# 按钮
#task 参数不知道有什么用

file = open('output.txt', 'w')
# file.close()

# 打开视频文件
cap = cv2.VideoCapture('./video/end.mp4')
# 0 指的是摄像头  也可以是文件路径，或者是screen
lb = 0
ld = 0
l_dark = 0
l_concave = 0
radiobutton_pre =[]
checkbox_pre = []
toolbar_pre = []
# 预处理第一帧
Start = datetime.datetime.now()
initial_time = 0
# 将视频的当前位置设置为初始时间
cap.set(cv2.CAP_PROP_POS_MSEC, initial_time)
num = 1
success,frame = cap.read()
fps=1
# = cap.get(cv2.CAP_PROP_FPS)
# 计算一帧的时间间隔
frame_time_interval = datetime.timedelta(seconds=1 / fps)
if success:
    results1 = model1.predict(frame,conf=0.6,iou=0.8)
    results2 = model2.predict(frame,conf=0.5,iou=0.7)
    annotated_frame = results2[0].plot()
    # cv2.imshow('YOLOv8 inference', annotated_frame)
    # cv2.imwrite('XXX.jpg', annotated_frame)
    checkbox_on,radiobutton_on,toolbar,toolbar_name= process1(results1)
    lenB = len(checkbox_on[0])
    lenC = len(radiobutton_on[0])
    lenD = len(toolbar[0])
    lb = lenB
    ld = lenD

    boxes = results1[0].boxes
    for i in range(lenB):
        x1 = boxes[checkbox_on[0][i]].data[0][0]
        y1 = boxes[checkbox_on[0][i]].data[0][1]
        x2 = boxes[checkbox_on[0][i]].data[0][2]
        y2 = boxes[checkbox_on[0][i]].data[0][3]
        x11 = x1.cpu().numpy()
        y11 = y1.cpu().numpy()
        x22 = x2.cpu().numpy()
        y22 = y2.cpu().numpy()
        checkbox_pre.append([x11, y11, x22, y22])
    
    if(lenD ==1):
        x1 = boxes[toolbar[0][i]].data[0][0]
        y1 = boxes[toolbar[0][i]].data[0][1]
        x2 = boxes[toolbar[0][i]].data[0][2]
        y2 = boxes[toolbar[0][i]].data[0][3]
        x11 = x1.cpu().numpy()
        y11 = y1.cpu().numpy()
        x22 = x2.cpu().numpy()
        y22 = y2.cpu().numpy()
        toolbar_pre.append([x11, y11, x22, y22])


    # radiobutton_pre =[]
    # print(boxes[0].data[0][0])
    for i in range(lenC):
        # print(radiobutton_on[0][i])
        x1 = boxes[radiobutton_on[0][i]].data[0][0]
        y1 = boxes[radiobutton_on[0][i]].data[0][1]
        x2 = boxes[radiobutton_on[0][i]].data[0][2]
        y2 = boxes[radiobutton_on[0][i]].data[0][3]
        x11 = x1.cpu().numpy()
        y11 = y1.cpu().numpy()
        x22 = x2.cpu().numpy()
        y22 = y2.cpu().numpy()
        radiobutton_pre.append([x11, y11, x22, y22])

save_step = 1
frame_pre = frame

# def 

while cap.isOpened():
    success,frame = cap.read()
    #读取一帧
    if success:
        # 获取视频的当前时间（以毫秒为单位）
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        # 将当前时间转换为时间间隔
        current_time_interval = datetime.timedelta(milliseconds=current_time)
        new_time = Start + current_time_interval
        formatted_time = new_time.strftime("%Y-%m-%d %H:%M:%S")
        num += 1
        if num % save_step == 0:
            results1 = model1.predict(frame,conf=0.55,iou=0.7,hide_labels="False")
            results2 = model2.predict(frame,conf=0.55,iou=0.7,hide_labels="False")
            # annotated_frame = results1[0].plot()
            annotated_frame = results1[0].plot()
            cv2.imshow('YOLOv8 inference', annotated_frame)
            checkbox_on,radiobutton_on,toolbar,toolbar_name = process1(results1)
            lenB = len(checkbox_on[0])
            lenC = len(radiobutton_on[0])
            lenD = len(toolbar[0])
            
            # 工具栏处理
            toolbar_now = []
            if(lenD == 1):
                x1 = results1[0].boxes[toolbar[0][0]].data[0][0]
                y1 = results1[0].boxes[toolbar[0][0]].data[0][1]
                x2 = results1[0].boxes[toolbar[0][0]].data[0][2]
                y2 = results1[0].boxes[toolbar[0][0]].data[0][3]
                x11 = x1.cpu().numpy()
                y11 = y1.cpu().numpy()
                x22 = x2.cpu().numpy()
                y22 = y2.cpu().numpy()
                toolbar_now.append([x11, y11, x22, y22])
                i = toolbar_now
                j = toolbar_pre
                if ld == 0:
                    if(len(toolbar_name[0]) >= 1):
                        x1 = results1[0].boxes[toolbar_name[0][0]].data[0][0]
                        y1 = results1[0].boxes[toolbar_name[0][0]].data[0][1]
                        x2 = results1[0].boxes[toolbar_name[0][0]].data[0][2]
                        y2 = results1[0].boxes[toolbar_name[0][0]].data[0][3]
                        x11 = x1.cpu().numpy()
                        y11 = y1.cpu().numpy()
                        x22 = x2.cpu().numpy()
                        y22 = y2.cpu().numpy()
                        img = frame[int(y11):int(y22), int(x11):int(x22)]
                        out = OCR(img)
                    else:
                        out=''
                    text = "工具栏控件记录---"+formatted_time+"选中工具栏"+'"'+out+'"'+'\n'
                    file.write(text)
                    text_box.insert(tk.END,text)
                    text_box.pack()
                    window.update()
            
            # 复选框处理
            checkbox_now =[]
            for i in range(lenB):
                x1 = results1[0].boxes[checkbox_on[0][i]].data[0][0]
                y1 = results1[0].boxes[checkbox_on[0][i]].data[0][1]
                x2 = results1[0].boxes[checkbox_on[0][i]].data[0][2]
                y2 = results1[0].boxes[checkbox_on[0][i]].data[0][3]
                x11 = x1.cpu().numpy()
                y11 = y1.cpu().numpy()
                x22 = x2.cpu().numpy()
                y22 = y2.cpu().numpy()
                checkbox_now.append([x11, y11, x22, y22])
            
            if(lenB == lb+1):
                for i in checkbox_now:
                    tag = False
                    for j in checkbox_pre:
                        if np.abs(i[0]-j[0]) <= 20 and np.abs(i[1]-j[1]) <= 20:
                            tag = True
                    if tag == False:
                        x1 = i[0]
                        y1 = i[1]
                        x2 = i[2]
                        y2 = i[3]
                        img = frame[int(y1):int(y2), int(x1):int(x2)]
                        out = OCR(img)
                        # text = str(num)
                        text = "复选框控件记录---"+formatted_time+"勾选复选框"+'"'+out+'"'+'\n'
                        file.write(text)
                        text_box.insert(tk.END,text)
                        text_box.pack()
                        window.update()
                        break
            
            if(lenB == lb-1):
                for i in checkbox_pre:
                    tag = False
                    for j in checkbox_now:
                        if np.abs(i[0]-j[0]) <= 20 and np.abs(i[1]-j[1]) <= 20:
                            tag = True
                    if tag == False:
                        x1 = i[0]
                        y1 = i[1]
                        x2 = i[2]
                        y2 = i[3]
                        img = frame[int(y1):int(y2), int(x1):int(x2)]
                        out = OCR(img)
                        # text = str(num)
                        text = "复选框控件记录---"+formatted_time+"取消复选框"+'"'+out+'"'+'\n'
                        file.write(text)
                        text_box.insert(tk.END,text)
                        text_box.pack()
                        window.update()
                        break

            # 单选按钮位置处理
            radiobutton_now =[]
            for i in range(lenC):
                img = frame
                x1 = results1[0].boxes[radiobutton_on[0][i]].data[0][0]
                y1 = results1[0].boxes[radiobutton_on[0][i]].data[0][1]
                x2 = results1[0].boxes[radiobutton_on[0][i]].data[0][2]
                y2 = results1[0].boxes[radiobutton_on[0][i]].data[0][3]
                x11 = x1.cpu().numpy()
                y11 = y1.cpu().numpy()
                x22 = x2.cpu().numpy()
                y22 = y2.cpu().numpy()
                radiobutton_now.append([x11, y11, x22, y22])

            for i in radiobutton_now:
                tag = False
                for j in radiobutton_pre:
                    if np.abs(i[0]-j[0]) <= 20 and np.abs(i[1]-j[1]) <= 20:
                        tag = True
                if tag == False:
                    x1 = i[0]
                    y1 = i[1]
                    x2 = i[2]
                    y2 = i[3]
                    img = frame[int(y1):int(y2), int(x1):int(x2)]
                    out = OCR(img)
                    file.write(str(num))
                    # cv2.imwrite(str(num-save_step)+'.jpg', annotated_frame)
                    # cv2.imwrite(str(num)+'.jpg', annotated_frame)
                    # text = str(num)
                    text = "单选按钮控件记录---"+formatted_time+"选中单选按钮"+'"'+out+'"\n'
                    file.write(text)
                    text_box.insert(tk.END,text)
                    text_box.pack()
                    window.update()
                    break
            # if(num==260):
            #     cv2.imwrite(str(num-save_step)+'.jpg', annotated_frame)
            #     cv2.imwrite(str(num)+'.jpg', annotated_frame)
            
            # 按钮处理 
            dark_button, concave_button = process2(results2)
            # 单选按钮 取消的记录位置 取得中心点 如果和前面的差别太大则选中且立即取消
            # 第一类按钮的话则只需要识别按下的 颜色深的 还有凹下去的
            lendark = len(dark_button[0])
            lenconcave = len(concave_button[0])

            # text="lendark"+str(lendark)+" lenconcave"+str(lenconcave)+"\n"
            # file.write(text)
            # text_box.insert(tk.END,text)
            # text_box.pack()
            # window.update()
            if(lendark == 1 and l_dark == 0):
                x1 = results2[0].boxes[dark_button[0][0]].data[0][0]
                y1 = results2[0].boxes[dark_button[0][0]].data[0][1]
                x2 = results2[0].boxes[dark_button[0][0]].data[0][2]
                y2 = results2[0].boxes[dark_button[0][0]].data[0][3]
                img = frame[int(y1):int(y2), int(x1):int(x2)]
                out = OCR(img)
                # text = str(num)
                text = "按钮控件记录---"+formatted_time+"选中按钮"+'"'+out+'"\n'
                file.write(text)
                text_box.insert(tk.END,text)
                text_box.pack()
                window.update()
            if(lenconcave == 1 and l_concave == 0):
                x1 = results2[0].boxes[concave_button[0][0]].data[0][0]
                y1 = results2[0].boxes[concave_button[0][0]].data[0][1]
                x2 = results2[0].boxes[concave_button[0][0]].data[0][2]
                y2 = results2[0].boxes[concave_button[0][0]].data[0][3]
                img = frame[int(y1):int(y2), int(x1):int(x2)]
                out = OCR(img)
                # text = str(num)
                text = "按钮控件记录---"+formatted_time+"选中按钮"+'"'+out+'"\n'
                file.write(text)
                text_box.insert(tk.END,text)
                text_box.pack()
                window.update()


            lb = lenB
            radiobutton_pre = radiobutton_now
            checkbox_pre = checkbox_now
            toolbar_pre = toolbar_now
            ld = lenD
            frame_pre = annotated_frame
            l_dark = lendark
            l_concave = lenconcave
            #如果按下q健，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #如果视频到达结尾，则退出循环
    else:
        break

#释放视频捕获对象并关闭显示窗口
window.mainloop()
cap.release()
cv2.destroyAllWindows()
