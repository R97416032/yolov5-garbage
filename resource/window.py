"""
-------------------------------------------------
Project Name: yolov5-garbage
File Name: Window.py
Author: R
Create Date: 2022/03/27
Description：客户端，检测图片、视频以及模型更换
任玉洁于2022年4月1日完成
感谢女友灿灿对我身心长久以来的爱护及帮助
-------------------------------------------------
"""
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import os
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.datasets import  LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, check_imshow , non_max_suppression, scale_coords,  xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


'''窗口主类'''
class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        # 窗口标题
        self.setWindowTitle('水面垃圾检测系统')
        self.resize(1200, 800)
        # 窗口图标
        self.setWindowIcon(QIcon("resource/ui/images/R.jpg"))
        # 图片显示大小
        self.output_size = 720
        # 视频与图像源
        self.source=''
        self.device = 'cpu'
        # 设置线程通信，便于挂起视频检测线程
        self.stopEvent = threading.Event()
        # 摄像开关
        self.webcam = False
        # 区分source是图片还是视频的1标志位
        self.pic = False
        self.weight="resource/models/default.pt"
        # 指明默认模型和加载模型的设备
        self.model = self.model_load(weights=self.weight,device=self.device)
        print("内置模型加载完成!")
        # 初始化窗口组件
        self.initUI()

    '''载入模型'''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        # Half
        half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            model.model.half() if half else model.model.float()
        return model


    '''初始化界面组件'''
    def initUI(self):
        #字体设置
        font_title = QFont('宋体', 18)
        font_main = QFont('宋体', 16)
        # 图片识别界面
        img_detect_widget = QWidget()
        #这是QT的垂直布局
        img_detect_layout = QVBoxLayout()
        #图片显示区域
        mid_img_widget = QWidget()
        #这是QT的水平布局
        mid_img_layout = QHBoxLayout()
        #用QLabel显示图片
        self.source_img = QLabel()
        #默认占位图
        self.source_img.setPixmap(QPixmap("resource/ui/images/up.jpg"))
        self.source_img.setAlignment(Qt.AlignCenter)
        #添加组件到widget
        mid_img_layout.addWidget(self.source_img)
        #设置布局
        mid_img_widget.setLayout(mid_img_layout)
        #添加图片按钮
        add_img_button = QPushButton("上传图片")
        #检测按钮
        det_img_button = QPushButton("开始检测")
        #槽函数
        add_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect)
        #设置按钮字体
        add_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        #给按钮添加样式
        add_img_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: #bbd69d;}"#50绿
            "QPushButton{background-color:rgb(128,0,32)}"#勃艮第红
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: #bbd69d;}"#50绿
            "QPushButton{background-color:rgb(128,0,32)}"#勃艮第红
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}")
        # 添加组件到布局上
        img_detect_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detect_layout.addWidget(add_img_button)
        img_detect_layout.addWidget(det_img_button)
        img_detect_widget.setLayout(img_detect_layout)

        '''
        视频检测界面，原理与上相似
        '''
        vedio_detect_widget = QWidget()
        vedio_detect_layout = QVBoxLayout()
        self.vedio_img = QLabel()
        #占位图
        self.vedio_img.setPixmap(QPixmap("resource/ui/images/upvedio.jpg"))
        self.vedio_img.setAlignment(Qt.AlignCenter)
        self.webcam_detect_btn = QPushButton("摄像头实时检测")
        self.mp4_detect_btn = QPushButton("视频文件检测")
        self.vedio_stop_btn = QPushButton("停止检测")
        self.webcam_detect_btn.setFont(font_main)
        self.mp4_detect_btn.setFont(font_main)
        self.vedio_stop_btn.setFont(font_main)
        self.webcam_detect_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: #94d2ef;}"#10元蓝
            "QPushButton{background-color:#bac3ab}"#1元绿
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}")
        self.mp4_detect_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: #94d2ef;}"#10元蓝
            "QPushButton{background-color:#bac3ab}"#1元绿
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}")
        self.vedio_stop_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: #94d2ef;}"#10元蓝
            "QPushButton{background-color:#bac3ab}"#1元绿
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}")
        #未开始进行检测的时候，停止检测按钮不可使用
        self.vedio_stop_btn.setEnabled(False)
        self.webcam_detect_btn.clicked.connect(self.open_cam)
        self.mp4_detect_btn.clicked.connect(self.open_mp4)
        self.vedio_stop_btn.clicked.connect(self.close_vedio)
        # 添加组件到布局上
        vedio_detect_layout.addWidget(self.vedio_img)
        vedio_detect_layout.addWidget(self.webcam_detect_btn)
        vedio_detect_layout.addWidget(self.mp4_detect_btn)
        vedio_detect_layout.addWidget(self.vedio_stop_btn)
        vedio_detect_widget.setLayout(vedio_detect_layout)

        ''' 
        模型更换页面，同上，不多赘述
        '''
        change_widget = QWidget()
        change_layout = QHBoxLayout()
        change_title = QLabel('位置：')
        change_title.setFont(font_main)
        self.change_path = QLineEdit()
        self.change_path.setReadOnly(True)
        self.change_path.setFont(font_main)
        self.change_btn = QPushButton("选择模型")
        self.load_btn = QPushButton("载入模型")
        self.change_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color: #509a80;}"  # 波尔多红
            "QPushButton{background-color:#dba880}"  # 20元棕
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        self.load_btn.setStyleSheet(
            "QPushButton{color:white}"
            "QPushButton:hover{background-color:#badde9;}"  # 50元蓝
            "QPushButton{background-color: #d5c0cf}"  # 5元紫
            "QPushButton{border:2px}"
            "QPushButton{border-radius:5px}"
            "QPushButton{padding:5px 5px}"
            "QPushButton{margin:5px 5px}"
        )
        self.change_btn.clicked.connect(self.choose_model)
        self.load_btn.clicked.connect(self.load_model)
        #设置间隔
        change_layout.setSpacing(7)
        change_layout.addWidget(change_title)
        change_layout.addWidget(self.change_path)
        change_layout.addWidget(self.change_btn)
        change_layout.addWidget(self.load_btn)
        change_widget.setLayout(change_layout)
        '''
        依然同上
        '''
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用水面垃圾检测系统！')
        about_title.setFont(font_title)
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        # 这里的 scaled(500,700) 因为图片过大，设置一下大小，避免显示变形
        about_img.setPixmap(QPixmap('resource/ui/images/contact2me.jpg').scaled(500, 700))
        about_img.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(about_title)
        about_layout.addWidget(about_img)
        about_widget.setLayout(about_layout)

        #将tab页添加
        self.addTab(img_detect_widget, '图片检测')
        self.addTab(vedio_detect_widget, '视频检测')
        self.addTab(change_widget, '更换模型')
        self.addTab(about_widget, '联系我')
        #设置tab的图标
        self.setTabIcon(0, QIcon('resource/ui/images/first.jpg'))
        self.setTabIcon(1, QIcon('resource/ui/images/second.jpg'))
        self.setTabIcon(2, QIcon('resource/ui/images/third.jpg'))
        self.setTabIcon(3, QIcon('resource/ui/images/fourth.jpg'))


    '''上传图片'''
    def upload_img(self):
        #选择图片
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            if not os.path.exists('resource/ui/temp'):
                os.makedirs('resource/ui/temp')
            im0 = cv2.imread(fileName)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            #修改大小存入我们的临时文件夹，显示在占位图上
            cv2.imwrite("resource/ui/temp/source.jpg", im0)
            #检测时我们使用原图，减小误差
            self.source = fileName
            #显示预览
            self.source_img.setPixmap(QPixmap("resource/ui/temp/source.jpg"))
            #图片检测修改pic位，以便知晓source位图片文件，而不是视频
            self.pic = True


    '''
    检测函数，主要参照yolo源码detect.py
    '''
    def detect(self):
        model = self.model
        output_size = self.output_size
        imgsz = 960  # inference size (pixels)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        view_img = False  # show results
        save_txt = False  # save results to *.txt
        save_conf = False  # save confidences in --save-txt labels
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 2  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        dnn = False  # use OpenCV DNN for ONNX inference
        source = str(self.source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        '''
        读取数据
        如果是视频检测修改source为0，pic为False，webcam为False
        如果是摄像检测，修改source为0，pic为False，webcam为True
        '''
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
                if self.pic:
                    line_thickness=4
            if pt and device.type != 'cpu':
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            '''
            以下代码可参阅，yolo源码中的detect。py
            '''
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3
                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # 保存结果
                    im0 = annotator.result()
                    frame = im0
                    resize_scale = output_size / frame.shape[0]
                    frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                    #将结果保存为result.jpg用来替换占位图
                    cv2.imwrite("resource/ui/temp/result.jpg", frame_resized)
                    '''
                    如果pic为True说明为图片检测修改图片检测的占位图，
                    如果为False则说明为视频检测，修改视频区域的占位图
                    '''
                    if self.pic:
                        #显示结果
                        self.source_img.setPixmap(QPixmap("resource/ui/temp/result.jpg"))
                        '''
                        图片检测结束清空源文
                        恢复pic标志
                        '''
                        self.source=""
                        self.pic=False
                    else:
                        #修改视频检测区域占位图
                        self.vedio_img.setPixmap(QPixmap("resource/ui/temp/result.jpg"))
                '''
                检查是否有停止检测信号，
                如果点击停止检测，将stopEvent.set()
                设置stopEvent.is_set()为True
                '''
                if self.stopEvent.is_set() == True:
                    #恢复stopEvent信号
                    self.stopEvent.clear()
                    self.webcam_detect_btn.setEnabled(True)
                    self.mp4_detect_btn.setEnabled(True)
                    self.reset_vedio()
                    break


    '''使用摄像进行视频检测'''
    def open_cam(self):
        #视频检测时设置按钮不可用，设置停止按钮可用
        self.webcam_detect_btn.setEnabled(False)
        self.mp4_detect_btn.setEnabled(False)
        self.vedio_stop_btn.setEnabled(True)
        self.source = '0'
        self.webcam = True
        #开启子线程
        th = threading.Thread(target=self.detect)
        '''
        开启守护线程，
        主线程终止子线程也会终止，
        否则主线程终止，子线程还在运行
        '''
        th.setDaemon(True)
        #启动
        th.start()

    '''选择视频文件进行检测'''
    def open_mp4(self):
        self.pic = False
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detect_btn.setEnabled(False)
            self.mp4_detect_btn.setEnabled(False)
            self.vedio_stop_btn.setEnabled(True)
            self.source = fileName
            self.webcam = False
            #同上，不赘述
            th = threading.Thread(target=self.detect)
            th.setDaemon(True)
            th.start()


    '''关闭视频检测'''
    def close_vedio(self):
        #将进程信号设置为True便于停止
        self.stopEvent.set()
        #重置视频界面
        self.reset_vedio()


    '''重置视频界面'''
    def reset_vedio(self):
        self.webcam_detect_btn.setEnabled(True)
        self.mp4_detect_btn.setEnabled(True)
        self.vedio_stop_btn.setEnabled(False)
        self.vedio_img.setPixmap(QPixmap("resource/ui/images/upvedio.jpg"))
        self.source = ""
        self.webcam = False


    '''选择新模型'''
    def choose_model(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        if fileName:
            name = fileName.split("/")[-1]
            if not os.path.exists('resource/models'):
                os.makedirs('resource/models')
            save_path = "resource/models/"+name
            '''
            检查缓存文件夹是否存在同名模型，有的话则不保存直接用
            没有就复制到程序文件夹中
            '''
            if not os.path.exists(save_path):
                shutil.copy(fileName, save_path)
        self.change_path.setText(fileName)
        self.weight=str(fileName)

    '''载入新模型'''
    def load_model(self):
        self.model = self.model_load(weights=self.weight,device=self.device)
        print("载入模型成功！")


    '''关闭程序提示'''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,'退出',"是否退出?",QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
