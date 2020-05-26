import sys
import os
import time
import threading
import multiprocessing as mp
import cv2
from time import strftime
from datetime import datetime
from ModelsManager import ModelsManager
from Detection.ConfigManager import ConfigManager
from PIL import Image
import datetime
import numpy as np

from server.rtpm_url_config import *
import requests
import json
import base64
from LOG import *
from paddle_detection import Yolov3Detection
QUEUE_SIZE = 2
PIXEL_MEANS = (0.485, 0.456, 0.406)  # RGB  format mean and variances
PIXEL_STDS = (0.229, 0.224, 0.225)

SHOW_TIME = True

class AiCameraData:
    def __init__(self, rtmp_str, val):
        self.rtmp_str = rtmp_str
        self.valid = False
        self.rtmp = val
        self.applydate = val.applydate
        self.time_out = False
        self.camera_id = val.camera_id
        self.gen_camera_time()
        self.invalid_cap_num = 0
        self.invalid_frame_num = 0 #用于记录获取视频帧數

    def get_rtmp_str(self):
        return self.rtmp_str

    def set_cap_handler(self, cap):
        self.cap_handler = cap

    def gen_camera_time(self):
        self.invalid_cap_num = 0
        self.last_scann_time = time.time()

    def get_camera_id(self):
        return self.camera_id

    def reset_invalid_frame_num(self):
        self.invalid_frame_num = 0

    def add_invalid_frame_num(self):
        self.invalid_frame_num += 1
        if self.invalid_frame_num > MAX_INVLAID_FRAME_NUM:
            self.valid = False
            self.time_out = True
            return True
        else:
            return False
    def release_cap(self):
        try:
            if self.cap_handler is not None:
                self.cap_handler.release()
        except:
            pass
        #self.valid = False
        #self.time_out = True

    def time_out_valid(self):
        cur_time = time.time()
        if (cur_time - self.last_scann_time) > MAX_WAIT_TIME:
            self.valid = False
            self.time_out = True
            self.release_cap()

            return True
        else:
            return False
    def get_cap_handler(self):
        return self.cap_handler

class VideoInf:

    def __init__(self):
        self.camera_list = list()
        self.rtpm_dict = dict()
        self.load_video_conf()

    def load_video_conf(self):
        self.rtpm_dict =  load_ai_glasses_rtpm_conf()

        for (key, val) in self.rtpm_dict.items():
            ai_camera_data = AiCameraData(key,val)
            self.camera_list.append(ai_camera_data)

    def refresh_rtpm_conf(self):
        cur_rtpm_conf = load_ai_glasses_rtpm_conf()

        for (key,val) in cur_rtpm_conf.items():
            if key not in self.rtpm_dict:
                ai_camera_data = AiCameraData(key, val)
                if self.reconnect_cap(key, ai_camera_data):
                    self.camera_list.append(ai_camera_data)
                else:
                    self.camera_list.append(ai_camera_data)
                self.rtpm_dict[key] = val

    def refresh_cap_list(self):
        for ai_camera_data in self.camera_list:
            cap = cv2.VideoCapture(ai_camera_data.get_rtmp_str())
            if cap.isOpened():
                ai_camera_data.valid = True
                ai_camera_data.set_cap_handler(cap)
                ai_camera_data.gen_camera_time()
            else:
                ai_camera_data.valid = False

                ai_camera_data.set_cap_handler(None)
        print('')

    def reconnect_cap(self,rtmp_str, ai_camera_data):
        #ai_camera_data.release_cap()
        cap = cv2.VideoCapture(rtmp_str)
        if cap.isOpened():

            ai_camera_data.valid = True
            ai_camera_data.gen_camera_time()
            ai_camera_data.set_cap_handler(cap)
            return True

        return False

    def remove_cap(self, ai_camera_data):
        rtmp_str = ai_camera_data.get_rtmp_str()
        if rtmp_str in self.rtpm_dict:
            self.rtpm_dict.pop(rtmp_str)
            save_rtpm_conf(self.rtpm_dict)
        for i in range(0,len(self.camera_list)):
            if rtmp_str == self.camera_list[i].get_rtmp_str():
                self.camera_list[i].release_cap()
                self.camera_list.pop(i)
                break

    def remove_camera_by_info(self, rtpmurl, applydate):
        rtpm = Rtpm(rtpmurl, applydate, gen_unique_cameraid())
        if rtpmurl in self.rtpm_dict:  # already exists
            for ai_camera_data in self.camera_list:
                if rtpmurl == ai_camera_data.get_rtmp_str():
                    self.remove_cap(ai_camera_data)
                    logger.info("remove camera " + rtpmurl)
                    break
        else:
            pass
        #save_rtpm_conf(self.rtpm_dict)


    def add_camera_by_info(self,rtpmurl, applydate):
        rtpm = Rtpm(rtpmurl, applydate, gen_unique_cameraid())

        if rtpmurl in self.rtpm_dict: #already exists
            for ai_camera_data in self.camera_list:
                if rtpmurl == ai_camera_data.get_rtmp_str():
                    cap = cv2.VideoCapture(ai_camera_data.get_rtmp_str())
                    if cap.isOpened():
                        ai_camera_data.valid = True
                        ai_camera_data.set_cap_handler(cap)
                        ai_camera_data.gen_camera_time()
                    else:
                        ai_camera_data.valid = False
                        ai_camera_data.release_cap()
                        ai_camera_data.set_cap_handler(None)
                    break
        else:
            ai_camera_data = AiCameraData(rtpmurl,rtpm)

            if self.reconnect_cap(rtpmurl, ai_camera_data):
                self.camera_list.append(ai_camera_data)
            else:
                self.camera_list.append(ai_camera_data)
            logger.info("add camera " + rtpmurl)
            self.rtpm_dict[rtpmurl] = rtpm
        save_rtpm_conf(self.rtpm_dict)


    def clear_cap_list(self):
        for ai_camera_data in self.camera_list:
            ai_camera_data.release_cap()
            ai_camera_data.valid = False
            ai_camera_data.set_cap_handler(None)

    def get_camera_list(self):
        return self.camera_list

class TaskInf:
    def __init__(self,ip, port , channel_num, task_dictt):
        self.ip = ip
        self.port = port
        self.channel_num = channel_num
        self.task_dict = task_dictt
    def get_task_list(self):
        task_list = list()
        for k,v in self.task_dict.items():
            task_list.append(k)
        return task_list


class TaskParser:
    def __init__(self, channel_task_path, task_conf_path):
        task_ens_lines = open(task_conf_path).readlines()
        task_map = dict()
        for task in task_ens_lines:
            task = task.strip('\n').strip('\r').strip()
            if task == '':
                continue
            task_ens = task.split('\t')
            if len(task_ens) > 1:
                key_task = task_ens[0]
                task_map[key_task] = task_ens[1:]

        self.device_task_map = dict()
        channel_inf_lines = open(channel_task_path).readlines()
        for device_groupt in channel_inf_lines:
            device_groupt = device_groupt.strip('\n').strip('\r').strip()
            if device_groupt == '':
                continue
            device_groupt_infs = device_groupt.split('\t')
            if len(device_groupt_infs) > 3:
                ip = device_groupt_infs[0]
                port = device_groupt_infs[1]
                channel_num = device_groupt_infs[2]
                task_uni_list = dict()
                for i in range(3,len(device_groupt_infs)):
                    task_settings_str = device_groupt_infs[i]
                    task_settings = task_settings_str.split('!')
                    if len(task_settings) > 0:

                        group_name = task_settings[0]
                        start_time = task_settings[1]
                        end_time = task_settings[2]
                        days = task_settings[3]
                        interval_time = task_settings[4]
                        set_date = task_settings[5]
                        linger_time = task_settings[6]

                        if group_name in task_map:
                            task_list = task_map[group_name]
                            for task_name in task_list:
                                task_uni_list[task_name] = 1
                        else:
                            continue
                key_str = ip + '_' + port + '_' + channel_num
                task_inf = TaskInf(ip,port,channel_num, task_uni_list)
                self.device_task_map[key_str] = task_inf

    def get_device_channel_task_list(self, ip, port, channel_num):
        key_str = ip + '_' + port + '_' + channel_num
        if  key_str in  self.device_task_map:
            return self.device_task_map[key_str].get_task_list()
        else:
            return []
# ctx = mx.gpu()
class lingerTask:
    def __init__(self, taskName, lingerTime):
        self.taskName = taskName
        self.startTime = ''
        self.alarmTime = ''
        self.lingerTime = lingerTime
        self.lastImOcurTime = ''
        self.valid = False  # 用于表明是否有效
        self.noAimNum = 0  # 计数
        self.missImNum = (int)(lingerTime / 10)  # only permit missing one image per 10 second

    def updateTime(self, imTime):  # 初始化 或更新当前时间信息
        self.startTime = datetime.datetime.strptime(imTime, '%Y:%m:%d_%H:%M:%S')
        self.alarmTime = self.startTime + datetime.timedelta(seconds=self.lingerTime)
        self.lastImOcurTime = self.startTime
        self.valid = True

    def compareTime(self, imTime):
        if not self.valid:
            self.updateTime(imTime)
            return -1

        curTime = datetime.datetime.strptime(imTime, '%Y:%m:%d_%H:%M:%S')
        if self.alarmTime <= curTime:  # need to alarm
            self.updateTime(imTime)
            return 1
        else:
            self.lastImOcurTime = imTime
            return -1

    def resetLingerState(self):
        if self.noAimNum > self.missImNum:  # no aim num
            self.valid = False
            self.noAimNum = 0
        else:
            self.noAimNum += 1

class DetectionStage:
    def __init__(self,target,pre_queue,next_queue, queue_size=None):
        self.target = target
        self.stopped = False
        self.prev = pre_queue
        self.loadConfig()
        # initialize the queue used to store data
        #if queue_size is not None:
        #self.Q = mp.Queue(maxsize=queue_size)
        self.Q = next_queue

    def get_next(self, timeout=None):
        if self.prev:
            return self.prev.get(timeout=timeout)
        else:
            return None
    def is_empty(self):
        if self.prev:
            if self.prev.qsize() > 0:
                return False
            else:
                return True
        return True

    def wait_for_queue(self, time_step):
        while self.Q.full():
            time.sleep(time_step)

    def wait_for_stop(self, time_step):
        if self.prev is not None:
            while not self.prev.stopped:
                time.sleep(time_step)
        while not self.Q.empty():
            time.sleep(time_step)

    def parseImName(self, imName):
        str = imName.split('.j')[0]
        strs = str.split('_')
        ip = strs[1]
        port = strs[2]
        channelNum = strs[3]
        dateTime = strs[4] + '_' + strs[5]
        return ip, port, channelNum, dateTime

    def loadConfig(self):
        configManager = ConfigManager()

        imRoot = configManager.getImageDir() + 'images/'
        projetRoot = configManager.getClientDir()
        self.modelRoot = configManager.getFasterRcnnDir()

        self.videoImageDir = projetRoot + 'detection_images/'
        self.rootSavePath = imRoot + 'detectTempImage/'
        self.rootDetectSavePath = imRoot + 'detectImage/'
        self.taskClassPath = projetRoot + 'config/taskClass.txt'

        self.ensembleTaskPath = projetRoot + 'config/ensembleTask.txt'

        self.interactionDir = projetRoot + "config/interaction/"

        self.deviceGroupTaskPath = projetRoot + 'config/deviceGroupTask.txt'
        self.groupTaskPath = projetRoot + 'config/groupTask.txt'
        self.fstTaskDetectThresPath = projetRoot + 'config/fstTaskDetectThres.txt'
        self.sndTaskDetectThresPath = projetRoot + 'config/sndTaskDetectThres.txt'
        self.trdTaskDetectThresPath = projetRoot + 'config/trdTaskDetectThres.txt'
        self.root_path = configManager.getFasterRcnnDir()

        fstLayFlag = configManager.getTestInfo('FstLayer')
        # 0 denotes using the normal ones, 1 denotes using tmp one
        if fstLayFlag == '0':
            self.detection1TaskListPath = projetRoot + configManager.getAlterConfigInfo('FstLayer', 'ClassPath')
        else:
            self.detection1TaskListPath = projetRoot + configManager.getAlterConfigInfo('FstLayer', 'ClassTmpPath')

        sndLayFlag = configManager.getTestInfo('SndLayer')
        # 0 denotes using the normal one, 1 denotes using tmp one
        if sndLayFlag == '0':
            self.detection2TaskListPath = projetRoot + configManager.getAlterConfigInfo('SndLayer', 'ClassPath')

        else:
            self.detection2TaskListPath = projetRoot + configManager.getAlterConfigInfo('SndLayer', 'ClassTmpPath')

        trdLayFlag = configManager.getTestInfo('TrdLayer')
        # 0 denotes using the normal one, 1 denotes using tmp one
        if trdLayFlag == '0':
            self.detection3TaskListPath = projetRoot + configManager.getAlterConfigInfo('TrdLayer', 'ClassPath')
        else:
            self.detection3TaskListPath = projetRoot + configManager.getAlterConfigInfo('TrdLayer', 'ClassTmpPath')

        self.task_conf = TaskParser(self.deviceGroupTaskPath, self.groupTaskPath)
        self.ai_camera_server_url = 'http://139.9.5.146:12072/webapi/api/aiReceiveRecord'

        self.imDetectResultDispatchTypePath = projetRoot + 'config/imDetectResultDispatchType.txt'
        self.detect_task_type = self.load_imDetectResultDispatchType(self.imDetectResultDispatchTypePath)

    def load_imDetectResultDispatchType(self, imDetectResultDispatchTypePath):
        detect_task_type = dict()
        with open(imDetectResultDispatchTypePath) as f:
            lines = f.readlines()
            for lin in lines:
                lin = lin.strip().strip('\r').strip('\n')
                if lin =='':
                    continue
                strs = lin.split('\t')
                if len(strs) > 1:
                    task_name = strs[0]
                    flag = int(strs[1])
                    detect_task_type[task_name] = flag

        return detect_task_type



def load_im(next_queue,pre_queuq):
    image_loader =  ImageLoader(None,next_queue)
    image_loader.loop_load_image()

def first_layer_detection(pre_queue,next_queue):
    first_layer = FirstLayerDetection(pre_queue,next_queue)
    first_layer.fstlayer_detection()

def detection_yolo(pre_queue,next_queue):
    detection_yolo = DetectionLoader(pre_queue,next_queue)
    detection_yolo.yolo_detection()

def pose_estimator(pre_queue,next_queue):
    pose_estimator_func = PoseEstimator(pre_queue, next_queue)
    pose_estimator_func.pose_estimating()

def detection_procedure(pre_queue,next_queue):
    detection_procedure_func = DetectionProcedure(pre_queue, next_queue)
    detection_procedure_func.im_detection()

def post_process(pre_queue, next_queue):
    post_process_func = PostProcess(pre_queue, next_queue)
    post_process_func.post_process()


class ImageLoader(DetectionStage):
    '''Load images for prediction'''

    def __init__(self, pre_queue,next_queue, queue_size=QUEUE_SIZE):
        super(ImageLoader, self).__init__(self.loop_load_image, pre_queue,next_queue, queue_size)

        curServerId = -1
        self.count = 0
        serverDir = self.interactionDir + 'server/'
        self.serverPath = ""
        for i in range(0, 20):  # the max  sever num is 20
            srvName = serverDir + str(i) + '.txt'
            # if os.path.exists(srvName):
            #    continue
            curServerId = 0
            self.serverPath = srvName
            break
        self.curServerId = 0
        f = open(self.serverPath, 'w')  # info the client the added server
        f.write('1\r\n')
        f.close()
        print("server started")

    def loadTaskClass(self, task_path):
        lines = open(task_path).readlines()
        class_list = list()
        for line in lines:
            line = line.strip().strip('\r').strip('\n')
            if line == '':
                continue
            class_list.append(line)
        return class_list

    def listen_camera_info(self):
        if os.path.exists(rtpmurl_info_path):
            time.sleep(0.2)
            lines = open(rtpmurl_info_path, 'r').readlines()
            logger.info("new rtpmurl")
            os.remove(rtpmurl_info_path)

            for line in lines:
                line = line.strip('\r').strip('\n').strip()
                if line == "":
                    continue
                strs = line.split('\t')
                if len(strs) > 2:
                    rtpmurl = strs[0]
                    logger.info("new rtpmurl:" + rtpmurl + " status:" + strs[2])
                    applydate  = strs[1]
                    camera_status = strs[2]
                    if camera_status == CAMERA_STATUS_SHUTDOWN:
                        self.video_inf.remove_camera_by_info(rtpmurl,applydate)
                    elif camera_status == CAMERA_STATUS_START:
                        self.video_inf.add_camera_by_info(rtpmurl, applydate)
                    else:
                        pass
                    break

    def loop_load_image(self):
        self.video_inf = VideoInf()
        self.video_inf.refresh_cap_list()
        ip = '192.168.101.64'
        port = '8000'
        channel_num = '1'
        device_name = 'test'
        update_type = '0'

        #cv2.namedWindow('hello', flags=cv2.WINDOW_FREERATIO)
        toc_refresh = time.time()
        while True:
            tic_refresh = time.time()

            tm_inter = tic_refresh - toc_refresh
            if tm_inter > 600:
                logger.info("heart detecting")
                toc_refresh = tic_refresh
            self.listen_camera_info()

            cap = cv2.VideoCapture('/home/zzj/work1/20200506_170555.mp4')
            if cap.isOpened():
                ret, image = cap.read()

                tic = time.time()
                frame = image
                time.sleep(0.01)
                if not ret or frame is None:
                    logger.info('invalid cap:' + device_name)
                    continue
                cv2.putText(frame, strftime("%H:%M:%S"), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('hello', frame)
                cv2.waitKey(1)
                while self.Q.qsize() > 1:
                    self.Q.get()

                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im = np.array(im)

                task_list = self.task_conf.get_device_channel_task_list(ip, port, channel_num)
                im_name, tmp_detect_save_dir_name = self.gen_im_name(device_name, ip, port, channel_num)
                # logger.info('valid im, camera name:' + im_name)
                detectTempImageDir = self.rootSavePath + tmp_detect_save_dir_name + '/'

                queue_data = dict()
                queue_data['rtmp_url'] = ''
                if not os.path.exists(detectTempImageDir):
                    os.mkdir(detectTempImageDir)
                self.wait_for_queue(0.3)
                self.Q.put((update_type, im, im_name, task_list, detectTempImageDir, True, queue_data))
                toc = time.time()
                # time_str = datetime.datetime.now().strftime('%Y:%m:%d_%H:%M:%S:%f')
                # logger.info(time_str)
                time.sleep(0.05)
                if SHOW_TIME:
                    pass
                    #logger.info('\t' + 'load all image time:' + str(toc - tic))

            '''
            for ai_camera_dat in self.video_inf.get_camera_list():
                #logger.info( ai_camera_dat.get_rtmp_str())
                if ai_camera_dat.valid:
                    cap = ai_camera_dat.get_cap_handler()
                    if cap.isOpened():
                        ret, image = cap.read()
                        ai_camera_dat.gen_camera_time()
                        device_name = ai_camera_dat.get_camera_id()
                        tic = time.time()
                        frame = image
                        time.sleep(0.01)

                        if not ret or frame is None:
                            logger.info('invalid cap:' + device_name)
                            time_out_flag = ai_camera_dat.add_invalid_frame_num()

                            if time_out_flag:
                                logger.info('invalid cap,delete the rtmp' + ai_camera_dat.get_rtmp_str())
                                self.video_inf.remove_cap(ai_camera_dat)
                            continue

                        ai_camera_dat.reset_invalid_frame_num()
                        #cv2.putText(frame, strftime("%H:%M:%S"), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        #            (0, 255, 0), 2, cv2.LINE_AA)
                        #cv2.imshow('hello', frame)
                        #cv2.waitKey(1)
                        while self.Q.qsize() > 1:
                            self.Q.get()

                        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        im = np.array(im)

                        task_list = self.task_conf.get_device_channel_task_list(ip, port, channel_num)
                        im_name, tmp_detect_save_dir_name = self.gen_im_name(device_name, ip, port, channel_num)
                        # logger.info('valid im, camera name:' + im_name)
                        detectTempImageDir = self.rootSavePath + tmp_detect_save_dir_name + '/'

                        queue_data = dict()
                        queue_data['rtmp_url'] = ai_camera_dat.get_rtmp_str()
                        if not os.path.exists(detectTempImageDir):
                            os.mkdir(detectTempImageDir)
                        self.wait_for_queue(0.3)
                        self.Q.put((update_type, im, im_name, task_list, detectTempImageDir, True, queue_data))
                        toc = time.time()
                        # time_str = datetime.datetime.now().strftime('%Y:%m:%d_%H:%M:%S:%f')
                        # logger.info(time_str)
                        time.sleep(0.05)
                        if SHOW_TIME:
                            pass
                            #logger.info(ai_camera_dat.get_rtmp_str() + '\t' + 'load all image time:' + str(toc - tic))
                    else:
                        time_out_ret =  ai_camera_dat.time_out_valid()
                        #logger.info('time out' + ai_camera_dat.get_rtmp_str() + ' num: ' + str(ai_camera_dat.invalid_cap_num))
                        if time_out_ret:
                            logger.info('time out,delete the rtmp' + ai_camera_dat.get_rtmp_str())
                            self.video_inf.remove_cap(ai_camera_dat)
                else:
                    time_out_ret = ai_camera_dat.time_out_valid()
                    #logger.info('time out' + ai_camera_dat.get_rtmp_str() + ' num: ' + str(ai_camera_dat.invalid_cap_num))
                    if time_out_ret:
                        logger.info('time out,delete the rtmp' + ai_camera_dat.get_rtmp_str())
                        self.video_inf.remove_cap(ai_camera_dat)
                '''

        self.wait_for_stop(1)
        # logger.info('ImageLoader: %fs' % (np.mean(time_rec)))
        self.stopped = True
    def gen_im_name(self,device_name, ip, port, channel_num):
        time_str = datetime.now().strftime('%Y:%m:%d_%H:%M:%S:%f')
        name_str = device_name + '_' + ip + '_' + port  + '_' + channel_num + '_'+time_str
        tmp_detect_save_dir_name = ip + '_' + port  + '_' + channel_num
        return name_str,tmp_detect_save_dir_name

class FirstLayerDetection(DetectionStage):
    def __init__(self, pre_queue,next_queue, queue_size=QUEUE_SIZE):
        super(FirstLayerDetection, self).__init__(self.fstlayer_detection,pre_queue,next_queue, queue_size)
        self.yolo_model = Yolov3Detection(1)

    def fstlayer_detection(self):
        while True:
            if self.is_empty():
                continue
            try:
                update_type, im, im_name, ann_task_list, detectTempImageDir, detect_mode, queue_data = self.get_next(
                    timeout=10)
            except Exception:
                continue
            if update_type == '2':  # update thresholds
                pass
                #self.modelsManager.loadDetection1(self.detection1TaskListPath, self.fstTaskDetectThresPath)
            else:
                tic = time.time()
                logger.info("first layer detection")
                tic = time.time()

                detect1dict = self.yolo_model.single_image_detect(im)


                toc = time.time()
                if SHOW_TIME:
                    logger.info('first layer time:' + str(toc - tic))
                self.wait_for_queue(1)
                self.Q.put(
                    (update_type, im,  im_name, ann_task_list, detectTempImageDir, detect1dict,
                     detect_mode, queue_data))

        self.wait_for_stop(1)
        self.stopped = True

class DetectionLoader(DetectionStage):
    def __init__(self, pre_queue, next_queue, queue_size=QUEUE_SIZE):
        super(DetectionLoader, self).__init__(self.yolo_detection, pre_queue, next_queue, queue_size)
        self.yolo = Yolov4()

    def yolo_detection(self):
        while (1):
            if self.is_empty():
                continue
            try:
                update_type, im, img_tensor, im_name, ann_task_list, detectTempImageDir, detect1dict, feats, detect_mode, queue_data = \
                    self.get_next(timeout=10)
            except Exception:
                continue
            tic = time.time()
            detect1dict['行人'] = []
            detect1dict['手机'] = []
            detect1dict['狗'] = []
            detect1dict['汽车'] = []
            detect1dict['卡车'] = []

            boxes = self.yolo.im_detect(im)

            for box in boxes:
                cls_conf = box[5]
                cls_id = box[6]
                detection = [box[0], box[1], box[2], box[3], box[5], 1.0,box[6]]
                detection = np.array(detection)
                # x1, y1, x2, y2, conf, cls_conf, cls_pred = detection
                # cls_pred = int(cls_pred)
                # cls_pred = 1
                # logger.info(cls_pred)
                # cls_pred = min(len_cls - 1, cls_pred)
                detection[0] = max(detection[0], 0)
                detection[1] = max(detection[1], 0)
                detection[2] = max(detection[2], 0)
                detection[3] = max(detection[3], 0)
                obj = self.yolo.class_names[cls_id]

                if obj == 'person':
                    detect1dict['行人'].append(detection)
                elif obj == 'cell phone':
                    detect1dict['手机'].append(detection)
                elif obj == 'dog':
                    detect1dict['狗'].append(detection)
                elif obj == 'car':
                    detect1dict['汽车'].append(detection)
                elif obj == 'truck':
                    detect1dict['卡车'].append(detection)

            toc = time.time()
            if SHOW_TIME:
                logger.info('pedestrain detection time:' + str(toc - tic))

            self.wait_for_queue(1)
            self.Q.put(
                (update_type, im, img_tensor, im_name, ann_task_list, detectTempImageDir, detect1dict, feats, detect_mode, queue_data))

        self.wait_for_stop(1)
        self.stopped = True

class PoseEstimator(DetectionStage):
    def __init__(self,pre_queue,next_queue, queue_size=QUEUE_SIZE):
        super(PoseEstimator, self).__init__(self.pose_estimating, pre_queue,next_queue,queue_size)
        self.modelsManager = ModelsManager()
        self.modelsManager.loadPosEstimator()

    def pose_estimating(self):
        while (1):
            if self.is_empty():
                continue
            try:
                update_type, im, img_tensor, im_name, ann_task_list, detectTempImageDir, detect1dict, feats, detect_mode, queue_data = \
                    self.get_next(timeout=10)
            except Exception:
                continue
            tic = time.time()
            #with torch.no_grad():
            self.modelsManager.process_pos_detection(detect1dict, img_tensor, 0)  # pos_detection

            toc = time.time()
            if SHOW_TIME:
                logger.info('pose estimating time:' + str(toc - tic))
            pos_map = self.modelsManager.pos_map
            self.wait_for_queue(1)
            self.Q.put(
                (update_type, im, im_name, ann_task_list, detectTempImageDir, detect1dict, feats, pos_map, detect_mode, queue_data))

        self.wait_for_stop(1)
        self.stopped = True


class DetectionProcedure(DetectionStage):
    def __init__(self,pre_queue,next_queue, queue_size=QUEUE_SIZE):
        super(DetectionProcedure, self).__init__(self.im_detection,pre_queue,next_queue, queue_size)
        self.modelsManager = ModelsManager()
        self.modelsManager.loadDetection2(self.detection2TaskListPath, self.sndTaskDetectThresPath)
        self.modelsManager.loadDetection3(self.detection3TaskListPath, self.trdTaskDetectThresPath)

        self.modelsManager.loadFaceDetection()
        self.modelsManager.loadEnsTaskMap(self.ensembleTaskPath)

    def im_detection(self):
        while (1):
            #if self.prev.stopped:
            #    break
            if self.is_empty():
                continue
            update_type, im, im_name, ann_task_list, detectTempImageDir, detect1dict,detect_mode, queue_data = self.get_next(
                timeout=10)
            '''
            try:
                update_type, im, im_name, ann_task_list, detectTempImageDir, detect1dict, feats, pos_map, detect_mode = self.get_next(
                    timeout=10)
            except Exception:
                continue
            '''
            if update_type == '2':  # update thresholds

                self.modelsManager.loadDetection2(self.detection2TaskListPath, self.sndTaskDetectThresPath)
                self.modelsManager.loadDetection3(self.detection3TaskListPath, self.trdTaskDetectThresPath)
            else:
                self.modelsManager.setCurImINf(im_name)  # for moving judgement

                tic = time.time()

                detectResult, feats = self.modelsManager.snd_trd_recognition(im, ann_task_list, detect1dict)
                '''
                try:
                    logger.info('')
                    # detectResult, feats = self.modelsManager.snd_trd_recognition(im, ann_task_list , detect1dict, feats, pos_map)
                except:
                    logger.info("invalid snd+ layer detection im_name:" + im_name)
                    continue
                '''
                self.wait_for_queue(1)
                self.Q.put((update_type, detectResult, im, im_name, detectTempImageDir, detect_mode, queue_data))

                self.modelsManager.updateImPos(detectResult)
                self.modelsManager.update()
                toc = time.time()
                if SHOW_TIME:
                    logger.info('other layers times:' + str(toc - tic))
        self.wait_for_stop(1)
        self.stopped = True


class PostProcess(DetectionStage):
    def __init__(self, pre_queue,next_queue, batch_size=1, queue_size=QUEUE_SIZE):
        super(PostProcess, self).__init__(self.post_process, pre_queue,next_queue,queue_size)
        self.imBoxStoreDict = dict()
        self.imLingerConfig = self.loadGroupLingerInf(self.deviceGroupTaskPath, self.groupTaskPath)

    def post_process(self):
        while (1):
            #if self.prev.stopped:
            #    break
            if self.is_empty():
                continue
            update_type, resultItems, feats, im, im_name, detectTempImageDir, detect_mode, queue_data = self.get_next(
                timeout=10)

            if update_type == '5':  # update face thres
                self.imLingerConfig = self.loadGroupLingerInf(self.deviceGroupTaskPath, self.groupTaskPath)
                continue
            tic = time.time()
            detectResult = resultItems
            flag = 0
            if detect_mode:
                self.lingerJudge(self.imLingerConfig, detectResult, im_name)  # ljudge

                flag = self.filterSameIm(im_name, detectResult)
            flagSaveDetect = 0

            if flag == 0:
                cur_im_name = im_name
                tx_name = im_name.split('.jp')[0] + '.txt'
                logger.info('im_name:'+cur_im_name)
                im_save_path = detectTempImageDir + cur_im_name + '.jpg'
                txt_save_path = detectTempImageDir + tx_name
                restr = ''
                for itm in detectResult:
                    if itm.flag == 3 or itm.flag == 7 or itm.flag == 5:
                        restr = restr + itm.name + ' ' + str(itm.flag) + '\t'
                    if itm.flag == 3:
                        flagSaveDetect = 1
                logger.info('detected results:' + restr)

                if flagSaveDetect == 1:

                    try:
                        im = np.copy(im[:, :, ::-1])
                        cv2.imwrite(im_save_path, im)

                        cv2.waitKey(20)  # to save image
                        self.saveDetectResult(detectResult, txt_save_path)
                    except:
                        logger.info("error in writing im")
                    # logger.info('save_path:' + im_save_path)
                    if True:
                        pass
                        #self.dispatch_data(detectResult, im_save_path, queue_data['rtmp_url'])
            else:
                logger.info('filter same detection')
            toc = time.time()
            if SHOW_TIME:
                logger.info('post process time:' + str(toc - tic))

    def dispatch_data(self, detectResult, im_path, rtmp_url):
        with open(im_path, 'rb') as f:
            im_byte = base64.b64encode(f.read())
            im_str = im_byte.decode('ascii')

        pos_list = []

        for it in detectResult:
            pos = dict()
            pos['type'] = it.name
            pos['flag'] = it.flag
            if it.name in self.detect_task_type:
                if self.detect_task_type[it.name] == 1:
                    continue
            pos_str = ''
            first_flag = True
            for box in it.list:
                pos_inf = str(box[0]) + ',' + str(box[1]) + ','+ str(box[2]) + ','+ str(box[3])
                if first_flag:
                    pos_str = pos_inf
                    first_flag = False
                else:
                    pos_str += '|' + pos_inf
            pos['pos'] = pos_str
            pos_list.append(pos)
        if len(pos_list) > 0:
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data = {'rtmpurl': rtmp_url, 'base64': im_str, 'data':pos_list, 'checkdata': time_str}
            json_mod = json.dumps(data).encode(encoding='utf-8')
            headers = {"content-Type": "application/json"}

            send_data = 'json=' + str(json_mod)
            res = requests.post(self.ai_camera_server_url, data=json_mod, headers = headers)
            logger.info(res.text)


    def loadGroupLingerInf(self, deviceGroupTaskPath, groupTaskPath):
        imLingerConfig = dict()
        groupTaskDict = dict()
        if os.path.exists(groupTaskPath):
            lines = open(groupTaskPath).readlines()
            for line in lines:
                line = line.strip().strip('\r').strip('\n')
                strs = line.split('\t')
                if len(strs) > 1:
                    groupName = strs[0]
                    taskList = [strs[i] for i in range(1, len(strs))]
                    groupTaskDict[groupName] = taskList
        if os.path.exists(deviceGroupTaskPath):
            lines = open(deviceGroupTaskPath).readlines()

            for line in lines:
                line = line.strip().strip('\r').strip('\n')
                strs = line.split('\t')
                len_strs = len(strs)
                if len_strs > 3:
                    ip = strs[0]
                    port = strs[1]
                    channelNum = strs[2]
                    for k in range(3, len_strs):
                        groupInfs = strs[k].split('!')
                        if len(groupInfs) > 6:
                            groupName = groupInfs[0]
                            lingerTime = int(groupInfs[6])
                            if groupName in groupTaskDict and lingerTime > 0:  # only keep the linger task
                                taskList = groupTaskDict[groupName]
                                lingerTasks = [lingerTask(taskList[i], lingerTime) for i in range(0, len(taskList))]
                                key = ip + '_' + port + '_' + channelNum
                                imLingerConfig[key] = lingerTasks
        return imLingerConfig

    def lingerJudge(self, imLingerConfig, detectResults, curImName):
        ip, port, channelNum, dateTime = self.parseImName(curImName)
        imKey = ip + '_' + port + '_' + channelNum
        c = list()

        if imKey in imLingerConfig:
            lingerTasks = imLingerConfig[imKey]
            for i in range(0, len(lingerTasks)):

                findFlag = False
                for item in detectResults:
                    if item.flag != 3:
                        continue
                    taskName = item.name
                    if lingerTasks[i].taskName == taskName:  # 如果找到，则比较时间，决定是否保留
                        findFlag = True
                        keepFlag = lingerTasks[i].compareTime(dateTime)
                        if keepFlag == -1:
                            detectResults.remove(item)
                            break
                if not findFlag:
                    lingerTasks[i].resetLingerState()

    def filterSameIm(self, curImName, detectResult):
        strs = curImName.split("_")
        curList = list()
        for it in detectResult:
            itList = it.list
            if len(itList) > 200:
                itList = itList[0:200]
            curList.extend(itList)
        if len(curList) < 1:
            return 0
        flags = [0 for z in range(0, len(curList))]

        if len(strs) >= 6:
            strIp = strs[1]
            strPort = strs[2]
            strChannel = strs[3]
            key = strIp + '_' + strPort + '_' + strChannel
            if key in self.imBoxStoreDict:
                preDetectResult = self.imBoxStoreDict[key]
                matchId = 1
                if len(curList) == len(preDetectResult):
                    for box in preDetectResult:
                        maxOverLapRatio = 0.0
                        maxOverLapInd = -1
                        for j in range(0, len(curList)):
                            if flags[j] == 1:
                                continue
                            overLap = self._overlapInterUnion(box, curList[j])
                            if maxOverLapRatio < overLap:
                                maxOverLapRatio = overLap
                                maxOverLapInd = j
                        if maxOverLapRatio > 0.5:
                            flags[maxOverLapInd] = 1
                            # matchId = 1
                        else:
                            matchId = 0
                if matchId == 0:
                    self.imBoxStoreDict[key] = curList
                    return 0
                self.imBoxStoreDict[key] = curList
                return 1
            else:
                self.imBoxStoreDict[key] = curList
                return 0
        return 0

    def _overlapInterUnion(self, box1, box2):
        if box1.any() == None or len(box1) < 1 or box2.any() == None or len(box2) < 1:
            return 0
        # print len(box1)
        # print len(box2)
        new_x1 = float(max(box1[0], box2[0]))
        new_x2 = float(min(box1[2], box2[2]))
        new_y1 = float(max(box1[1], box2[1]))
        new_y2 = float(min(box1[3], box2[3]))

        union_x1 = float(min(box1[0], box2[0]))
        union_x2 = float(max(box1[2], box2[2]))
        union_y1 = float(min(box1[1], box2[1]))
        union_y2 = float(max(box1[3], box2[3]))

        union_area = (union_x2 - union_x1) * (union_y2 - union_y1)
        if (new_x1 >= new_x2 or new_y1 >= new_y2):
            return 0
        over_lap = (new_x2 - new_x1) * (new_y2 - new_y1)
        if union_area <= 0:
            return 0
        return over_lap / union_area

    def saveDetectResult(self, annTaskList, savePath):
        f = open(savePath, 'w')
        for it in annTaskList:
            f.write(it.name + '\t')
            f.write(str(it.flag))
            for box in it.list:
                f.write('\t')
                f.write(str(box[0]) + ' ')
                f.write(str(box[1]) + ' ')
                f.write(str(box[2]) + ' ')
                f.write(str(box[3]))
            f.write('\r\n')
        f.close()
