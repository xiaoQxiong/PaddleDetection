# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from ppdet.data.transform.operators import DecodeImage, NormalizeImage, Permute,ResizeImage

import os
import glob
import cv2
import numpy as np
from PIL import Image
import six
from six.moves import zip, range, xrange

def set_paddle_flags(**kwargs):
    for key, value in kwargs.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


# NOTE(paddle-dev): All of these flags should be set before
# `import paddle`. Otherwise, it would not take any effect.
set_paddle_flags(
    FLAGS_eager_delete_tensor_gb=0,  # enable GC to save memory
)

from paddle import fluid
from paddle.fluid import core
from ppdet.core.workspace import load_config, merge_config, create

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint

from ppdet.data.reader import create_reader,batch_arrange
from paddle.fluid.data_feeder import DataFeeder, BatchedTensorProvider,DataToLoDTensorConverter

from paddle import fluid
from paddle.fluid.reader import  _convert_places
from paddle.fluid import default_main_program,Variable
from ppdet.modeling import (MaskRCNN, ResNet, ResNetC5, RPNHead, RoIAlign,
                            BBoxHead, MaskHead, BBoxAssigner, MaskAssigner)
from ppdet.utils.coco_eval import bbox2out, mask2out, coco17_category_info
from ppdet.utils.visualizer import visualize_results
import time

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
'''
voc_clses = ['人头部','人手','人脚','人脸','伞','入门匝机杆','入门匝机道','入门闸机',
             '凳子','卡车','土面','在建建筑','在建建筑作业面','基坑','塔吊','塔吊挂钩','头肩','安全帽','安全网',
             '小汽车','建材1','戴安全帽头部','手机','护栏','挖掘机','摩托车','未干的水泥','未戴安全帽二级',
             '未戴安全帽头部','树','水管','水面','没穿上衣','火或烟','灯','盖网','石块','红色安全帽',
             '脚手架','自行车','蓝色安全帽','行人','通告牌','重物','钢筋堆','雪','香烟','马甲','黄色安全帽','下颚带',
             '拖鞋','料斗','料斗未超载','未挂安全网','桶','水','汽车','混凝土搅拌车','狗','白色安全帽','砖块儿']
'''
voc_clses = ['rentoubu','renshou','renjiao','renlian','san','rumenzhajigan','rumenzhajidao','rumenzhaji',
             'dengzi','kache','tumian','zaijianjianzhu','zaijianjianzhuzuoyemian','jikeng','tadiao','tadiaoguagou','toujian','anquanmao','anquanwang',
             'xiaoqiche','jiancai','daianquanmaotoubu','shouji','huzhao','wajueji','motuoche','weigandeshuini','weidaianquanmaoerji',
             'weidaianquanmaotoubu','shu','shuiguan','shuimian','mei chuan shang yi','huo huo yan','deng','gai wang','shikuai','hongse an quan mao',
             'jiao shou jia','zi xing che ','lan se an quan mao ','xingren','tong gao pai','zhong wu ','gang jin  dui','xue','xiangyan','ma jia','huangse an quan mao','xia e dai',
             'tuo xie','liao dou','liao dou chao zhai','wei gua an quan wang','tong','shui','qi che','hun ning tu jiao ban che','gou','baise an quan mao','zhuan kuai er']

def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
    images = []

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        images.append(infer_img)
        return images

    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.extend(glob.glob('{}/*.{}'.format(infer_dir, ext)))

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


class Yolov3Detection():
    def __init__(self):
        parser = ArgsParser()
        parser.add_argument(
            "--infer_dir",
            type=str,
            default=None,
            help="Directory for images to perform inference on.")

        parser.add_argument(
            "--config",
            type=str,
            default='configs/dcn/zzj_1.yml',
            help="configs")

        parser.add_argument(
            "--infer_img",
            type=str,
            default='./demo/000000570688.jpg',
            help="Image path, has higher priority over --infer_dir")
        parser.add_argument(
            "--output_dir",
            type=str,
            default="output",
            help="Directory for storing the output visualization files.")
        parser.add_argument(
            "--draw_threshold",
            type=float,
            default=0.5,
            help="Threshold to reserve the result for visualization.")
        parser.add_argument(
            "--use_tb",
            type=bool,
            default=False,
            help="whether to record the data to Tensorboard.")
        parser.add_argument(
            '--tb_log_dir',
            type=str,
            default="tb_log_dir/image",
            help='Tensorboard logging directory for image.')
        FLAGS = parser.parse_args()

        cfg = load_config(FLAGS.config)

        if 'architecture' in cfg:
            main_arch = cfg.architecture
        else:
            raise ValueError("'architecture' not specified in config file.")

        merge_config(FLAGS.opt)

        # check if set use_gpu=True in paddlepaddle cpu version
        check_gpu(cfg.use_gpu)
        # check if paddlepaddle version is satisfied
        check_version()

        dataset = cfg.TestReader['dataset']
        # 仅读取文件路径
        test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
        dataset.set_images(test_images)

        scope = fluid.Scope()
        fluid.scope_guard(scope=scope)
        place =  fluid.CUDAPlace(0)



        exe = fluid.Executor(place)

        #[infer_prog, _, fetch_targest] = fluid.io.load_inference_model(dirname=cfg.weights, executor=exe)


        model = create(main_arch)

        startup_prog = fluid.Program()
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                inputs_def = cfg['TestReader']['inputs_def']
                inputs_def['iterable'] = True
                feed_vars, loader = model.build_inputs(**inputs_def)

                test_fetches = model.test(feed_vars)
        infer_prog = infer_prog.clone(True)

        # reader = create_reader(cfg.TestReader)
        # loader.set_sample_list_generator(reader, place)

        exe.run(startup_prog)
        if cfg.weights:
            checkpoint.load_params(exe, infer_prog, cfg.weights)

        # parse infer fetches
        assert cfg.metric in ['COCO', 'VOC', 'OID', 'WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)
        extra_keys = []
        if cfg['metric'] in ['COCO', 'OID']:
            extra_keys = ['im_info', 'im_id', 'im_shape']
        if cfg['metric'] == 'VOC' or cfg['metric'] == 'WIDERFACE':
            extra_keys = ['im_id', 'im_shape']
        keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)

        # parse dataset category
        if cfg.metric == 'COCO':
            from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
        if cfg.metric == 'OID':
            from ppdet.utils.oid_eval import bbox2out, get_category_info
        if cfg.metric == "VOC":
            from ppdet.utils.voc_eval import bbox2out, get_category_info
        if cfg.metric == "WIDERFACE":
            from ppdet.utils.widerface_eval_utils import bbox2out, get_category_info

        anno_file = dataset.get_anno()
        with_background = dataset.with_background
        use_default_label = dataset.use_default_label

        clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                    use_default_label)

        clsid2catid = {}
        for i in  range(0,61):
            clsid2catid[i] = i + 1

        catid2name = {}
        catid2name[0] = 'background'
        for iind, cat in enumerate(voc_clses):
            catid2name[iind + 1] = cat

        # whether output bbox is normalized in model output layer
        is_bbox_normalized = False
        if hasattr(model, 'is_bbox_normalized') and \
                callable(model.is_bbox_normalized):
            is_bbox_normalized = model.is_bbox_normalized()

        # use tb-paddle to log image
        if FLAGS.use_tb:
            from tb_paddle import SummaryWriter
            tb_writer = SummaryWriter(FLAGS.tb_log_dir)
            self.tb_image_step = 0
            self.tb_image_frame = 0  # each frame can display ten pictures at
            self.tb_writer = tb_writer
            #self.tb_image_frame = tb_image_frame

        self.is_bbox_normalized = is_bbox_normalized
        self.clsid2catid = clsid2catid
        self.catid2name = catid2name

        self.decode = DecodeImage(to_rgb=True, with_mixup=False)
        self.resize_im = ResizeImage(target_size=608, interp=2)
        self.normalize = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=False,
            is_channel_first=False)
        self.permute = Permute(to_bgr=False, channel_first=True)

        self.fields = copy.deepcopy(inputs_def[
                                   'fields']) if inputs_def else None

        feed_list = list(feed_vars.values())
        feed_dtypes = []
        feed_names = []
        feed_shapes = []
        feed_lod_level = []
        program = None
        if program is None:
            program = default_main_program()
        for each_var in feed_list:
            if isinstance(each_var, six.string_types):
                each_var = program.block(0).var(each_var)
            if not isinstance(each_var, Variable):
                raise TypeError("Feed list should contain a list of variable")
            feed_dtypes.append(each_var.dtype)
            feed_names.append(each_var.name)
            feed_lod_level.append(each_var.lod_level)
            feed_shapes.append(each_var.shape)

        converter = []
        for lod_level, shape, dtype in six.moves.zip(
                feed_lod_level, feed_shapes, feed_dtypes):
            converter.append(
                DataToLoDTensorConverter(
                    place=fluid.CPUPlace(),
                    lod_level=lod_level,
                    shape=shape,
                    dtype=dtype))

        self.converter = converter
        _places = _convert_places(place)

        self.dataset = dataset
        self.feed_names = feed_names
        self.exe = exe
        self.infer_prog = infer_prog
        self.values = values
        self.keys = keys

    def decode_im(self,frame, sample):
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        im = np.array(im)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)
        return sample
    def single_image_tensor_detect(self,im,im_shape):
        #imid2path = self.dataset.get_imid2path()
        # end new added
        batch = []
        time_1 = time.time()
        #sample = {'im_file': imid2path[0]}
        roidbs = self.dataset.get_roidb()
        # pos = self.indexes[self._pos]
        sample = copy.deepcopy(roidbs[0])
        batch.append(sample)
        time_2 = time.time()
        batch_new = []

        for sample in batch:
            sample = self.permute(self.normalize(self.resize_im(self.decode_im(im,sample))))

            batch_new.append(sample)
        time_3 = time.time()
        print('preprocess time : ' + str(time_3 - time_2))
        if len(batch) > 0 and self.fields:
            batch = batch_arrange(batch_new, self.fields)
        time_4 = time.time()
        print('batch arrange time : ' + str(time_4 - time_3))
        for each_sample in batch:
            # each_sample =  batch
            assert len(each_sample) == len(self.converter), (
                                                           "The number of fields in data (%d) does not match " +
                                                           "len(feed_list) (%d)") % (len(each_sample), len(self.converter))
            for each_converter, each_slot in six.moves.zip(self.converter,
                                                           each_sample):
                each_converter.feed(each_slot)
        ret_dict = {}
        for each_name, each_converter in six.moves.zip(self.feed_names,
                                                       self.converter):
            ret_dict[each_name] = each_converter.done()
        time_5 = time.time()
        print('converter time : ' + str(time_5 - time_4))
        # slots = [ret_dict[var.name] for var in feed_list]
        # array = core.LoDTensorArray()
        # array.append(ret_dict)
        # array.append(slots)

        outs = self.exe.run(self.infer_prog,
                       feed=ret_dict,
                       fetch_list=self.values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(self.keys, outs)
            }
        time_6 = time.time()
        print('pure det time : ' + str(time_6-time_5))

        #logger.info('Infer iter {}'.format(0))

        bbox_results = None
        #mask_results = None
        if 'bbox' in res:
            bbox_results = bbox2out([res], self.clsid2catid, self.is_bbox_normalized)
        # visualize result
        return res, bbox_results

    def video_detection(self):
        cap = cv2.VideoCapture("rtmp://58.200.131.2:1935/livetv/hunantv")
        count = 0
        while count < 100000:
            count += 1
            if cap.isOpened():
                ret, frame = cap.read()
                time.sleep(0.01)
                if not ret or frame is None:
                    continue
                tic = time.time()
                res, bbox_results = self.single_image_tensor_detect(frame, None)

                #print(res)
                det_tim = time.time() - tic
                print('det time elapse:' + str(det_tim))
                im_ids = ['0']
                for im_id in im_ids:
                    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


                    image = visualize_results(im,
                                              int(im_id), self.catid2name,
                                              FLAGS.draw_threshold, bbox_results,
                                              None)
                    image = np.array(image)
                    image = np.copy(image[:, :, ::-1])
                    cv2.imshow('rtmp test',image)
                    cv2.waitKey(10)
                toc = time.time()
                tm = toc - tic
                print('time elapse:' + str(tm))

                    # use tb-paddle to log image with bbox
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgsParser()

    parser.add_argument("-c", "--config", default='configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml', help="configuration file to use")

    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    '''
    parser.add_argument(
        "--config",
        type=str,
        default='configs/dcn/yolov3_r50vd_dcn_db_iouloss_obj365_pretrained_coco.yml',
        help="configs")
    '''
    parser.add_argument(
        "--infer_img",
        type=str,
        default='demo/000000570688.jpg',
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="infer_output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--use_tb",
        type=bool,
        default=False,
        help="whether to record the data to Tensorboard.")
    parser.add_argument(
        '--tb_log_dir',
        type=str,
        default="tb_log_dir/image",
        help='Tensorboard logging directory for image.')
    FLAGS = parser.parse_args()

    #yolo = Yolov3Detection()
    #yolo.load_im()
    #main()
    yolo = Yolov3Detection()
    yolo.video_detection()