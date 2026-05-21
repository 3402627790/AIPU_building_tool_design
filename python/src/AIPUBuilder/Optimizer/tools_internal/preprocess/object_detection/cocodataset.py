# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
tf.enable_eager_execution()


def extract_valid_files(img_dir, filename_list):
    files_list = os.listdir(img_dir)
    files_name_list = [os.path.splitext(files)[0] for files in files_list]
    files_ext_list = [os.path.splitext(files)[1] for files in files_list]
    valid_files_list = []

    for file_name in filename_list:
        if file_name in files_list:
            valid_files_list.append(file_name)
        else:
            file_name_parts = os.path.splitext(file_name)
            file_full_name = file_name
            if len(file_name_parts[1]) >= 1:
                print(file_name + ' not exists!')
                continue
            elif len(file_name_parts[1]) == 0:
                file_name = file_name.split('.')[0]
                if file_name in files_name_list:
                    file_index = files_name_list.index(file_name)
                    file_ext = files_ext_list[file_index]
                    file_full_name = file_full_name + file_ext
                    valid_files_list.append(file_full_name)
                else:
                    print(file_name + ' not exists!')
                    continue

    return valid_files_list


def map_image_annt(image_list, annt_list):
    image_annt_list = []
    image_name_list = [os.path.splitext(image_name)[0] for image_name in image_list]
    annt_name_list = [os.path.splitext(annt_name)[0] for annt_name in annt_list]

    for img_idx, image_name in enumerate(image_name_list):
        if image_name in annt_name_list:
            annt_idx = annt_name_list.index(image_name)
            image_annt_list.append([image_list[img_idx], annt_list[annt_idx]])
        else:
            print(image_list[img_idx] + ' have no annotation file.')
            continue
    return image_annt_list


class COCO():
    LABELS = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
        9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
        25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
        35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
        41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
        49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
        57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
        65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
        77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
    }
    CLASS_NUM = len(list(LABELS.keys()))

    @staticmethod
    def get_label_name(label_id):
        assert label_id in range(0, COCO.CLASS_NUM)
        return COCO.LABELS[label_id]

    @staticmethod
    def get_label_id(label_name):
        assert label_name in COCO.LABELS.values()
        for (id, name) in COCO.LABELS.items():
            if name == label_name:
                return id


def get_annotation_info(xml_file, dataset='coco'):
    if not os.path.exists(xml_file):
        print(xml_file + ' not exists!')
        return

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # image size
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    channel = int(size.find('depth').text)

    # all the bounding boxes and class labels in the image
    labels_name = []
    labels_index = []
    bbox_xmin = []
    bbox_xmax = []
    bbox_ymin = []
    bbox_ymax = []
    # bboxes = []
    for obj in root.findall('object'):
        # if int(obj.find('difficult').text) == 0:
        # labels_index.append(int(obj.find('label_index').text))
        if obj.find('label_index') is not None:
            labels_index.append(int(obj.find('label_index').text))
        elif obj.find('name') is not None:  # coco name to id or voc name to id
            if dataset == 'coco':
                labels_index.append(COCO.get_label_id(obj.find('name').text))
            # elif dataset == 'voc':
            #     labels_index.append(VOC.get_label_id(obj.find('name').text))

        bbox = obj.find('bndbox')
        bbox_ymin.append(float(bbox.find('ymin').text) / height)
        bbox_ymax.append(float(bbox.find('ymax').text) / height)
        bbox_xmin.append(float(bbox.find('xmin').text) / width)
        bbox_xmax.append(float(bbox.find('xmax').text) / width)

    return height, width, channel, labels_index, labels_name, bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax


preprocess_param = {
    # in_h, in_w, in_c, mean, var, max, min
    'yolo': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'yolo_v3': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'caffe_yolo': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'tf_yolov3': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'caffe_yolo3': [608, 608, 3, 0.0, 256.0, 1.0, 0.0],
    'fasterrcnn': [600, 600, 3, [123.68, 116.779, 103.939], 1, 151, -123],
    'onnx_yolov4': [416, 416, 3, 0.0, 255, 1.0, 0.0],
    'tflite_yolov4': [512, 512, 3, 0.0, 255, 1.0, 0.0],
    'yolov2_onnx': [416, 416, 3, 0.0, 255, 1.0, 0.0],
    'yolov5_onnx': [640, 640, 3, 0.0, 255, 1.0, 0.0],
    'retinanet': [256, 512, 3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1.0, 0.0],
    'fcos': [800, 1216, 3, [102.9801, 115.9465, 122.7717], 1, 1.0, 0.0],
}


def resize_image(image, height, width, method='BILINEAR'):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
        image: A 3-D image `Tensor`.
        height: The target height for the resized image.
        width: The target width for the resized image.

    Returns:
        resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    resize_func = tf.image.ResizeMethod.NEAREST_NEIGHBOR if method == 'NEAREST' else tf.image.ResizeMethod.BILINEAR
    return tf.image.resize_images(image, [height, width], method=resize_func, align_corners=False)


def det_preprocess(model_name, data_num, img_dir, annotation_dir, datalist, data_path, label_path, mode='nhwc', batchnorm=True):
    if model_name in preprocess_param:
        param = preprocess_param[model_name]
        # the input height and width of the model.
        input_height = param[0]
        input_width = param[1]
        input_channel = param[2]
        # the mean and variance of the input images.
        mean = param[3]
        var = param[4]
    else:
        print("Not supported model: ", model_name)
        return
    filename_list = [os.path.splitext(line.rstrip('\n'))[0] for line in open(datalist)]
    img_num = len(filename_list)

    image_list = extract_valid_files(img_dir, filename_list)
    annt_list = extract_valid_files(annotation_dir, filename_list)
    image_annt_list = map_image_annt(image_list, annt_list)

    # nhwc
    images = np.zeros([data_num, input_height, input_width, input_channel], np.float32)
    labels = {}

    origin_coco_labels = list(range(91))
    coco_80_missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
    coco_80_label_map = {}
    for l in origin_coco_labels:
        if model_name == 'yolov2_onnx':
            l_original = l
            if l >= 1 and l <= 11:
                l = l - 1
            elif l >= 13 and l <= 25:
                l = l - 2
            elif l >= 27 and l <= 28:
                l = l - 3
            elif l >= 31 and l <= 44:
                l = l - 5
            elif l >= 46 and l <= 65:
                l = l - 6
            elif l == 67:
                l = l - 7
            elif l == 70:
                l = l - 9
            elif l >= 72 and l <= 82:
                l = l - 10
            elif l >= 84 and l <= 90:
                l = l - 11
            coco_80_label_map.update({l_original: l})

        else:
            l_offset = np.sum(l > np.array(coco_80_missing_labels, np.int64))
            coco_80_label_map.update({l: l - l_offset})

    for image_annt, img_idx in zip(image_annt_list, range(img_num)):
        image_name, annt_name = image_annt
        img_name = img_dir + image_name

        if not os.path.exists(img_name):
            print(img_name + ' not exists!')
            continue
        # if PROCESS_OPTION == PROCESS_MODE.TENSORFLOW:
        img_stream = tf.gfile.GFile(img_name, 'rb').read()
        # img_stream = tf.gfile.GFile('/home/matche01/000000000030.jpg', 'rb').read()
        image = tf.image.decode_jpeg(img_stream)
        image = tf.cast(image, tf.float32)
        image = tf.clip_by_value(image, 0., 255.)
        image /= 255
        image = resize_image(image, input_height, input_width)
        if batchnorm == True:
            image = (image - mean) / var
        image = image.numpy()
        _, _, ch = image.shape
        if ch == 1:
            image = tf.tile(image, multiples=[1, 1, 3])
            image = image.numpy()

        annotation_name = annotation_dir + annt_name
        height, width, channel, labels_index, labels_name,\
            bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = get_annotation_info(annotation_name)
        # !! different label_idx for different yolo series
        if model_name in ['caffe_yolo', 'onnx_ssd', 'onnx_yolo2_coco', 'yolo_v3', 'caffe_yolo3', 'onnx_yolov4', 'tflite_yolov4', 'yolov2_onnx', 'yolov5_onnx', 'retinanet', 'tf_yolov3']:
            labels_index = [coco_80_label_map[l] - 1 if model_name in ['onnx_yolo2_coco', 'caffe_yolo3', 'yolo_v3', 'onnx_yolov4', 'tflite_yolov4', 'yolov5_onnx', 'retinanet', 'tf_yolov3'] else coco_80_label_map[l]
                            for l in labels_index]

        if model_name in ['fasterrcnn']:
            bbox_xmax = [box * input_width for box in bbox_xmax]
            bbox_ymax = [box * input_height for box in bbox_ymax]
            bbox_xmin = [box * input_width for box in bbox_xmin]
            bbox_ymin = [box * input_height for box in bbox_ymin]

        labels.update({img_idx: [labels_index, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax]})

        images[img_idx] = image

        if img_idx >= data_num-1:
            break

    if mode == 'nchw':
        images = images.transpose((0, 3, 1, 2))

    np.save(data_path, images*256)
    np.save(label_path, labels)


def process_single(model_name, mode='nhwc', norm=True, data_num=100, output_dir='./'):
    if data_num > 5000 or data_num < 1:
        print('Error: datanum max is 5000, min is 1, now is ', data_num)
        return
    dataset_dir = "/project/ai/scratch01/ruolu01/share/toShengwei/coco_5000/"
    img_dir = dataset_dir + "images/"
    filelist_txt = dataset_dir + 'label.txt'
    annotation_dir = dataset_dir + 'annotations/'

    image_model_dir = os.path.join(output_dir, 'preprocess', model_name)
    if not os.path.exists(image_model_dir):
        os.makedirs(image_model_dir)
    data_path = os.path.join(image_model_dir, 'dataset' + str(data_num) + '.npy')
    label_path = os.path.join(image_model_dir, 'label' + str(data_num) + '.npy')
    det_preprocess(model_name, data_num, img_dir, annotation_dir, filelist_txt, data_path, label_path, mode, norm)

# main
# name # data_format # normalize #data_num #output_dir
# name: support 'caffe_yolo','onnx_ssd', 'onnx_yolo2_coco','yolo_v3','caffe_yolo3'
# dataset format: default nhwc, support nhwc/nchw
# normalize: default True, image = (image - mean) / var when setting normalize = True
# data_num: default is 100, range is 1-5000
#output_dir: default is './'


# process_single('yolo_v3', 'nchw', False)
# process_single('onnx_yolov4', 'nhwc', True, './')
# process_single('tflite_yolov4', 'nhwc', True, 1000)
# process_single('centerpose', 'nchw', True, 100)
# process_single('yolov2_onnx', 'nchw', True, 1000)
# process_single('yolov5_onnx', 'nchw', True, 100)
# process_single('tf_yolov3', 'nhwc', True, 100, './')
# process_single('retinanet', 'nhwc', True, 100)
process_single('fcos', 'nchw', True, 3)
