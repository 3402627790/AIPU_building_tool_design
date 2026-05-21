# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys

input_shape = [416, 416]
save_path_pb = '/project/ai/scratch01/samhan01/work/test/models/yolo_v2_tiny_onnx'

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


class VOCDataset(object):

    @staticmethod
    def get_label_id(label_name):
        if label_name in VOC_LABELS:
            return VOC_LABELS[label_name][0]
        else:
            print(f"label_name = {label_name} is not a VOC dataset label name [{VOC_LABELS.keys()}].")
            return None

    # @staticmethod
    # def get_label_name(label_id):
    #     if label_id in COCO_NAME_ID_LABELS.keys():
    #         return COCO_NAME_ID_LABELS[label_id]
    #     else:
    #         print(f"label_id = {label_id} is not a COCO dataset label id.")
    #         return None

    @staticmethod
    def get_annotation_info(xml_file):
        if not os.path.exists(xml_file):
            print(f"{xml_file} not exists!")
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
            l_idx = None
            l_name = None
            if obj.find('label_index') is not None:
                l_idx = int(obj.find('label_index').text)
                # l_name = VOCDataset.get_label_name(l_idx)

            if obj.find('name') is not None:  # coco name to id or voc name to id
                l_name = obj.find('name').text
                l_idx = VOCDataset.get_label_id(l_name)

            if l_idx is None or l_name is None:
                print(f"l_idx={l_idx} or l_name = {l_name} is None.")
            labels_index.append(l_idx)
            # labels_name.append(l_name)

            bbox = obj.find('bndbox')
            bbox_ymin.append(float(bbox.find('ymin').text) / height)
            bbox_ymax.append(float(bbox.find('ymax').text) / height)
            bbox_xmin.append(float(bbox.find('xmin').text) / width)
            bbox_xmax.append(float(bbox.find('xmax').text) / width)

        return height, width, channel, labels_index, labels_name, bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax

    @staticmethod
    def get_data(img_dir):
        image = Image.open(img_dir)
        image = cvtColor(image)
        image_data = resize_image(image, input_shape, False)
        # processed_data = preprocess_input(image_data)
        # yolov2_tiny_onnx preprocess has existed in CompassIR.
        return image_data


def check_and_get_jpg_with_xml(img_list, annotation_list):
    img_xml_list = []
    for img in img_list:
        xml = img.split('.')[0] + '.xml'
        if xml not in annotation_list:
            print(f"xml={xml} not in annotation list")
            continue
        img_xml_list.append([img, xml])
    return img_xml_list


def yolov2_tiny_onnx_preprocess(save_path_bp, num, is_test=True):

    dataset_dir = "/project/ai/scratch01/Dataset/voc/VOCdevkit/VOC2007/"
    img_dir = os.path.join(dataset_dir, "JPEGImages")
    test_label_txt_dir = os.path.join(dataset_dir, 'ImageSets/Main/test.txt')
    val_label_txt_dir = os.path.join(dataset_dir, 'ImageSets/Main/val.txt')
    annotation_dir = os.path.join(dataset_dir, 'Annotations/')

    target_txt_dir = test_label_txt_dir if is_test else val_label_txt_dir
    img_name_list = []
    with open(target_txt_dir, 'r') as fr:
        lines = fr.readlines()
        for l in lines:
            img_name_list.append(l.strip('\n'))

    img_name_list = [img_name + '.jpg' for img_name in img_name_list if '.jpg' not in img_name]
    img_num = len(img_name_list)
    print(f"img_num = {img_num}")
    if num > img_num or num == 0:
        num = img_num

    annotation_list = os.listdir(annotation_dir)

    img_xml_list = check_and_get_jpg_with_xml(img_name_list, annotation_list)
    if img_xml_list is None:
        print(f'failed')

    images = np.zeros([num, *input_shape, 3])
    labels = {}
    labels_1 = {}

    for i, img_xml in enumerate(img_xml_list):
        img_name, xml_name = img_xml

        img_pth = os.path.join(img_dir, img_name)
        xml_dir = os.path.join(annotation_dir, xml_name)

        image_data = VOCDataset.get_data(img_pth)
        xml_infos = VOCDataset.get_annotation_info(xml_dir)

        height = xml_infos[0]
        width = xml_infos[1]
        channel = xml_infos[2]
        label_index = xml_infos[3]
        bbox_xmin = xml_infos[5]
        bbox_xmax = xml_infos[6]
        bbox_ymin = xml_infos[7]
        bbox_ymax = xml_infos[8]

        labels.update({i: [label_index, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax]})

        label_index_1 = [li - 1 for li in label_index]
        labels_1.update({i: [label_index_1, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax]})
        images[i] = image_data

        if i + 1 == num:
            break

    np.save(os.path.join(save_path_bp, 'dataset_' + str(num)+'.npy'), images)
    np.save(os.path.join(save_path_bp, 'label_' + str(num)+'.npy'), labels)
    np.save(os.path.join(save_path_bp, 'label_-1_' + str(num)+'.npy'), labels_1)
    print('********Preprocess %s dataset Done********' % (str(num)))


def main(argv):
    dataset_num_list = [1, 100, 1000, 0]
    for n in dataset_num_list:
        if n <= 1000 and n != 0:
            # for optimizer calibration_data
            is_test = False
        else:
            is_test = True
        yolov2_tiny_onnx_preprocess(save_path_pb, n, is_test=is_test)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
