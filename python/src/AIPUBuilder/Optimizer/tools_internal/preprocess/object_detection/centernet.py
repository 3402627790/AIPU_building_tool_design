# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import os
import sys

input_shape = [512, 512]
imagenet_bp = '/project/ai/scratch01/samhan01/work/test/models/centernet'

COCO_ID_NAME_LABELS = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
    87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}
COCO_NAME_ID_LABELS = dict(zip(COCO_ID_NAME_LABELS.values(), COCO_ID_NAME_LABELS.keys()))

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


def preprocess_input(image):
    '''centernet preprocess '''
    image = np.array(image, dtype=np.float32)[:, :, ::-1]
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std


class COCODataset(object):

    @staticmethod
    def get_label_id(label_name):
        if label_name in COCO_NAME_ID_LABELS.keys():
            return COCO_NAME_ID_LABELS[label_name]
        else:
            print(f"label_name = {label_name} is not a COCO dataset label name.")
            return None

    @staticmethod
    def get_label_name(label_id):
        if label_id in COCO_NAME_ID_LABELS.keys():
            return COCO_NAME_ID_LABELS[label_id]
        else:
            print(f"label_id = {label_id} is not a COCO dataset label id.")
            return None

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
                l_name = COCODataset.get_label_name(l_idx)

            if obj.find('name') is not None:  # coco name to id or voc name to id
                l_name = obj.find('name').text
                l_idx = COCODataset.get_label_id(l_name)

            if l_idx is None or l_name is None:
                print(f"l_idx={l_idx} or l_name = {l_name} is None.")
            labels_index.append(l_idx)
            labels_name.append(l_name)

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
        processed_data = preprocess_input(image_data)
        return processed_data


def check_and_get_jpg_with_xml(img_list, annotation_list):
    if len(img_list) != len(annotation_list):
        print(f"img num != xml num")
        return
    img_xml_list = []
    for img in img_list:
        xml = img.split('.')[0] + '.xml'
        if xml not in annotation_list:
            print(f"xml={xml} not in annotation list")
            continue
        img_xml_list.append([img, xml])
    return img_xml_list


def get_label_map():
    origin_coco_labels = list(range(91))
    coco_80_missing_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83]
    coco_80_label_map = {}
    for l in origin_coco_labels:
        l_offset = np.sum(l > np.array(coco_80_missing_labels, np.int64))
        coco_80_label_map.update({l: l - l_offset})
    return coco_80_label_map


def centernet_preprocess(save_path_bp, num):

    dataset_dir = "/project/ai/scratch01/Dataset/MSCOCO/preprocess/centernet/coco_5000/"
    img_dir = dataset_dir + "images/"
    label_txt_dir = dataset_dir + 'label.txt'
    annotation_dir = dataset_dir + 'annotations/'

    img_name_list = []
    with open(label_txt_dir, 'r') as fr:
        lines = fr.readlines()
        for l in lines:
            img_name_list.append(l.strip('\n'))

    img_num = len(img_name_list)
    print(f"img_num = {img_num}")
    if num > img_num or num == 0:
        num = img_num

    annotation_list = os.listdir(annotation_dir)

    img_xml_list = check_and_get_jpg_with_xml(img_name_list, annotation_list)
    if img_xml_list is None:
        print(f'failed')

    coco_80_label_map = get_label_map()

    images = np.zeros([num, *input_shape, 3])
    labels = {}
    labels_1 = {}

    for i, img_xml in enumerate(img_xml_list):
        img_name, xml_name = img_xml

        img_pth = os.path.join(img_dir, img_name)
        xml_dir = os.path.join(annotation_dir, xml_name)

        image_data = COCODataset.get_data(img_pth)
        xml_infos = COCODataset.get_annotation_info(xml_dir)

        height = xml_infos[0]
        width = xml_infos[1]
        channel = xml_infos[2]
        label_index = xml_infos[3]
        bbox_xmin = xml_infos[5]
        bbox_xmax = xml_infos[6]
        bbox_ymin = xml_infos[7]
        bbox_ymax = xml_infos[8]
        mapped_label = [coco_80_label_map[l] for l in label_index]
        mapped_label_1 = [coco_80_label_map[l] - 1 for l in label_index]

        labels.update({i: [mapped_label, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]})
        labels_1.update({i: [mapped_label_1, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax]})
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
        centernet_preprocess(imagenet_bp, n)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
