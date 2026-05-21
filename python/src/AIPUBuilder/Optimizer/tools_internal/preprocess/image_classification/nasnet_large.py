# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from PIL import Image
import sys
import cv2
import numpy as np
import os

input_height = 331
input_width = 331
input_channel = 3

imagenet_bp = '/project/ai/scratch01/Dataset/ImageNet/'


def get_data(param):
    image_path = param['image_path']
    filename_list = param['filename_list']
    img_num = param['img_num']
    save_path = param['save_path']
    mean = param['mean']
    var = param['var']

    images = np.zeros([img_num, input_height, input_width, input_channel], np.float32)
    for file_name, idx in zip(filename_list, range(img_num)):
        image_file = os.path.join(image_path, file_name)
        img_s = cv2.imread(image_file)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        image = cv2.resize(img_s, (input_height, input_width))
        image = np.array(image)
        _, _, ch = image.shape
        if ch == 1:
            image = np.tile(image, [1, 1, 3])
        image = (image - mean) / var
        images[idx] = image
    np.save(save_path, images)


def get_label(param):
    label_file = param['label_file']
    label_save_path = param['label_save_path']
    num = param['label_num']
    label_data = open(label_file)
    filename_list = []
    label_list = []
    for line in label_data:
        filename_list.append(line.rstrip('\n').split(' ')[0])
        label_list.append(int(line.rstrip('\n').split(' ')[1]))
    label_data.close()
    labels = np.array(label_list)
    np.save(label_save_path, labels[:num])
    return filename_list


def preprocess(save_path_bp, num, mean, var):
    img_val = 'ILSVRC2012_img_val'
    lbl_val = 'val_label.txt'

    label_param = {
        'label_file': os.path.join(imagenet_bp, lbl_val),
        'label_save_path': os.path.join(save_path_bp, 'label_'+str(num)),
        'label_num': num,
    }
    filename_list = get_label(label_param)

    data_param = {
        'image_path': os.path.join(imagenet_bp, img_val),
        'filename_list': filename_list,
        'img_num': num,
        'save_path': os.path.join(save_path_bp, 'dataset_'+str(num)),
        'mean': mean,
        'var': var,
    }
    get_data(data_param)

    # print('label_path:', label_param['label_save_path'], 'dataset_path:', data_param['save_path'])
    print('********Preprocess %s dataset Done********' % (str(num)))


def main(argv):
    dataset_num_list = [1, 100, 200, 1000, 5000]
    if len(argv) > 1:
        save_path_bp = argv[1]  # '/project/ai/scratch01/samhan01/work/test/models/tf_efficientnet_b5/new_dataset'
    else:
        save_path_bp = './'
    MEAN_RGB = [127.5] * 3
    STDDEV_RGB = [127.5] * 3

    for n in dataset_num_list:
        preprocess(save_path_bp, n, MEAN_RGB, STDDEV_RGB)


if __name__ == '__main__':
    argv = sys.argv
    main(argv)
