# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import numpy as np
import scipy
import cv2
import os
import tensorflow as tf
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


def interp_data(image, width, height):  # HWC to [900,900,3]
    label_margin = 186
    data_valid = True

    num_channels, input_height, input_width = 3, width, height

    image_size = image.shape
    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin
    image = cv2.copyMakeBorder(image, label_margin, label_margin, label_margin, label_margin, cv2.BORDER_REFLECT_101)

    num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)
    if num_tiles_h > 1 or num_tiles_w > 1:
        data_valid = False
        # return  # check image exceed 900*900, we skip them all for now

    margin = [0, input_height - image.shape[0], 0, input_width - image.shape[1]]
    image = cv2.copyMakeBorder(image, margin[0], margin[1], margin[2], margin[3], cv2.BORDER_REFLECT_101)

    return data_valid, image  # HWC


def interp_label(image, height, width, resize_method='BILINEAR'):  # output to image HWC
    # to square
    max_size = max(image.shape[0], image.shape[1])
    squared_image = np.zeros([max_size, max_size, image.shape[-1]], dtype=np.float32)
    max_h = max(max_size, image.shape[0])
    offset_h = (max_h - image.shape[0]) // 2
    max_w = max(max_size, image.shape[1])
    offset_w = (max_w - image.shape[1])//2
    squared_image[offset_h: offset_h + image.shape[0], offset_w:offset_w + image.shape[1], :] = image

    # resize
    resize_func = tf.image.ResizeMethod.NEAREST_NEIGHBOR if resize_method == 'NEAREST' else tf.image.ResizeMethod.BILINEAR
    zoom_image = tf.image.resize_images(squared_image, [height, width], method=resize_func, align_corners=False).numpy()
    return zoom_image


config = {  # in_h, in_w, in_c, mean, var, zoom, dilate
    'dilation_8': [900, 900, 3, 102.93, 111.36, 116.52, 1., 1., 1., 8, 8],

}
plette = {
    'dilation_8': [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                   [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [
                       64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                   [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [224, 224, 192]]
}


def process_dataset(model_name, mode='nhwc', data_num=100, output_dir='./'):
    if data_num > 5000 or data_num < 1:
        print('Error: datanum max is 5000, min is 1, now is ', data_num)
        return
    img_path = '/project/ai/scratch01/Dataset/voc/VOCdevkit/VOC2012/JPEGImages'
    segclass_path = '/project/ai/scratch01/Dataset/voc/VOCdevkit/VOC2012/SegmentationClass'
    segobj_path = '/project/ai/scratch01/Dataset/voc/VOCdevkit/VOC2012/SegmentationObject'
    imgs, shape, segclass, segobj = [], [], [], []

    image_model_dir = os.path.join(output_dir, 'preprocess', model_name)
    if not os.path.exists(image_model_dir):
        os.makedirs(image_model_dir)
    cnt = 0
    skip = 0
    for itm in os.listdir(img_path):
        if skip <= 100:
            skip += 1
            continue
        # check exist. if not, see u next time
        if not os.path.exists(os.path.join(segclass_path, itm.replace('jpg', 'png'))):
            # print(itm + ' not exists in segclass path!')
            continue

        if not os.path.exists(os.path.join(segobj_path, itm.replace('jpg', 'png'))):
            # print(itm + ' not exists in segobj path!')
            continue

        # image
        origin_image = image = cv2.imread(os.path.join(img_path, itm), 1).astype(np.float32)
        image -= config[model_name][3:6]
        image /= config[model_name][6:9]
        _, _, ch = image.shape
        img_height = config[model_name][0]
        img_width = config[model_name][1]
        if ch == 1:
            image = np.tile(image, 3)
        image = image[..., ::-1]
        res = interp_data(image, img_height, img_width)
        if res[0]:
            imgs.append(np.expand_dims(res[1], 0))
        else:  # shape invalid
            continue

        # segment class
        seg_class = cv2.imread(os.path.join(segclass_path, itm.replace('jpg', 'png')), 1)
        seg_class = seg_class[..., ::-1]
        seg_class_21 = np.zeros(list(seg_class.shape[:-1]) + [1])
        for idx in np.ndindex(seg_class.shape[:-1]):
            local_idx = plette['dilation_8'].index(list(seg_class[idx].astype(int)))
            if local_idx == 21:
                local_idx = 0
            seg_class_21[idx] = local_idx
        segclass.append(np.expand_dims(interp_label(seg_class_21, 500, 500, resize_method='NEAREST'), 0))

        # segment object
        # seg_obj = cv2.imread(os.path.join(segobj_path,itm.replace('jpg','png')), 1)
        # seg_obj = seg_obj[..., ::-1]
        # seg_obj_21 = np.zeros(list(seg_obj.shape[:-1]) + [1])
        # for idx in np.ndindex(seg_obj.shape[:-1]):
        #     local_idx = plette['dilation_8'].index(list(seg_obj[idx].astype(int)))
        #     seg_obj_21[idx]=local_idx
        # segobj.append(np.expand_dims(interp_label(seg_obj_21, img_height, img_width,resize_method='NEAREST'),0))

        shape.append(np.expand_dims(np.array(origin_image.shape), 0))

        print('datanum=%d/%d, filenam=%s, done.' % (cnt+1, data_num, itm))

        cnt += 1
        if cnt == data_num:
            break
    imgs = np.concatenate(imgs, 0)
    segclass = np.concatenate(segclass, 0)
    # segobj = np.concatenate(segobj,0)
    shape = np.concatenate(shape, 0)

    np.save(os.path.join(image_model_dir, 'inputs'), {'0': imgs})
    np.save(os.path.join(image_model_dir, 'segclass'), {'0': segclass, '1': shape})
    # np.save(os.path.join(image_model_dir,'segobj'),{'0':segobj,'1':shape})


process_dataset('dilation_8', data_num=200)
