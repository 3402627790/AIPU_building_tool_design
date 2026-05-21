# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

import cv2
import zipfile
import numpy as np
import os

from matlab_cp2tform import get_similarity_transform_for_cv2


lfw_pth = '/project/ai/scratch01/Dataset/LFW/preprocess/sphereface'

lfw_zip_pth = os.path.join(lfw_pth, 'lfw.zip')
pair_pth = os.path.join(lfw_pth, 'pairs.txt')
landmark_pth = os.path.join(lfw_pth, 'lfw_landmark.txt')


def alignment(src_img, src_pts):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def get_zip_landmark_pair():

    zfile = zipfile.ZipFile(lfw_zip_pth)

    landmark = {}
    with open(landmark_pth) as f:
        landmark_lines = f.readlines()
    for line in landmark_lines:
        l = line.replace('\n', '').split('\t')
        landmark[l[0]] = [int(k) for k in l[1:]]

    pairs = []
    with open(pair_pth) as f:
        pairs_lines = f.readlines()[1:]
        for pl in pairs_lines:
            pairs.append(pl.replace('\n', '').split('\t'))

    return zfile, landmark, pairs


def dataset(num):
    zfile, landmark, pairs = get_zip_landmark_pair()
    img = None
    imgs = np.zeros([num*4, 3, 112, 96])
    img_idx = 0
    labels = {}
    for i in range(num):
        p = pairs[i]

        if 3 == len(p):
            sameflag = 1
            name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
            name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))

        if 4 == len(p):
            sameflag = 0
            name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
            name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))

        img1 = alignment(cv2.imdecode(np.frombuffer(zfile.read('lfw/' + name1), np.uint8), 1), landmark[name1])
        img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read('lfw/' + name2), np.uint8), 1), landmark[name2])

        imglist = [img1, cv2.flip(img1, 1), img2, cv2.flip(img2, 1)]
        for j in range(len(imglist)):
            imglist[j] = imglist[j].transpose(2, 0, 1).reshape((1, 3, 112, 96))
            # imglist[j] = (imglist[j] - 127.5) / 128.0
            imgs[img_idx, ...] = (imglist[j] - 127.5) / 128.0
            label = {img_idx: [{'name1': name1}, {'name2': name2}, {'sameflag': sameflag}]}
            labels.update(label)

            img_idx += 1
        if (i % 100) == 0:
            print(f" i = {i}")

    return imgs, labels


def main():
    bp = '/project/ai/scratch01/samhan01/work/test/models/sphereface/dataset'
    img, labels = dataset(num=6000)
    np.save(os.path.join(bp, 'dataset_6000x4.npy'), img)
    np.save(os.path.join(bp, 'label_6000x4.npy'), labels)

    calib_img, calib_label = dataset(num=100)
    np.save(os.path.join(bp, 'dataset_100x4.npy'), calib_img)
    np.save(os.path.join(bp, 'label_100x4.npy'), calib_label)


if __name__ == '__main__':
    main()
