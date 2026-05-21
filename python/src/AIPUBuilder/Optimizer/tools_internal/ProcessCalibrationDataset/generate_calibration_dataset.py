import random
import argparse
import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image
tf.enable_eager_execution()


class preprocessing():

    @staticmethod
    def aspect_preserving_resize(input, resize_min):
        height, width, _ = input.shape
        smaller_size = np.minimum(height, width)
        scale = resize_min / smaller_size
        new_height = int(round(height * scale))
        new_width = int(round(width * scale))
        output = cv2.resize(input, (new_height, new_width), interpolation=cv2.INTER_LINEAR)
        return output

    @staticmethod
    def get_smallest_size(input):
        height, width, _ = input.shape
        return np.minimum(height, width)

    @staticmethod
    def normalize(input, mean, var):
        output = (input - mean) / var
        return output

    @staticmethod
    def crop_central(input, crop_height, crop_width):
        height, width, _ = input.shape
        crop_h_size = height - crop_height
        crop_w_size = width - crop_width
        crop_h_start = crop_h_size // 2
        crop_w_start = crop_w_size // 2
        output = input[crop_h_start:crop_h_start+crop_height, crop_w_start:crop_w_start+crop_width]
        return output


preprocess_param = {
    # in_h, in_w, in_c, mean, var, max, min
    'alexnet': [28, 28, 1, 0.0, 256.0, 1.0, 0.0],
    'resnet': [224, 224, 3, [123.68, 116.78, 103.94], 1, 151, -123],  # resnet50, resnet101
    'mobilenet': [224, 224, 3, 128, 128, 1, -1],
    'mobilenet_v2': [512, 512, 3, 127.5, 127.5, 1, -1],
    'inception': [299, 299, 3, 0, 255, 1, 0],  # inceptionv3, inceptionv4
    'fcn': [224, 224, 3, [123.68, 116.78, 103.94], 1, 151, -123],
    'deeplab': [513, 513, 3, 128, 128, 1, -1],
    'tflite_deeplab': [257, 257, 3, 128, 128, 1, -1],
    'yolo': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'yolo_v3': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'ssd': [640, 640, 3, [123.68, 116.78, 103.94], 1, 151, -123],
    'shufflenet': [224, 224, 3, 128, 128, 1, -1],
    'swin': [224, 224, 3, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], 2.64, -2.118],
    'mobilenet_ssd': [300, 300, 3, 128, 128, 1, -1],
    'nasnet': [224, 224, 3, 127.5, 127.5, 1, -1],
    'vgg': [224, 224, 3, [123.68, 116.78, 103.94], 1, 151, -123],
    'onnx_ssd': [1200, 1200, 3, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], 2.64, -2.118],
    'onnx_yolo2_coco': [416, 416, 3, 0, 255, 1.0, 0.0],
    'caffe_yolo': [416, 416, 3, 0.0, 256.0, 1.0, 0.0],
    'caffe_yolo3': [608, 608, 3, 0.0, 256.0, 1.0, 0.0],
    'fasterrcnn': [600, 600, 3, [123.68, 116.779, 103.939], 1, 151, -123],
    'caffe_resnet_xx': [224, 224, 3, [122.7717, 115.9465, 102.9801], 1, 152, -123],
    'caffe_mobilenet_v2_ssd': [300, 300, 3, 127.5, 127.5, 1, -1],
    'caffe_alexnet': [227, 227, 3, [122.6789, 116.6688, 104.007], 1, 152, -123],
    'tensorflow_efficientnet_b5': [224, 224, 3, 0.0, 255.0, 1.0, 0.0],
    'tensorflow_facenet': [160, 160, 3, 0.0, 255.0, 1.0, 0.0],
    'segformer': [1024, 1024, 3, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375], 2.64, -2.118],
}


def resize_image(image, height, width, method='BILINEAR'):
    resize_func = tf.image.ResizeMethod.NEAREST_NEIGHBOR if method == 'NEAREST' else tf.image.ResizeMethod.BILINEAR
    return tf.image.resize_images(image, [height, width], method=resize_func, align_corners=False)


def featureNormalize(X0):

    n = X0.shape[1]
    X = np.zeros((n, n))
    X[:] = X0[:]
    mu = np.zeros((1, n))
    sigma = np.zeros((1, n))

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(n):
        X[:, i] = (X[:, i] - mu[i]) / (sigma[i]+1e-8)
    return X, mu, sigma


def projectData(X_norm, U, K):
    Z = np.zeros((X_norm.shape[0], K))

    U_reduce = U[:, 0:K]  # 取前K个
    Z = np.dot(X_norm, U_reduce)
    return Z


def recoverData(Z, U_recude, mu, sigma):
    X_rec = np.zeros((Z.shape[0], U_recude.shape[0]))
    X_rec = np.dot(Z, np.transpose(U_recude))  # recover data
    for i in range(X_rec.shape[0]):
        X_rec[:, i] = (X_rec[:, i]*sigma[i] + mu[i])

    return X_rec


def pca_decompose(image_new, k):
    U = []
    Z = []
    MU = []
    SIGMA = []
    for c in range(3):
        X, mu, sigma = featureNormalize(image_new[..., c])
        m = X.shape[0]

        Sigma = np.dot(np.transpose(X), X) / m  # Sigma
        u, s, v = np.linalg.svd(Sigma)
        z = projectData(X, u, k)

        # #############################################
        # X_rec = np.zeros((z.shape[0], u.shape[0]))
        # U_recude = u[:, 0:k]
        # X_rec = np.dot(z, np.transpose(U_recude))
        # ############################################3
        U.append(u[:, 0:k])
        Z.append(z)
        MU.append(mu)
        SIGMA.append(sigma)
    return Z, U, MU, SIGMA
##########################################################################################


def image_fused_based_pca(image_new, fused_image, alpha):
    # PCA

    # Z is reduced dimension matrix,U is principal sigular vector
    # Z_org, U_org, mu_org, sigma_org = pca_decompose(fused_image,image_new.shape[0]//2)

    Z_new, U_new, mu_new, sigma_new = pca_decompose(image_new, int(image_new.shape[0]/4))

    for c in range(3):
        X_org = fused_image[..., c]  # recoverData(Z_org[c],U_org[c],mu_org[c], sigma_org[c])  #

        X_new = recoverData(Z_new[c], U_new[c], mu_new[c], sigma_new[c])

        R = X_org*(1-alpha)+(alpha)*X_new
        fused_image[..., c] = np.clip(R, 0, 255)


def crop_central(input, crop_height, crop_width):
    height, width, _ = input.shape
    crop_h_size = height - crop_height
    crop_w_size = width - crop_width
    crop_h_start = crop_h_size // 2
    crop_w_start = crop_w_size // 2
    output = input[crop_h_start:crop_h_start+crop_height, crop_w_start:crop_w_start+crop_width]
    return output
# 3


def cluster_image(filename_list, img_dir, images, img_param, argv):

    input_height = img_param[0]
    input_width = img_param[1]
    input_channel = img_param[2]
    mean = img_param[3]
    var = img_param[4]
    # init cluster_n image center
    find_n = 0
    clusters_n = argv.data_num
    loop_n = argv.fused_num
    corr_thr0 = argv.corr_thr0
    corr_thr1 = argv.corr_thr1
    fused_ratio = argv.fused_ratio
    DATA_NUM = argv.metric_datasize
    clust_s = 0
    clust_e = argv.cluster_end
    fused_s = clust_e
    fused_e = DATA_NUM
    while (find_n != clusters_n):
        filename = random.sample(filename_list[clust_s:clust_e], 1)[0]
        image_file = os.path.join(img_dir, filename)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (int(input_height), int(input_width)), interpolation=cv2.INTER_LANCZOS4)
        # image = crop_central(image, input_height, input_width)
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = image.astype(np.float32)

        # U,Z = pca_decompose(image,20)

        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        find_i = -1
        find = True
        for find_i in range(find_n):
            # images_gray = cv2.cvtColor(images[find_i], cv2.COLOR_BGR2GRAY)
            # r_corr = cv2.matchTemplate(image_gray, images_gray, cv2.TM_CCOEFF_NORMED)
            r_corr = cv2.matchTemplate(image, images[find_i], cv2.TM_CCOEFF_NORMED)
            if r_corr > corr_thr0 * enhance_ratio:
                find = False
                break
        if find and (find_i + 1 == find_n):
            images[find_n] = image
            find_n = find_n + 1
            enhance_ratio = find_n / (find_n + 1)

    images_tmp = np.zeros([clusters_n, input_height, input_width, input_channel], np.float32)
    match_num = np.zeros([clusters_n], np.float32)
    images_tmp[:] = images[:]

    corr_thresh = corr_thr1

    while((np.min(match_num) < loop_n*0.55) and (np.sum(match_num == loop_n)/clusters_n < 0.9)):
        if np.mean(match_num) > loop_n*0.55:
            corr_thresh = corr_thr1*0.7826

    # for j in range(loop_n):
        filename = random.sample(filename_list[fused_s:fused_e], clusters_n)
        for file_name, img_idx in zip(filename, range(clusters_n)):
            image_file = os.path.join(img_dir, file_name)
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(input_height), int(input_width)), interpolation=cv2.INTER_LANCZOS4)
            # image = crop_central(image, input_height, input_width)
            image = np.clip(image, 0, 255)
            image = image.astype(np.float32)
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # assign to most corr cluster
            # images_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
            r_corr = 0  # cv2.matchTemplate(image_gray, images_gray, cv2.TM_CCOEFF_NORMED)
            # r_corr = np.corrcoef(image.flatten(), images[0].flatten())[0][1]
            best_i = 0

            for i in range(0, clusters_n, 1):
                if match_num[i] == loop_n:
                    continue
                images_gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
                corr = cv2.matchTemplate(image_gray, images_gray, cv2.TM_CCOEFF_NORMED)
                if corr > r_corr:
                    r_corr = corr
                    best_i = i
            if (r_corr) > corr_thresh and match_num[best_i] < loop_n:
                match_num[best_i] = match_num[best_i] + 1
                image_fused_based_pca(image, images_tmp[best_i], fused_ratio)

    idx = np.argsort(-match_num)
    images[:] = images_tmp[idx]
    for i in range(len(idx)):
        images[i] = (images[i] - mean) / var
    return images


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name", default='')
    parser.add_argument('--dataset_dir', type=str, help="dataset dir",
                        default="/project/ai/scratch01/Dataset/ImageNet/ILSVRC2012_img_val/")
    parser.add_argument('--data_num', type=int, default=20, help='init momentum value')
    parser.add_argument('--output_dir', type=str, help="dataset dir", default='./')
    parser.add_argument('--fused_num', type=int, default=20, help='fused images number')
    parser.add_argument('--fused_ratio', type=float, default=0.1, help='fused ratio')
    parser.add_argument('--cluster_end', type=int, default=2000,
                        help='how many search space for cluster alg in metric dataset')
    parser.add_argument('--metric_datasize', type=int, default=5000, help='how many search space in valid dataset')
    parser.add_argument('--corr_thr0', type=float, default=0.07, help='find most correlated images threshold')
    parser.add_argument('--corr_thr1', type=float, default=0.23, help='find least correlated images threshold')
    parser.add_argument('--layout_format', type=str, help="data layout format", default='nhwc')
    args = parser.parse_args()
    return args


def main():
    argv = get_args()
    model_name = argv.model
    data_num = argv.data_num
    dataset_dir = argv.dataset_dir
    output_dir = argv.output_dir
    layout_format = argv.layout_format
    image_model_dir = os.path.join(output_dir, 'preprocess', model_name)
    if not os.path.exists(image_model_dir):
        os.makedirs(image_model_dir)
    h = preprocess_param[model_name][0]
    w = preprocess_param[model_name][1]
    out_path = os.path.join(image_model_dir, 'dataset_' + str(h) + '_' + str(w) + '_' + str(data_num)+'.npy')

    if model_name in preprocess_param:
        param = preprocess_param[model_name]
        # the input height and width of the model.
        input_height = param[0]
        input_width = param[1]
        input_channel = param[2]

    else:
        print("Not supported model: ", model_name)
        return
    filename_list = []
    # notice:all images files are saved in same dir
    for rootdir, dirnames, filename in os.walk(dataset_dir):

        if len(filename) != 0:
            filename_list += filename
        images = np.zeros([data_num, input_height, input_width, input_channel], np.float32)
        ##################################################################################
        # call new alg to generate calibration dataset
        images = cluster_image(filename_list, dataset_dir, images, param, argv)
        if layout_format == 'nchw':
            images = images.transpose((0, 3, 1, 2))
        np.save(out_path, images)


if __name__ == '__main__':

    main()
