# SPDX-License-Identifier: Apache-2.0
# Copyright © 2022-2025 Arm Technology (China) Co. Ltd.

from PIL import Image
import tensorflow as tf
import numpy as np
import os


class ImageData(object):

    def __init__(self, param):
        pass

    @staticmethod
    def smallest_size_at_least(height, width, resize_min):
        """Computes new shape with the smallest side equal to `smallest_side`.

        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.

        Args:
          height: an int32 scalar tensor indicating the current height.
          width: an int32 scalar tensor indicating the current width.
          resize_min: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.

        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: an int32 scalar tensor indicating the new width.
        """
        resize_min = tf.cast(resize_min, tf.float32)

        # Convert to floats to make subsequent calculations go smoothly.
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

        smaller_dim = tf.minimum(height, width)
        scale_ratio = resize_min / smaller_dim

        # Convert back to ints to make heights and widths that TF ops will accept.
        new_height = tf.cast(tf.round(height * scale_ratio), tf.int32)
        new_width = tf.cast(tf.round(width * scale_ratio), tf.int32)

        return new_height, new_width

    @staticmethod
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

    @staticmethod
    def aspect_preserving_resize(image, resize_min, channels=3, method='BILINEAR'):
        """Resize images preserving the original aspect ratio.

        Args:
          image: A 3-D image `Tensor`.
          resize_min: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.

        Returns:
          resized_image: A 3-D tensor containing the resized image.
        """
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        new_height, new_width = ImageData.smallest_size_at_least(height, width, resize_min)
        return ImageData.resize_image(image, new_height, new_width, method)

    @staticmethod
    def central_crop(image, crop_height, crop_width, channels=3):
        """Performs central crops of the given image list.

        Args:
          image: a 3-D image tensor
          crop_height: the height of the image following the crop.
          crop_width: the width of the image following the crop.

        Returns:
          3-D tensor with cropped image.
        """
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        amount_to_be_cropped_h = height - crop_height
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = width - crop_width
        crop_left = amount_to_be_cropped_w // 2
        # return tf.image.crop_to_bounding_box(image, crop_top, crop_left, crop_height, crop_width)

        size_assertion = tf.Assert(
            tf.logical_and(
                tf.greater_equal(height, crop_height),
                tf.greater_equal(width, crop_width)),
            ['Crop size greater than the image size.']
        )
        with tf.control_dependencies([size_assertion]):
            if channels == 1:
                image = tf.squeeze(image)
                crop_start = [crop_top, crop_left, ]
                crop_shape = [crop_height, crop_width, ]
            elif channels >= 3:
                crop_start = [crop_top, crop_left, 0]
                crop_shape = [crop_height, crop_width, -1]

            image = tf.slice(image, crop_start, crop_shape)

        return tf.reshape(image, [crop_height, crop_width, -1])

    @staticmethod
    def mean_image_subtraction(image):
        return tf.subtract(image, 128.0)

    @staticmethod
    def read_image(image_file, encode_format='jpeg'):
        ext_name = os.path.splitext(image_file)[-1][1:].lower()
        if ext_name in ['jpeg', 'jpg']:
            with tf.Session() as sess:
                img_stream = tf.gfile.GFile(image_file, 'rb').read()
                img = tf.image.decode_jpeg(img_stream)
                img = sess.run(img)
                [height, width, channel] = img.shape
        elif ext_name == 'png':
            with tf.Session() as sess:
                # read the label image -- pillow
                img = Image.open(image_file)
                img = np.array(img)
                img = np.expand_dims(img, 2)
                img_stream = tf.image.encode_png(img)
                img_stream = sess.run(img_stream)
                [height, width, channel] = img.shape
                # read the label image -- tensorflow
                # label_stream = tf.gfile.GFile(label_name, 'rb').read()
                # label = tf.image.decode_png(label_stream, channels=1)
                # label = sess.run(label)
        elif ext_name == 'npy':
            img = np.load(image_file)
            [height, width, channel] = img.shape
            with tf.Session() as sess:
                if encode_format == 'jpeg':
                    img_stream = img.tobytes()
                elif encode_format == 'png':
                    img_stream = tf.image.encode_png(img)
                    img_stream = sess.run(img_stream)
        else:  # binary
            height, width, channel = 0, 0, 0
            print('Not supported image format yet.')
        return img_stream, height, width, channel
