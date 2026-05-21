from AIPUBuilder.plugin_loader import register_plugin
from AIPUBuilder.plugin_loader import PluginType
from torch.utils.data import Dataset
# from skimage import io,transform
from torchvision import transforms
import numpy as np
import os
import cv2


@register_plugin(PluginType.Dataset, '0.01')
class FaceboxDataset(Dataset):

    def __init__(self, data_file,resize_height=1024, resize_width=1024):
        self.image_label_list = self.read_file(data_file)
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.len = len(self.image_label_list)

        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return 3
        return len(self.image_label_list)


    def __getitem__(self, index):
        image_path = self.image_label_list[index]
        img = cv2.imread(image_path)
        im2 = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_LINEAR)
        label=1
        sample = [im2, label]
        return sample

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip()
                image_label_list.append(content)
        return image_label_list