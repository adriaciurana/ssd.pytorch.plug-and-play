#from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd

class LogoDataset(data.Dataset):
    CLASSES = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google', 'Heineken', 'HP', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma', 'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo']
        
    def __init__(self, root, image_set='train', transform=None,
                 target_transform=None, dataset_name='Flickr Logos 27', label="flickr_logos_27"):
        filename = 'flickr_logos_27_dataset_training_set_annotation.txt' if image_set == 'train' else 'flickr_logos_27_dataset_query_set_annotation.txt'
        self.root = root
        csv_data = pd.read_csv(root + "/" + filename, sep=' ', header=None, usecols=[0, 1, 2, 3, 4, 5, 6], names=['filename', 'brand', 'train_subset', 'xmin', 'ymin', 'xmax', 'ymax'])
        self.classes = LogoDataset.CLASSES #list(csv_data.loc[:, 'brand'].unique())
        self.list_of_images = list(csv_data.loc[:, 'filename'].unique())
        # Brand_id
        for i, class_ in enumerate(self.classes):
            csv_data.loc[csv_data.loc[:, 'brand'] == class_, 'brand_id'] = i

        # Dict of images
        self.bbox_of_images = []
        len_list = len(self.list_of_images)
        for i, im_name in enumerate(self.list_of_images):
            data_name = csv_data.loc[csv_data.loc[:, 'filename'] == im_name, ('xmin', 'ymin', 'xmax', 'ymax', 'brand_id')].drop_duplicates().values
            self.bbox_of_images.append(data_name)
             
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.label = label
        self.num_classes = len(self.classes)

    def __getitem__(self, index):
        im_name = self.list_of_images[index]
        target = self.bbox_of_images[index]

        sample = cv2.imread(self.root + "/images/" + im_name) #[..., ::-1]
        height, width, _ = sample.shape
        
        if self.transform is not None:
            sample, target = self.transform(sample, target)
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        return torch.from_numpy(sample).permute(2, 0, 1), target

    def __len__(self):
        return len(self.list_of_images)