import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms

import random
import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageOps
import sklearn
from tqdm import tqdm, trange

class GeneralDataset(Dataset):
    def __init__(self, root,
                 main_transform=None, pair_transform=None, comp_transform=None, 
                 invert=False, **kwargs):
        super(GeneralDataset).__init__()
        self.root = root
        self.main_transform = main_transform
        self.pair_transform = pair_transform
        self.comp_transform = comp_transform
        self.invert = invert
        self._dataset = ImageFolder(root=self.root)
        self.label_image_dict = self._create_label_image_dict()
        self.dataframe = self._build_pair_set()

    def _build_pair_set(self):
        simm_df = self._create_similar_pair()
        diff_df = self._create_different_pair()
        diff_df = self._balance_different_pair(simm_df, diff_df)
        dataframe = self._combine_simm_diff_pair(simm_df, diff_df)
        return dataframe

    def _create_label_image_dict(self):
        img_label_list = sorted(self._dataset.imgs, key=lambda x: x[1],  reverse=False)
        label_image_dict = {lbl: [] for img, lbl in img_label_list}
        for impath, label in img_label_list:
            label_image_dict[label].append(impath)
        return label_image_dict

    def _create_similar_pair(self):
        # create pair same pair
        simm_pair = {'main_image': [], 'main_label_idx':[], 'main_label_name': [], 
                    'comp_image': [], 'comp_label_idx':[], 'comp_label_name': [], 
                    'label': [], 'status':[]}
        for key, list_value in self.label_image_dict.items():
            for idx, main_img in enumerate(list_value):
                for jdx, comp_image in enumerate(list_value):
                    if idx!=jdx:
                        simm_pair['main_image'].append(main_img)
                        simm_pair['main_label_name'].append(self._dataset.classes[key])
                        simm_pair['main_label_idx'].append(key)
                        simm_pair['comp_image'].append(comp_image)
                        simm_pair['comp_label_name'].append(self._dataset.classes[key])
                        simm_pair['comp_label_idx'].append(key)
                        simm_pair['label'] = int(key != key)
                        simm_pair['status'] = 'similar'
        simm_df = pd.DataFrame(simm_pair) 
        return simm_df

    def _create_different_pair(self):
        diff_pair = {'main_image': [], 'main_label_idx':[], 'main_label_name': [], 
             'comp_image': [], 'comp_label_idx':[], 'comp_label_name': [], 
             'label': [], 'status':[]}
        for main_key, main_list_value in tqdm(self.label_image_dict.items()):
            for diff_key, diff_list_value in self.label_image_dict.items():
                if main_key!=diff_key:
                    for idx, main_img in enumerate(main_list_value):
                        for jdx, comp_image in enumerate(diff_list_value):
                            diff_pair['main_image'].append(main_img)
                            diff_pair['main_label_name'].append(self._dataset.classes[main_key])
                            diff_pair['main_label_idx'].append(main_key)
                            diff_pair['comp_image'].append(comp_image)
                            diff_pair['comp_label_name'].append(self._dataset.classes[diff_key])
                            diff_pair['comp_label_idx'].append(diff_key)
                            diff_pair['label'] = int(main_key != diff_key)
                            diff_pair['status'] = 'different'
        diff_df = pd.DataFrame(diff_pair)
        return diff_df

    def _balance_different_pair(self, simm_df, diff_df, random_state=1261):
        diff_df_list = []
        for idx, name in enumerate(self._dataset.classes):
            label_name = self._dataset.classes[idx]
            simm_df_by_idx = simm_df[simm_df['main_label_name'] == label_name]
            len_simm_idx = len(simm_df_by_idx)

            label_name = self._dataset.classes[idx]
            diff_df_by_idx = diff_df[diff_df['main_label_name'] == label_name]
            len_diff_idx = len(diff_df_by_idx)

            balance_ratio = len_simm_idx / len_diff_idx
            diff_df_ratio_idx = diff_df_by_idx.sample(frac=balance_ratio, random_state=random_state).reset_index(drop=True)
            diff_df_list.append(diff_df_ratio_idx)

        diff_df = pd.concat(diff_df_list)
        return diff_df
    
    def _combine_simm_diff_pair(self, simm_df, diff_df, shuffle=True, random_state=1261):
        main_df = pd.concat([simm_df, diff_df])
        main_df = main_df.reset_index(drop=True)
        if shuffle:
            main_df = sklearn.utils.shuffle(main_df, random_state=1261)
            main_df = main_df.reset_index(drop=True)
        return main_df
    
    def _load_image(self, path: str, to_rgb=True):
        image = Image.open(path)
        if to_rgb:
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        return image
    
    def _preprocess_label(self, label):
        label_numpy = np.array([label],dtype=np.float32)
        return torch.from_numpy(label_numpy)
                    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        main_path, comp_path = record['main_image'], record['comp_image']
        main_image, comp_image = self._load_image(main_path), self._load_image(comp_path)
        label = self._preprocess_label(record['label'])
        
        if self.invert:
            main_image = ImageOps.invert(main_image)
            comp_image = ImageOps.invert(comp_image)


        if self.pair_transform:
            main_image, comp_image = self.pair_transform(main_image, comp_image)
        
        if self.main_transform:
            main_image = self.main_transform(main_image)
            
        if self.comp_transform:
            comp_image = self.comp_transform(comp_image)
        
        return main_image, comp_image, label


class GeneralRandomDataset(Dataset):
    def __init__(self, root: str, transform=None, should_invert: bool = False, **kwargs):
        super(GeneralRandomDataset, self).__init__()
        self.root = root
        self.dataset: ImageFolder = ImageFolder(root=root)
        self.transform = transform
        self.should_invert = should_invert
        
    def _get_random_pair(self):
        img0_tuple = random.choice(self.dataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.dataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.dataset.imgs)
            
        return img0_tuple, img1_tuple
            
    def _load_image(self, path: str, to_rgb=True):
        image = Image.open(path)
        if to_rgb:
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        return image
    
    def _get_pair_label(self, label1, label2):
        label = int(label1 != label2)
        label_numpy = np.array([label],dtype=np.float32)
        return torch.from_numpy(label_numpy)
        
    def __getitem__(self, idx):
        (img0_path, img0_label), (img1_path, img1_label) = self._get_random_pair()
        label = self._get_pair_label(img0_label, img1_label)
        
        img0 = self._load_image(img0_path)
        img1 = self._load_image(img1_path)
        
        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        
        return img0, img1, label
    
    def __len__(self):
        return len(self.dataset.imgs)
    
    
if __name__ == "__main__":
    
    train_path  = 'dataset/train'
    valid_path  = 'dataset/valid'
    
    dataset = GeneralDataset(root=train_path)
    im1, im2, label = dataset[0]
    print(label)
    
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2,1)
    
    axarr[0].imshow(im1)
    axarr[1].imshow(im2)
    plt.show(block=True)