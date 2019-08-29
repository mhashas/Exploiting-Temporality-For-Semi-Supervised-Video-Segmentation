import os
import torch
import numpy as np
import random
from PIL import Image
from torch.utils import data
import util.custom_transforms as custom_transforms
import torchvision.transforms as standard_transforms

root = "../datasets/cityscapes/"

visualize = standard_transforms.ToTensor()
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

CITYSCAPES_CLASS_COLOR_MAPPING = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
    -1: (0, 0, 142),
    255: (0, 0, 0),
}

TRAINID_TO_ID = {0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17,
                 6: 19, 7: 20, 8: 21, 9: 22, 10: 23, 11: 24,
                 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33, 255: 255}

MY_CITYSCAPES_CLASS_COLOR_MAPPING = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    255: 255
}

visualize = standard_transforms.ToTensor()

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def colorize_mask_submit(mask):
    # mask: numpy array of the mask
    new_mask = np.random.rand(256,512)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            new_mask[i][j] = TRAINID_TO_ID[int(mask[i][j])]

    return Image.fromarray(new_mask.astype(np.uint8), 'L')

def colorize_mask_color(mask):
    # mask: numpy array of the mask
    new_mask = np.random.rand(512,256,3)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            new_mask[i][j] = CITYSCAPES_CLASS_COLOR_MAPPING[int(mask[i][j])]

    return Image.fromarray(new_mask.astype(np.uint8))


class CityScapes(data.Dataset):

    num_classes = 19
    classes = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
               "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    def __init__(self, quality, split, blank=False, input_size=512, base_size=1024, time_dilation=1, reconstruct=0, shuffle=0):
        self.ignore_label = 255
        self.quality = quality
        self.split = split
        self.reconstruct = reconstruct
        self.shuffle = shuffle
        self.blank = blank
        self.input_size = input_size
        self.base_size = base_size
        self.time_dilation = time_dilation

        if split == 'demoVideo':
            self.imgs = self.make_demo_dataset()
        else:
            self.imgs = self.make_dataset()

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform, self.input_transform, self.target_transform = self.get_transforms()

        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                              3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                              7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                              14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                              18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}


    def get_transforms(self):
        mean_std = ([0.3006, 0.3365, 0.2956], [0.1951, 0.1972, 0.1968])


        if self.split == 'train':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size),
                custom_transforms.RandomHorizontallyFlip(),
                custom_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                custom_transforms.RandomGaussianBlur()
            ])
        elif self.split == 'val' or self.split == 'test' or self.split == 'demoVideo':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.input_size),
            ])
        else:
            raise RuntimeError('Invalid dataset mode')

        '''
        if self.split == 'train':
            joint = custom_transforms.Compose([
                custom_transforms.Scale(self.base_size),
                custom_transforms.RandomCrop(size=self.input_size),
                custom_transforms.RandomHorizontallyFlip(),
                custom_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                custom_transforms.RandomGaussianBlur(),
            ])
        elif self.split == 'val':
            joint = custom_transforms.Compose([
                custom_transforms.Scale(self.base_size)
            ])
        else:
            raise RuntimeError('Invalid dataset mode')
        '''

        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])

        target_transform = custom_transforms.MaskToTensor()

        return joint, input_transform, target_transform

    def __getitem__(self, index):
       if 'sequence' in self.quality or self.quality == 'fbf-1234':
           return self.get_item_sequence(index)
       elif 'fbf' in self.quality:
           return self.get_item_fbf(index)

    def get_item_sequence(self, index):
        img_paths, mask_path = self.imgs[index]
        images = []

        if self.shuffle:
            copy = img_paths[:len(img_paths) - 1]
            random.shuffle(copy)
            img_paths[:len(img_paths) - 1] = copy

        for img_path in img_paths:
            images.append(Image.open(img_path).convert('RGB'))

        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))
        og_mask = mask

        if self.joint_transform is not None:
            images, mask = self.joint_transform(images, og_mask)

        if self.input_transform is not None:
            for i in range(len(images)):
                images[i] = self.input_transform(images[i])

        if self.blank:
            images[len(images) - 1][True] = 0

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.reconstruct:
            reconstruct_images = [image.clone() for image in images]
            images.pop()
            reconstruct_images.pop(0)
            reconstruct_images = torch.stack(reconstruct_images)

        if self.quality == 'fbf-1234':
            images = torch.cat(images, dim=0)
        else:
            images = torch.stack(images)


        if self.reconstruct:
            return images, mask, reconstruct_images, img_path.split('/')[-1]
        else:
            return images, mask, img_path.split('/')[-1]

    def get_item_fbf(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform([img], mask)
            img = img[0]

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask, img_path.split('/')[-1]

    def __len__(self):
        return len(self.imgs)

    def make_dataset(self):
        fbf_img_dir_name = 'leftImg8bit' #used to faster get filenames when in quality sequence
        fbf_img_path = os.path.join(root, fbf_img_dir_name, self.split)

        if self.quality == 'coarse':
            img_dir_name = 'leftImg8bit_trainextra' if self.split == 'train_extra' else 'leftImg8bit_trainvaltest'
            mask_path = os.path.join(root, 'gtCoarse', 'gtCoarse', self.split)
            mask_postfix = '_gtCoarse_labelIds.png'
        elif self.quality == 'fbf':
            img_dir_name = fbf_img_dir_name
            mask_path = os.path.join(root, 'gtFine', self.split)
            mask_postfix = '_gtFine_labelIds.png'
        else: # 'fbf-previous' + everything sequence
            img_dir_name = 'leftImg8bit_sequence'
            mask_path = os.path.join(root, 'gtFine', self.split)
            mask_postfix = '_gtFine_labelIds.png'

        img_path = os.path.join(root, img_dir_name, self.split)

        items = []
        categories = os.listdir(mask_path)

        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(fbf_img_path, c))]

            for it in c_items:
                if 'sequence' in self.quality or self.quality == 'fbf-1234':
                    indices = [(int(i)-4)*self.time_dilation for i in list(self.quality.split('-')[-1])]
                    if self.reconstruct:
                        indices.append(indices[-1] + self.time_dilation)

                    images = []
                    sequence_number = it.split('_')[-1]

                    for i in indices:
                        new_sequence_number = str(int(sequence_number) + i).zfill(6)
                        new_sequence_number = it.rsplit('_' + sequence_number, 1)[0] + '_' + new_sequence_number

                        images.append(os.path.join(img_path, c, new_sequence_number + '_leftImg8bit.png'))

                    item = (list(images), os.path.join(mask_path, c, it  + mask_postfix))
                elif 'fbf' in self.quality:
                    if self.quality == 'fbf-previous':
                        sequence_number = it.split('_')[-1]
                        new_sequence_number = str(int(sequence_number) - self.time_dilation).zfill(6)
                        img_it = it.rsplit('_' + sequence_number, 1)[0] + '_' + new_sequence_number
                    else:
                        img_it = it

                    item = (os.path.join(img_path, c, img_it + '_leftImg8bit.png'), os.path.join(mask_path, c, it + mask_postfix))

                items.append(item)

        return items

    def make_demo_dataset(self):
        img_path = os.path.join(root, 'leftImg8bit', self.split)
        mask_path = '../datasets/cityscapes/gtFine/test/berlin/berlin_000143_000019_gtFine_labelIds.png'
        categories = os.listdir(img_path)
        items = []

        for c in categories:
            c_items = [name.split('_leftImg8bit.png')[0] for name in os.listdir(os.path.join(img_path, c))]
            c_items.sort()
            c_items = c_items[6:]

            for it in c_items:
                if 'sequence' in self.quality or self.quality == 'fbf-1234':
                    indices = [(int(i)-4)*self.time_dilation for i in list(self.quality.split('-')[-1])]
                    if self.reconstruct:
                        indices.append(indices[-1] + self.time_dilation)

                    images = []
                    sequence_number = it.split('_')[-1]

                    for i in indices:
                        new_sequence_number = str(int(sequence_number) + i).zfill(6)
                        new_sequence_number = it.rsplit('_' + sequence_number, 1)[0] + '_' + new_sequence_number

                        images.append(os.path.join(img_path, c, new_sequence_number + '_leftImg8bit.png'))

                    item = (list(images), mask_path)
                elif 'fbf' in self.quality:
                    if self.quality == 'fbf-previous':
                        sequence_number = it.split('_')[-1]
                        new_sequence_number = str(int(sequence_number) - self.time_dilation).zfill(6)
                        img_it = it.rsplit('_' + sequence_number, 1)[0] + '_' + new_sequence_number
                    else:
                        img_it = it

                    item = (os.path.join(img_path, c, img_it + '_leftImg8bit.png'), mask_path)

                items.append(item)

        return items
