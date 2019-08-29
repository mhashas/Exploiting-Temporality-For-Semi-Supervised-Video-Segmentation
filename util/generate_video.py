import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

class VideoGenerator():

    def __init__(self, images_dir, save_dir):
        self.images_dir = images_dir
        self.save_dir = save_dir

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    def generate_video(self):
        images = []
        image_names = os.listdir(self.images_dir)
        image_names.sort()

        for i in tqdm(range(len(image_names))):
            image_name = image_names[i]
            if not image_name.endswith('.png'):
                continue
            img = cv2.imread(self.images_dir + image_name)
            height, width, layers = img.shape
            size = (width, height)
            images.append(img)

        out = cv2.VideoWriter(save_dir + 'project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in tqdm(range(len(images))):
            out.write(images[i])
        out.release()

        return

if __name__ == '__main__':
    predictions_dir = '../images/results_whateva/cityscapes/unet/unet-num_downs:8-ngf:128-down_type:maxpool-mode:sequence-1234-seq_model:tcn+temporal_encoder-num_levels_tcn:2-tcn_kernel_size:1-init_type:normal-optim:amsgrad-lr:0.0001-clipping:5-resize:512,256-td:2-epochs:200/experiment_0/'
    save_dir = predictions_dir + 'video/'
    helper = VideoGenerator(predictions_dir, save_dir)
    helper.generate_video()
