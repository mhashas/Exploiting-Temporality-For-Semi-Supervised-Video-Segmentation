import os
import numpy as np
from tqdm import tqdm
from PIL import Image

class PredictionsUpsampler():

    def __init__(self, predictions_dir, targets_dir):
        self.predictions_dir = predictions_dir
        self.targets_dir = targets_dir

        if not os.path.isdir(targets_dir):
            os.makedirs(targets_dir)

    def upsample_images(self):
        images = []
        image_names = os.listdir(self.predictions_dir)
        image_names.sort()

        for i in tqdm(range(len(image_names))):
            image_name = image_names[i]
            if not image_name.endswith('.png'):
                continue
            image = Image.open(self.predictions_dir + image_name)
            image_name = image_name.replace('.png', '')
            image = image.resize((2048,1024), Image.NEAREST)
            image.save(self.targets_dir + image_name + '.png', )


        return images



if __name__ == '__main__':
    predictions_dir = '../images/results_submit_2/cityscapes/unet/unet-num_downs:8-ngf:128-down_type:maxpool-mode:sequence-1234-seq_model:tcn2d+temporal_encoder-num_levels_tcn:2-tcn_kernel_size:2-init_type:normal-optim:amsgrad-lr:0.0001-clipping:5-resize:512,256-td:2-epochs:200/experiment_0/'
    targets_dir = predictions_dir + 'upsampled/'
    helper = PredictionsUpsampler(predictions_dir, targets_dir)
    helper.upsample_images()
