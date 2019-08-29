import os
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class PixelMapGenerator():

    def __init__(self, predictions_dir, targets_dir):
        self.prediction_images = self.get_images(predictions_dir)
        self.target_images = self.get_images(targets_dir)

    def get_images(self, dir):
        images = []
        for image_name in os.listdir(dir):
            images.append(image.imread(dir + image_name))

        return images

    def update_histogram(self, pred, target, histogram):
        for x in range(pred.shape[0]):
            for y in range(pred.shape[1]):
                if np.array_equal(pred[x,y], target[x,y]):
                    histogram[x,y] += 1

    def generate_histogram(self):
        histogram = np.zeros(np.array(self.prediction_images[0]).shape[:-1])

        for i in tqdm(range(len(self.prediction_images))):
            pred = self.prediction_images[i]
            target = self.target_images[i]
            self.update_histogram(pred, target, histogram)

        histogram = histogram / len(self.prediction_images)
        np.save('../histograms/temporal_encoder.npy', histogram)


if __name__ == '__main__':
    predictions_dir = '../images/unet/unet-num_downs:8-ngf:128-down_type:strideconv-mode:sequence-1234-seq_model:tcn+temporal_encoder-num_levels_tcn:2-tcn_kernel_size:3-optim:amsgrad-lr:0.0001-clipping:5-resize:512,256-td:2/experiment_0/'
    targets_dir = '../images/targets/'

    helper = PixelMapGenerator(predictions_dir, targets_dir)
    helper.generate_histogram()
