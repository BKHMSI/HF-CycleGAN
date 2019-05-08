import os
import sys
import yaml 
import datetime
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from dataloader import Dataloader

class HFCycleGAN:
    def __init__(self, config):
        # Configure data loader
        self.dataloader = Dataloader(config)
        self.dataloader.load()

        self.save_path = os.path.join(config["paths"]["save"], config["run-title"])


        self.d_c_stream = load_model(config["paths"]["load-d-c-stream"], custom_objects={"InstanceNormalization": InstanceNormalization})
        self.d_r_stream = load_model(config["paths"]["load-d-r-stream"], custom_objects={"InstanceNormalization": InstanceNormalization})
        self.e_c_stream = load_model(config["paths"]["load-e-c-stream"], custom_objects={"InstanceNormalization": InstanceNormalization})

    def sample(self, index):
        r, c = 2, 2
        batch_size = 1
        
        imgs_A, imgs_B = self.dataloader.load_data(batch_size=batch_size, is_testing=True) 

        imgs_A_rand = self.dataloader.load_data_by_domain("A", batch_size=batch_size, is_testing=True)

        # Translate images to the other domain
        fake_B = self.d_c_stream.predict_on_batch(imgs_A)
        res_A  = self.d_r_stream.predict_on_batch(imgs_A_rand)

        fake_A = self.e_c_stream.predict_on_batch([imgs_B, res_A[0], res_A[1], res_A[2]])

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A]).squeeze()

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated']

        count = 0
        plt.switch_backend('agg')
        for batch in range(batch_size):
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[count])
                    axs[i,j].set_title(titles[j])
                    axs[i,j].axis('off')
                    count += 1
            fig.savefig(os.path.join(self.save_path, f"sample_{index}_{batch}.png"))
            plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HF-CycleGAN Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    gan = HFCycleGAN(config)
    gan.sample(1003)

