import os
import sys
import yaml 
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from models import Models
from dataloader import Dataloader

class HFCycleGAN:
    def __init__(self, config):
        # Input shape
        self.imsize = config["data"]["imsize"]
        self.imchannels = config["data"]["imchannels"]
        self.imshape = (self.imsize, self.imsize, self.imchannels)
        
        # Configure data loader
        self.dataloader = Dataloader(config)
        self.dataloader.load()

        # Configure models
        models = Models(config)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.imsize / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Loss weights
        self.lambda_cycle = config["train"]["lambda-cycle-loss"] # Cycle-consistency loss
        self.lambda_adv = 0.1 * self.lambda_cycle 
        self.lambda_res = 0.01 * self.lambda_cycle   

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_C = models.build_discriminator()
        self.d_V = models.build_discriminator()

        self.d_C.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_V.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Build streams
        self.d_c_stream = models.build_generator(network="disentangler", stream="C")
        self.d_r_stream = models.build_generator(network="disentangler", stream="R")

        self.e_c_stream = models.build_generator(network="entangler", stream="C")

        # Build and compile disentangler
        self.disentangler = self.build_disentangler()
        self.disentangler.compile(loss=['mse', 'mae'], 
                                  loss_weights=[self.lambda_adv, self.lambda_cycle], optimizer=optimizer)

        # Build and compile entangler
        self.entangler = self.build_entangler()
        self.entangler.compile(loss=["mse", "mae", "mae", "mae", "mae"],
                               loss_weights=[self.lambda_adv, self.lambda_cycle, self.lambda_res, self.lambda_res, self.lambda_res], optimizer=optimizer)


    def build_disentangler(self):

        # Input image 
        image_V = Input(shape=self.imshape) 
        
        # Translate image
        self.d_c_stream.trainable = True
        fake_C = self.d_c_stream(image_V)

        # Obtain feature-maps (residuals)
        self.d_r_stream.trainable = True
        residuals_V = self.d_r_stream(image_V)

        # Freeze discriminator
        self.d_C.trainable = False

        # Trick discriminator into thinking it is real
        valid_C = self.d_C(fake_C)
 
        # Freeze entangler
        self.e_c_stream.trainable = False 

        # Reconstruct image using entangler
        reconstr_V = self.e_c_stream([fake_C, residuals_V[0], residuals_V[1], residuals_V[2]])

        return Model(image_V, [valid_C, reconstr_V])

    def build_entangler(self):

        # Input image 
        image_C = Input(shape=self.imshape) 
    
        # Random input image from other domain
        image_V = Input(shape=self.imshape) 
        
        # Freeze disentangler's R stream
        self.d_r_stream.trainable = False 
        residuals_V = self.d_r_stream(image_V)

        # Translate image given residuals
        self.e_c_stream.trainable = True
        fake_V = self.e_c_stream([image_C, residuals_V[0], residuals_V[1], residuals_V[2]])

        # Freeze discriminator
        self.d_V.trainable = False 

        # Trick discriminator into thinking it is real
        valid_V = self.d_V(fake_V)

        # Freeze disentangler
        self.d_c_stream.trainable = False 
        
        # Reconstruct image and residuals using disentangler
        reconstr_C = self.d_c_stream(fake_V)
        residuals_C = self.d_r_stream(fake_V)

        return Model([image_C, image_V], [valid_V, reconstr_C, residuals_C[0], residuals_C[1], residuals_C[2]])


    def train(self, epochs, batch_size, save_interval):
        pass 

    def sample(self, index):
        pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HF-CycleGAN Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    gan = HFCycleGAN(config)
    gan.train(epochs=config["train"]["epochs"], batch_size=config["train"]["batch-size"], save_interval=config["train"]["save-interval"])

