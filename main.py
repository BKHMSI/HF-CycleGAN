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

        self.ressize = config["train"]["res-size"]
        self.resshape = (self.ressize, self.ressize, config["train"]["res-filters"])

        self.savepath = config["paths"]["save"]
        
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
        self.d_A = models.build_discriminator()
        self.d_B = models.build_discriminator()

        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

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
        image_A = Input(shape=self.imshape) 
        
        # Translate image
        self.d_c_stream.trainable = True
        fake_B = self.d_c_stream(image_A)

        # Obtain feature-maps (residuals)
        self.d_r_stream.trainable = True
        residuals_A = self.d_r_stream(image_A)

        # Freeze discriminator
        self.d_B.trainable = False

        # Trick discriminator into thinking it is real
        valid_B = self.d_B(fake_B)
 
        # Freeze entangler
        self.e_c_stream.trainable = False 

        # Reconstruct image using entangler
        reconstr_A = self.e_c_stream([fake_B, residuals_A[0], residuals_A[1], residuals_A[2]])

        return Model(image_A, [valid_B, reconstr_A])

    def build_entangler(self):

        # Input image 
        image_B = Input(shape=self.imshape) 

        residuals_A_0 = Input(shape=self.resshape)
        residuals_A_1 = Input(shape=self.resshape)
        residuals_A_2 = Input(shape=self.resshape)

        # Translate image given residuals
        self.e_c_stream.trainable = True
        fake_A = self.e_c_stream([image_B, residuals_A_0, residuals_A_1, residuals_A_2])

        # Freeze discriminator
        self.d_A.trainable = False 

        # Trick discriminator into thinking it is real
        valid_A = self.d_A(fake_A)

        # Freeze disentangler
        self.d_c_stream.trainable = False 
        self.d_r_stream.trainable = False 

        # Reconstruct image and residuals using disentangler
        reconstr_B = self.d_c_stream(fake_A)
        residuals_A = self.d_r_stream(fake_A)

        return Model([image_B, residuals_A_0, residuals_A_1, residuals_A_2], [valid_A, reconstr_B, residuals_A[0], residuals_A[1], residuals_A[2]])


    def train(self, epochs, batch_size, save_interval):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_idx, (imgs_A, imgs_B) in enumerate(self.dataloader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.d_c_stream.predict_on_batch(imgs_A)
                res_A  = self.d_r_stream.predict_on_batch(imgs_A)

                fake_A = self.e_c_stream.predict_on_batch([imgs_B, res_A[0], res_A[1], res_A[2]])

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # --------------------------------
                #  Train Disentangler & Entangler
                # --------------------------------

                dis_loss = self.disentangler.train_on_batch(imgs_A, [valid, imgs_A])
                ent_loss = self.entangler.train_on_batch([imgs_B, res_A[0], res_A[1], res_A[2]], [valid, imgs_B, res_A[0], res_A[1], res_A[2]])

                res_reconstr_loss = (ent_loss[2] + ent_loss[3] + ent_loss[4]) / 3.

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{self.dataloader.n_batches} [D Loss: {d_loss[0]}, acc: {d_loss[1]*100:.4f}] [Adv Loss: {dis_loss[0]}, {ent_loss[0]}] [Cycle Loss: {dis_loss[1]}, {ent_loss[1]}] [Res Loss: {res_reconstr_loss}] [Time: {elapsed_time}]", end="\r")

            if (epoch+1) % save_interval == 0:
                self.sample(epoch+1)
                # self.disentangler.save(os.path.join(self.save_path, ""))
                # self.entangler.save(os.path.join(self.save_path, ""))


    def sample(self, index):
        r, c = 2, 2
        
        imgs_A, imgs_B = self.dataloader.load_data(batch_size=1, is_testing=True) 

        # Translate images to the other domain
        fake_B = self.d_c_stream.predict_on_batch(imgs_A)
        res_A  = self.d_r_stream.predict_on_batch(imgs_A)

        fake_A = self.e_c_stream.predict_on_batch([imgs_B, res_A[0], res_A[1], res_A[2]])

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A]).squeeze()

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated']

        plt.switch_backend('agg')
        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[i*c+j])
                axs[i,j].set_title(titles[j])
                axs[i,j].axis('off')
                
        fig.savefig(os.path.join(self.savepath, f"sample_{index}.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HF-CycleGAN Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    gan = HFCycleGAN(config)
    gan.train(epochs=config["train"]["epochs"], batch_size=config["train"]["batch-size"], save_interval=config["train"]["save-interval"])

