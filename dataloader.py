import os
import sys 
import yaml
import scipy
import argparse
import numpy as np

from glob import glob

class Dataloader:
    def __init__(self, config):
        self.imsize = config["data"]["imsize"]
        self.imchannels = config["data"]["imchannels"]
        self.imshape = (self.imsize, self.imsize, self.imchannels)

        self.batch_size = config["train"]["batch-size"]
        self.base_path = config["paths"]["base"]
        self.dataset = config["data"]["dataset"]
        self.extension = config["data"]["extension"]

    def load(self):
        # load data from both domains
        self.train_A = glob(os.path.join(self.base_path, self.dataset, "trainA", f"*.{self.extension}"))
        self.train_B = glob(os.path.join(self.base_path, self.dataset, "trainB", f"*.{self.extension}"))

        self.test_A = glob(os.path.join(self.base_path, self.dataset, "testA", f"*.{self.extension}"))
        self.test_B = glob(os.path.join(self.base_path, self.dataset, "testB", f"*.{self.extension}"))

   
    def preprocess(self, data):
        data = np.array(data)/127.5 - 1.
        return data

    def load_batch(self, batch_size, is_testing=False):
        paths_A, paths_B = (self.train_A, self.train_B) if not is_testing else (self.test_A, self.test_B)

        self.n_batches = int(min(len(paths_A), len(paths_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        paths_A = np.random.choice(paths_A, total_samples, replace=False)
        paths_B = np.random.choice(paths_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = paths_A[i*batch_size:(i+1)*batch_size]
            batch_B = paths_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
    
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                if img_A.shape != self.imshape:
                    img_A = scipy.misc.imresize(img_A, self.imshape)

                if img_B.shape != self.imshape:
                    img_B = scipy.misc.imresize(img_B, self.imshape)

                if self.imchannels == 1:
                    img_A = np.reshape(img_A, (self.imsize, self.imsize, self.imchannels))
                    img_B = np.reshape(img_B, (self.imsize, self.imsize, self.imchannels))

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = self.preprocess(imgs_A)
            imgs_B = self.preprocess(imgs_B)

            yield imgs_A, imgs_B

    def load_data_by_domain(self, domain, batch_size=1, is_testing=False):
        if domain == "A":
            paths = self.train_A if not is_testing else self.test_A
        else:
            paths = self.train_B if not is_testing else self.test_B

        batch_images = np.random.choice(paths, size=batch_size, replace=False)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if img.shape != self.imshape:
                img = scipy.misc.imresize(img, self.imshape)
            if self.imchannels == 1:
                img = np.reshape(img, self.imshape)
            imgs.append(img)

        imgs = self.preprocess(imgs)
        return imgs 

    def load_data(self, batch_size=1, is_testing=False):
        paths_A = self.train_A if not is_testing else self.test_A
        paths_B = self.train_B if not is_testing else self.test_B

        batch_idxs = np.random.choice(len(paths_A), size=batch_size, replace=False)
        batch_images = zip(paths_A[batch_idxs], paths_B[batch_idxs])

        imgs_A, imgs_B = [], []
        for img_path_A, img_path_B in batch_images:

            img_A = self.imread(img_path_A)
            img_B = self.imread(img_path_B)
            
            if img_A.shape != self.imshape:
                img_A = scipy.misc.imresize(img_A, self.imshape)

            if img_B.shape != self.imshape:
                img_B = scipy.misc.imresize(img_B, self.imshape)

            if self.imchannels == 1:
                img_A = np.reshape(img_A, (self.imsize, self.imsize, self.imchannels))
                img_B = np.reshape(img_B, (self.imsize, self.imsize, self.imchannels))

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = self.preprocess(imgs_A)
        imgs_B = self.preprocess(imgs_B)

        return imgs_A, imgs_B

    def load_image(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.imshape)
        img = np.reshape(img, (1, self.imsize, self.imsize, self.imchannels))
        img = self.preprocess(img)
        return img

    def imread(self, path):
        return scipy.misc.imread(path, mode=("L" if self.imchannels == 1 else "RGB")).astype(np.float)

    def __str__(self):
        n_batches = int(min(len(self.train_A), len(self.train_B)) / self.batch_size)
        domain_A = "Domain A: {} training samples, {} testing samples".format(len(self.train_A), len(self.test_A))
        domain_B = "Domain B: {} training samples, {} testing samples".format(len(self.train_B), len(self.test_B))
        batches  = "Training with {} batches".format(n_batches)
        return "{}\n{}\n{}".format(domain_A, domain_B, batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HF-CycleGAN Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    dataloader = Dataloader(config)
    dataloader.load()
    print(dataloader)