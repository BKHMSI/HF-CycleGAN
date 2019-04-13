import os
import sys 
import yaml
import scipy
import argparse
import numpy as np

from glob import glob
from skimage import transform


class Dataloader:
    def __init__(self, config):
        self.config = config 
        self.imsize = (config["imsize"], config["imsize"])
        self.extension = config["extension"]

    def load(self):
        # load data from both domains
        paths_A = glob(os.path.join(self.config["base"], self.config["domain_A"], "*", f"*.{self.extension}"))
        paths_B = glob(os.path.join(self.config["base"], self.config["domain_B"], "*", f"*.{self.extension}"))

        # shuffle data
        np.random.shuffle(paths_A)
        np.random.shuffle(paths_B)

        paths_A = np.array(paths_A)
        paths_B = np.array(paths_B)

        # split data 
        split_idx = int(len(paths_A) * (1-self.config["val_split"]))
        self.train_A = paths_A[:split_idx]
        self.val_A   = paths_A[split_idx:]
        split_idx = int(len(paths_B) * (1-self.config["val_split"]))
        self.train_B = paths_B[:split_idx]
        self.val_B   = paths_B[split_idx:]

    def preprocess(self, data):
        data = np.array(data)/127.5 - 1.
        return data

    def load_batch(self, batch_size, is_testing=False):
        paths_A, paths_B = (self.train_A, self.train_B) if not is_testing else (self.val_A, self.val_B)

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

                img_A = scipy.misc.imresize(img_A, self.imsize)
                img_B = scipy.misc.imresize(img_B, self.imsize)

                img_A = np.reshape(img_A, (self.imsize[0], self.imsize[0], 1))
                img_B = np.reshape(img_B, (self.imsize[0], self.imsize[0], 1))

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
            paths = self.train_A if not is_testing else self.val_A
        else:
            paths = self.train_B if not is_testing else self.val_B

        batch_images = np.random.choice(paths, size=batch_size, replace=False)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            img = scipy.misc.imresize(img, self.imsize)
            img = np.reshape(img, (self.imsize[0], self.imsize[0], 1))
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.
        return imgs 

    def load_data(self, batch_size=1, is_testing=False):
        paths_A = self.train_A if not is_testing else self.val_A
        paths_B = self.train_B if not is_testing else self.val_B

        batch_idxs = np.random.choice(len(paths_A), size=batch_size, replace=False)
        batch_images = zip(paths_A[batch_idxs], paths_B[batch_idxs])

        imgs_A, imgs_B = [], []
        for img_path_A, img_path_B in batch_images:

            img_A = self.imread(img_path_A)
            img_B = self.imread(img_path_B)

            img_A = transform.resize(img_A, self.imsize)
            img_B = transform.resize(img_B, self.imsize)

            img_A = np.reshape(img_A, (self.imsize[0], self.imsize[0], 1))
            img_B = np.reshape(img_B, (self.imsize[0], self.imsize[0], 1))

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = self.preprocess(imgs_A)
        imgs_B = self.preprocess(imgs_B)

        return imgs_A, imgs_B

    def load_image(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.imsize)
        img = np.reshape(img, (1, self.imsize[0], self.imsize[0], 1))
        img = self.preprocess(img)
        return img

    def imread(self, path):
        return scipy.misc.imread(path, mode='L').astype(np.float)

    def __str__(self):
        n_batches = int(min(len(self.train_A), len(self.train_B)) / self.config["batch_size"])
        domain_A = "Domain A: {} training samples, {} testing samples".format(len(self.train_A), len(self.val_A))
        domain_B = "Domain B: {} training samples, {} testing samples".format(len(self.train_B), len(self.val_B))
        batches  = "Training with {} batches".format(n_batches)
        return "{}\n{}\n{}".format(domain_A, domain_B, batches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HF-CycleGAN Paramaters')
    parser.add_argument('-c', '--config',  type=str, default="config.yaml", help='path of config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.load(file)

    dataloader = Dataloader(config)
    dataloader.load()
    print(dataloader)