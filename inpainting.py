from typing import Type

import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from dictionaries import Dictionary
from algorithm import ApproximateKSVD
from pursuits import Pursuit


class KSVDImageInpainting(ApproximateKSVD):


    def __init__(self, dictionary: Dictionary, pursuit: Type[Pursuit]):

        super().__init__(dictionary, pursuit, 0)
        self.multiplier = None
        self.n_iter = None
        self.image = None
        self.patch_size = None
        self.image_size = None
        self.image_root = None

    def inpaint(self, image, noisy, n_iter = 10, patch_size=16):

        if image.shape[0] != image.shape[1]:
            raise ValueError("Image must be square!")

        # set initial values
        #self.orig = orig
        self.image = image
        self.noisy = noisy
        self.sigma = 20
        self.noise_gain = 1.075
        self.multiplier = 0.5
        self.patch_size = patch_size
        self.n_iter = n_iter

        # compute further values
        self.image_size = image.shape[0]

        # prepare K-SVD
        patches = extract_patches_2d(self.image, (self.patch_size, self.patch_size))
        Y = np.array([p.reshape(self.patch_size**2) for p in patches]).T

        # iterate K-SVD
        for itr in range(self.n_iter):
            print("Num of iter: {}".format(itr))
            self.sparse_coding(Y)
            self.dictionary_update(Y)


        patches = extract_patches_2d(self.noisy, (self.patch_size, self.patch_size))
        Y_noisy = np.array([p.reshape(self.patch_size**2) for p in patches]).T
        self.sparse_coding(Y_noisy)

        out = np.zeros(image.shape)
        weight = np.zeros(image.shape)
        print("reconstructing")
        i = j = 0
        for k in range((self.image_size - self.patch_size + 1) ** 2):
            patch = np.reshape(np.matmul(self.dictionary.matrix, self.alphas[:, k]), (self.patch_size, self.patch_size))
            out[j:j + self.patch_size, i:i + self.patch_size] += patch
            weight[j:j + self.patch_size, i:i + self.patch_size] += 1
            if i < self.image_size - self.patch_size:
                i += 1
            else:
                i = 0
                j += 1
        out = np.divide(out + self.multiplier * self.image, weight + self.multiplier)

        return out, self.dictionary, self.alphas

    def train(self, orig, n_iter=15, patch_size=16):

        if orig.shape[0] != orig.shape[1]:
            raise ValueError("Image must be square!")

        # set initial values
        self.orig = orig
        self.n_iter = n_iter
        self.patch_size = patch_size
        self.iter = n_iter
        self.sigma = 20
        self.noise_gain = 1.075
        self.multiplier = 0.5

        # compute further values
        self.image_size = orig.shape[0]

        # prepare K-SVD
        patches = extract_patches_2d(self.orig, (self.patch_size, self.patch_size))
        Y = np.array([p.reshape(self.patch_size**2) for p in patches]).T
        print(Y.shape)

        # iterate K-SVD
        for itr in range(self.n_iter):
            print("Num of iter: {}".format(itr))
            self.sparse_coding(Y)
            self.dictionary_update(Y)

        return self.dictionary
