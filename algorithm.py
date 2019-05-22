from typing import Type

import numpy as np

from dictionaries import Dictionary
from pursuits import Pursuit


class KSVD:


    def __init__(self, dictionary: Dictionary, pursuit: Type[Pursuit], sparsity: int, noise_gain=None, sigma=None):
        self.dictionary = Dictionary(dictionary.matrix)
        self.alphas = None
        self.pursuit = pursuit
        self.sparsity = sparsity
        self.noise_gain = noise_gain
        self.sigma = sigma
        self.original_image = None
        self.sparsity_values = []
        self.mses = []
        self.ssims = []
        self.psnrs = []
        self.iter = None

    def sparse_coding(self, Y: np.ndarray):
        if self.noise_gain and self.sigma:
            p = self.pursuit(self.dictionary, tol=(self.noise_gain * self.sigma))
        else:
            p = self.pursuit(self.dictionary, sparsity=self.sparsity)
        self.alphas = p.fit(Y)

    def dictionary_update(self, Y: np.ndarray):
        # iterate rows
        D = self.dictionary.matrix
        n, K = D.shape
        for k in range(K):
            wk = np.nonzero(self.alphas[k, :])[0]
            if len(wk) == 0:
                continue
            E = Y[:, wk]
            for j in range(K):
                if j != k:
                    E += -np.outer(D[:, j], self.alphas[j, wk])
            U, s, Vh = np.linalg.svd(E)
            D[:, k] = U[:, 0]
            self.alphas[k, wk] = s[0] * Vh[0, :]
        self.dictionary = Dictionary(D)

    def fit(self, Y: np.ndarray, iter: int):
        for i in range(iter):
            self.sparse_coding(Y)
            self.dictionary_update(Y)
        return self.dictionary, self.alphas


class ApproximateKSVD(KSVD):

    def dictionary_update(self, Y: np.ndarray):
        # iterate rows
        D = self.dictionary.matrix
        n, K = D.shape
        for k in range(K):
            wk = np.nonzero(self.alphas[k, :])[0]
            if len(wk) == 0:
                continue
            D[:, k] = 0
            g = np.transpose(self.alphas)[wk, k]
            d = np.matmul(Y[:, wk], g) - np.matmul(D, self.alphas[:, wk]).dot(g)
            d = d / np.linalg.norm(d)
            g = np.matmul(Y[:, wk].T, d) - np.transpose(np.matmul(D, self.alphas[:, wk])).dot(d)
            D[:, k] = d
            self.alphas[k, wk] = g.T
        self.dictionary = Dictionary(D)
