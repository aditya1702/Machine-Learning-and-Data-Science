import pandas as pd
import numpy as np
import math
import statistics
from sklearn.datasets import load_digits, load_iris, load_boston, load_breast_cancer
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
import sklearn
from copy import deepcopy
from sklearn.metrics import pairwise_distances


class GaussianMixtureModel():

    def __init__(self, k = 5, max_iters = 100, random_seed = 42, reg_covar = 1e-6, verbose = True):
        self.k = k # number of Gaussians
        self.max_iters = max_iters
        self.reg_covar = reg_covar
        self.verbose = verbose

        # Set random seed
        np.random.seed(random_seed)

    def _initialise_prams(self, X):

        # Get initial clusters using Kmeans
        kmeans = KMeans(k = self.k, max_iters = 500)
        kmeans.fit(X)
        kmeans_preds = kmeans.predict(X)

        N, col_length = X.shape
        mixture_labels = np.unique(kmeans_preds)
        initial_mean = np.zeros((self.k, col_length))
        initial_cov = np.zeros((self.k, col_length, col_length))
        initial_pi = np.zeros(self.k)

        for index, mixture_label in enumerate(mixture_labels):
            mixture_indices = (kmeans_preds == mixture_label)
            Nk = X[mixture_indices].shape[0]

            # Initial pi
            initial_pi[index] = Nk/N

            # Intial mean
            initial_mean[index, :] = np.mean(X[mixture_indices], axis = 0)

            # Initial covariance
            de_meaned = X[mixture_indices] - initial_mean[index, :]
            initial_cov[index] = np.dot(initial_pi[index] * de_meaned.T, de_meaned) / Nk
        assert np.sum(initial_pi) == 1
        return initial_pi, initial_mean, initial_cov

    def _compute_loss(self, X):
        N = X.shape[0]
        loss = np.zeros((N, self.k))

        for k in range(self.k):
            dist = mvn(self.mu[k], self.cov[k], allow_singular = True)
            loss[:, k] = self.gamma[:, k] * (np.log(self.pi[k] + 1e-5) + \
                                                  dist.logpdf(X) - np.log(self.gamma[:, k] + 1e-6))
        loss = np.sum(loss)
        return loss

    def _E(self, X):
        '''
        Find the responsibilties (gamma) for each sample in X and each component.
        '''

        row_length, col_length = X.shape
        self.gamma = np.zeros((row_length, self.k))

        # Calculate gamma
        for k in range(self.k):
            # Regularise the covariance to prevent singular matrix
            self.cov[k].flat[::col_length + 1] += self.reg_covar
            self.gamma[:, k] = self.pi[k] * mvn.pdf(X, self.mu[k, :], self.cov[k])

        # Normalise gamma
        self.gamma = self.gamma/np.sum(self.gamma, axis = 1, keepdims = True)

    def _M(self, X):
        N = X.shape[0]
        col_length = X.shape[1]

        Nk = self.gamma.sum(axis = 0)[:, np.newaxis]

        # Update pi
        self.pi = Nk/N

        # Update mu
        self.mu = (self.gamma.T @ X)/Nk

        # Update covariance
        for k in range(self.k):
            x = X - self.mu[k, :] # (N x d)

            gamma_diag = np.diag(self.gamma[:, k])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            cov_k = x.T * gamma_diag * x
            self.cov[k] = (cov_k) / Nk[k]

    def fit(self, X):

        # Initialise parameters
        self.pi, self.mu, self.cov = self._initialise_prams(X)

        iterations = 0
        while iterations <= self.max_iters:
            iterations += 1

            # Expectation Step
            self._E(X)

            # Maximisation Step
            self._M(X)

            # Get the loss
            loss = self._compute_loss(X)
            if self.verbose:
                print("Epoch - ", str(iterations), " Loss - ", str(loss))

    def predict_proba(self, X):
        labels = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            self.cov[k].flat[::X.shape[1] + 1] += self.reg_covar
            labels[:, k] = self.pi[k] * mvn.pdf(X, self.mu[k, :], self.cov[k])

        # Normalise
        labels = labels/np.sum(labels, axis = 1, keepdims = True)
        return labels

    def predict(self, X):
        labels = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            self.cov[k].flat[::X.shape[1] + 1] += self.reg_covar
            labels[:, k] = self.pi[k] * mvn.pdf(X, self.mu[k, :], self.cov[k])

        # Normalise
        labels = labels/np.sum(labels, axis = 1, keepdims = True)
        labels  = labels.argmax(axis = 1)
        return labels

    def sample(self, n_samples = 1):
        n_samples_comp = np.random.multinomial(n_samples, self.pi.reshape(1, -1)[0])
        X = np.vstack([
                np.random.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    self.mu, self.cov, n_samples_comp)])
        y = np.concatenate([np.full(sample, j, dtype = int) for j, sample in enumerate(n_samples_comp)])
        return X, y
