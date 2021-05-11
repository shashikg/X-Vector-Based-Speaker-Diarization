# Code taken fron https://github.com/wq2012/SpectralCluster
# Modified to suit our need

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from scipy.ndimage import gaussian_filter
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class eigenGap:
    def __init__(self, min_clusters=1, max_clusters=100, p_percentile=0.9, 
               gaussian_blur_sigma=2, stop_eigenvalue=1e-2,
               thresholding_soft_multiplier=0.01, thresholding_with_row_max=True):

        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.p_percentile = p_percentile
        self.thresholding_soft_multiplier = thresholding_soft_multiplier
        self.thresholding_with_row_max = thresholding_with_row_max
        self.stop_eigenvalue = stop_eigenvalue
        self.refinement_sequence = ["CropDiagonal",
                                    "GaussianBlur",
                                    "RowWiseThreshold",
                                    "Symmetrize",
                                    "Diffuse",
                                    "RowWiseNormalize",
                                    "Symmetrize"]

    def _get_refinement_operator(self, name):
        if name == "CropDiagonal":
            return CropDiagonal()
        elif name == "GaussianBlur":
            return GaussianBlur(self.gaussian_blur_sigma)
        elif name == "RowWiseThreshold":
            return RowWiseThreshold(
                self.p_percentile,
                self.thresholding_soft_multiplier,
                self.thresholding_with_row_max)
        elif name == "Symmetrize":
            return Symmetrize()
        elif name == "Diffuse":
            return Diffuse()
        elif name == "RowWiseNormalize":
            return RowWiseNormalize()

    def find(self, X):
        affinity = self.compute_affinity_matrix(X)
        for refinement_name in self.refinement_sequence:
            op = self._get_refinement_operator(refinement_name)
            affinity = op.refine(affinity)

        eigenvalues = self.compute_sorted_eigenvalues(affinity)
        k = self.compute_number_of_clusters(eigenvalues, self.max_clusters, self.stop_eigenvalue)

        return max(k, self.min_clusters)

    def compute_affinity_matrix(self, X):
        l2_norms = np.linalg.norm(X, axis=1)
        X_normalized = X / l2_norms[:, None]
        cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
        affinity = (cosine_similarities + 1.0) / 2.0
        return affinity

    def compute_sorted_eigenvalues(self, A):
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues = eigenvalues.real
        index_array = np.argsort(-eigenvalues)
        w = eigenvalues[index_array]
        return w

    def compute_number_of_clusters(self, eigenvalues, max_clusters=None, stop_eigenvalue=1e-2):
        max_delta = 0
        max_delta_index = 0
        range_end = len(eigenvalues)
        if max_clusters and max_clusters + 1 < range_end:
            range_end = max_clusters + 1
        for i in range(1, range_end):
            if eigenvalues[i - 1] < stop_eigenvalue:
                break
            delta = eigenvalues[i - 1] / eigenvalues[i]
            if delta > max_delta:
                max_delta = delta
                max_delta_index = i
        return max_delta_index
        

class AffinityRefinementOperation(metaclass=abc.ABCMeta):
    def check_input(self, X):
        """Check the input to the refine() method.
        Args:
            X: the input to the refine() method
        Raises:
            TypeError: if X has wrong type
            ValueError: if X has wrong shape, etc.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        shape = X.shape
        if len(shape) != 2:
            raise ValueError("X must be 2-dimensional")
        if shape[0] != shape[1]:
            raise ValueError("X must be a square matrix")

    @abc.abstractmethod
    def refine(self, X):
        """Perform the refinement operation.
        Args:
            X: the affinity matrix, of size (n_samples, n_samples)
        Returns:
            a matrix of the same size as X
        """
        pass


class CropDiagonal(AffinityRefinementOperation):
    """Crop the diagonal.
    Replace diagonal element by the max non-diagonal value of row.
    After this operation, the matrix has similar properties to a standard
    Laplacian matrix.
    This also helps to avoid the bias during Gaussian blur and normalization.
    """

    def refine(self, X):
        self.check_input(X)
        Y = np.copy(X)
        np.fill_diagonal(Y, 0.0)
        di = np.diag_indices(Y.shape[0])
        Y[di] = Y.max(axis=1)
        return Y


class GaussianBlur(AffinityRefinementOperation):
    """Apply Gaussian blur."""

    def __init__(self, sigma=1):
        self.sigma = sigma

    def refine(self, X):
        self.check_input(X)
        return gaussian_filter(X, sigma=self.sigma)


class RowWiseThreshold(AffinityRefinementOperation):
    """Apply row wise thresholding."""

    def __init__(self,
                 p_percentile=0.95,
                 thresholding_soft_multiplier=0.01,
                 thresholding_with_row_max=False):
        self.p_percentile = p_percentile
        self.multiplier = thresholding_soft_multiplier
        self.thresholding_with_row_max = thresholding_with_row_max

    def refine(self, X):
        self.check_input(X)
        Y = np.copy(X)

        if self.thresholding_with_row_max:
            # row_max based thresholding
            row_max = Y.max(axis=1)
            row_max = np.expand_dims(row_max, axis=1)
            is_smaller = Y < (row_max * self.p_percentile)
        else:
            # percentile based thresholding
            row_percentile = np.percentile(Y, self.p_percentile * 100, axis=1)
            row_percentile = np.expand_dims(row_percentile, axis=1)
            is_smaller = Y < row_percentile

        Y = (Y * np.invert(is_smaller)) + (Y * self.multiplier * is_smaller)
        return Y


class Symmetrize(AffinityRefinementOperation):
    """The Symmetrization operation."""

    def refine(self, X):
        self.check_input(X)
        return np.maximum(X, np.transpose(X))


class Diffuse(AffinityRefinementOperation):
    """The diffusion operation."""

    def refine(self, X):
        self.check_input(X)
        return np.matmul(X, np.transpose(X))


class RowWiseNormalize(AffinityRefinementOperation):
    """The row wise max normalization operation."""

    def refine(self, X):
        self.check_input(X)
        Y = np.copy(X)
        row_max = Y.max(axis=1)
        Y /= np.expand_dims(row_max, axis=1)
        return Y