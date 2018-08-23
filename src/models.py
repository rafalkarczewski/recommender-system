import numpy as np
from typing import List


class BlockCoordinateAscent:
    """
    Numpy implementation of block coordinate ascent optimization algorithm, which solves a
    recommendation system problem with binary feedback:
    Input:
        1. positive pairs (u, v) indicating interaction between user u and item v (negative pairs
    are unknown, i.e. if (u, v) is not given as positive it means it's negative or it wasn't tested)
        2. matrix of vector representation of items
        3. alpha - weight of a positive pair
        4. beta - weight of an unknown pair
        5. lambda_u - user embedding L2 regularization coefficient
        6. lambda_v - item adjustment L2 regularization coefficient
    Output:
        1. U - latent representation of the users
        2. eps - item representation "adjustment", i.e. V = item_vectors + eps is a latent
            representation of the items
        3. R^{hat} = U * V - prediction of a filled recommendation matrix
    """
    def __init__(
            self, train_ones: List[List[int]], item_vectors: np.array, alpha: float = 1.,
            beta: float = 0.01, lambda_u: float = 0.01, lambda_v: float = 0.01) -> None:
        """
        Initializes a ready to train model
        :param train_ones: indices of positive examples. train_ones is a list of length equal to
            the number of users. i-th element of train ones is a list of indices of items that
            are known to interact with the i-th user.
        :param item_vectors: matrix of vector representations of items
        :param alpha: weight of a positive pair
        :param beta: weight of an unknown pair
        :param lambda_u: user L2 regularization coefficient
        :param lambda_v: item L2 regularization coefficient
        """
        self.C = None  # weight matrix
        self.R = None  # known ratings matrix
        num_users = len(train_ones)
        num_items = item_vectors.shape[0]
        self._initialize(num_users, num_items, train_ones, alpha, beta)
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.latent_dim = item_vectors.shape[1]  # dimensionality of item vector representations
        self.u = np.random.normal(
            size=(num_users, self.latent_dim), scale=1 / lambda_u)  # user embedding
        eps = np.random.normal(
            size=(num_items, self.latent_dim), scale=1 / lambda_v)
        self.v = item_vectors + eps  # item embedding
        self.item_vectors = item_vectors

    def _initialize(
            self, num_users: int, num_items: int, train_ones: List[List[int]],
            alpha: float, beta: float) -> None:
        """
        Defines the cost matrix C and the binary recommendation matrix R
        """
        self.C = beta * np.ones(shape=(num_users, num_items))  # all values initialized with beta
        self.R = np.zeros(shape=(num_users, num_items))  # all values initialized with 0
        for i, positives in enumerate(train_ones):
            for positive in positives:
                self.C[i, positive] = alpha  # weights of positive examples set to alpha
                self.R[i, positive] = 1.0  # positive ratings

    def _u_block_update(self, c_i, r_i):
        a = np.dot(self.v.T, (c_i * self.v.T).T)
        a_inv = np.linalg.inv(a + self.lambda_u * np.identity(self.latent_dim))
        result = np.dot(self.v.T, (c_i * r_i.T).T)
        return np.dot(a_inv, result)[:, 0]

    def _v_block_update(self, c_j, r_j, theta_j):
        a = np.dot(self.u.T, (c_j * self.u.T).T)
        a_inv = np.linalg.inv(a + self.lambda_v * np.identity(self.latent_dim))
        result = np.dot(self.u.T, (c_j * r_j.T).T) + self.lambda_v * theta_j
        return np.dot(a_inv, result)[:, 0]

    def _u_update(self):
        result = np.zeros(shape=self.u.shape)
        for i in range(result.shape[0]):
            result[i, :] = self._u_block_update(
                self.C[i, :], self.R[i, :].reshape((-1, 1)))
        self.u = result

    def _v_update(self):
        result = np.zeros(shape=self.v.shape)
        for j in range(result.shape[0]):
            result[j, :] = self._v_block_update(
                self.C[:, j], self.R[:, j].reshape((-1, 1)),
                self.item_vectors[j, :].reshape((-1, 1))
            )
        self.v = result

    def _train_one_epoch(self):
        self._u_update()
        self._v_update()

    def fit(self, n_epochs: int) -> None:
        for _ in range(n_epochs):
            self._train_one_epoch()

    def predict(self) -> np.array:
        """
        Performs full prediction, i.e. predicts probabilities of interaction for all pairs of
        users and items
        :return: full matrix of predictions, i.e. matrix of shape (num_users, num_items),
        where (i, j)-th element of the matrix indicates probability that user i interacts with
        item j
        """
        return np.dot(self.u, self.v.T)
