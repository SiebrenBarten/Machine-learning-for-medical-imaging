import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class KNNRegressor:
    def __init__(self, X, y, k=5, trainsplit=0.8):
        self.k = k
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.trainsplit = trainsplit
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def test_train_split(self):
        if self.trainsplit < 1.0:
            split_index = int(self.trainsplit * self.X.shape[0])
            self.X_train = self.X[:split_index, :]
            self.y_train = self.y[:split_index].reshape(-1)
            self.X_test = self.X[split_index:, :]
            self.y_test = self.y[split_index:].reshape(-1)
        else:
            raise ValueError("trainsplit should be a float between 0 and 1.")

    def normalize_train_test(self):
        mu = self.X_train.mean(axis=0)
        sigma = self.X_train.std(axis=0, ddof=0)
        sigma[sigma == 0.0] = 1.0
        self.X_train = (self.X_train - mu) / sigma
        self.X_test  = (self.X_test  - mu) / sigma

    def eucl_dist_matrix(self, X_test, X_train):
        D = sp.spatial.distance.cdist(X_test, X_train, 'euclidean')
        shape = D.shape
        if shape[0] != X_test.shape[0] or shape[1] != X_train.shape[0]:
            raise ValueError("The shape of the distance matrix is incorrect.")
        return D

    def row_topk_indices(self, D, k):
        idx_part = np.argpartition(D, kth=k-1, axis=1)[:, :k]
        rows = np.arange(D.shape[0])[:, None]
        d_small = D[rows, idx_part]
        order_in_k = np.argsort(d_small, axis=1)
        idx_sorted = idx_part[rows, order_in_k]
        return idx_sorted

    def predict(self, X_test=None, k=None):
        if X_test is None:
            if self.X_test is None:
                raise RuntimeError("No test set. Call test_train_split() first.")
            X_test = self.X_test
        if k is None:
            k = self.k
        D = self.eucl_dist_matrix(X_test, self.X_train)
        idx = self.row_topk_indices(D, k)
        neighbor_targets = self.y_train[idx]
        y_pred = np.mean(neighbor_targets, axis=1)
        return y_pred

    def score(self, k=None):
        y_pred = self.predict(k=k)
        mse = np.mean((y_pred - self.y_test) ** 2)
        return mse

    def evaluate_over_k(self, k_values, plot=True, title=None):
        train_mse, test_mse = [], []
        for k in k_values:
            # train mse (use train as both test and train)
            D_tr = self.eucl_dist_matrix(self.X_train, self.X_train)
            idx_tr = self.row_topk_indices(D_tr, k)
            y_pred_tr = np.mean(self.y_train[idx_tr], axis=1)
            train_mse.append(np.mean((y_pred_tr - self.y_train) ** 2))

            # test mse
            test_mse.append(self.score(k=k))

        if plot:
            plt.figure(figsize=(7,4.5))
            plt.plot(k_values, test_mse, marker='o', label='Test MSE')
            plt.plot(k_values, train_mse, marker='s', label='Train MSE')
            plt.xlabel('k'); plt.ylabel('Mean Squared Error')
            if title is None:
                title = f'KNN regression MSE vs k (trainsplit={self.trainsplit})'
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return np.array(train_mse), np.array(test_mse)
