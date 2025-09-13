import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, X, y, k, trainsplit):
        self.k = k
        self.X = X  
        self.y = y
        self.trainsplit = trainsplit
        
    def test_train_split(self):
        """Splits the dataset into training and testing sets based on the trainsplit ratio."""
        if self.trainsplit < 1.0:
            split_index = int(self.trainsplit * self.X.shape[0])
            self.X_train = self.X[:split_index, :]
            self.y_train = self.y[:split_index].reshape(-1)  # Ensure 1D
            self.X_test = self.X[split_index:, :]
            self.y_test = self.y[split_index:].reshape(-1)   # Ensure 1D
        else:
            raise ValueError("trainsplit should be a float between 0 and 1.")
        
    def normalize_train_test(self):
        mu = self.X_train.mean(axis=0)
        sigma = self.X_train.std(axis=0, ddof=0)
        sigma[sigma == 0.0] = 1.0
        self.X_train = (self.X_train - mu) / sigma
        self.X_test  = (self.X_test  - mu) / sigma
            
    def eucl_dist_matrix(self, X_test, X_train):
        """
        Computes the Euclidean distance matrix between two sets of vectors A and B. The result
        should be a matrix of shape (A.shape[0], B.shape[0]).
        """
        D = sp.spatial.distance.cdist(X_test, X_train, 'euclidean')
        shape = D.shape
        if shape[0] != X_test.shape[0] or shape[1] != X_train.shape[0]:
            raise ValueError("The shape of the distance matrix is incorrect.")
        return D
    
    def row_topk_indices(self, D, k):
        """Return indices of k smallest per row, sorted by distance."""
        idx_part = np.argpartition(D, kth=k-1, axis=1)[:, :k]      # (n_test, k)
        rows = np.arange(D.shape[0])[:, None]
        d_small = D[rows, idx_part]
        order_in_k = np.argsort(d_small, axis=1)
        idx_sorted = idx_part[rows, order_in_k]
        return idx_sorted

    # --------------------- voting & predict (classificatie) ---------------------
    def vote_majority(self, neigh_labels):
        """neigh_labels: (n_test, k) -> (n_test,)"""
        n = neigh_labels.shape[0]
        C = int(neigh_labels.max()) + 1
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            out[i] = np.bincount(neigh_labels[i], minlength=C).argmax()
        return out

    def predict(self, X_test=None, k=None, weighted=False):
        """Predict labels for X_test (defaults to held-out test set)."""
        if X_test is None:
            if self.X_test is None:
                raise RuntimeError("No test set. Call test_train_split() first.")
            X_test = self.X_test
        if k is None:
            k = self.k

        D = self.eucl_dist_matrix(X_test, self.X_train)          # (n_test, n_train)
        idx = self.row_topk_indices(D, k)                        # (n_test, k)
        neigh_labels = self.y_train[idx]
        return self.vote_majority(neigh_labels)

    def score(self, k=None, weighted=False):
        """Accuracy on the held-out test set."""
        preds = self.predict(k=k, weighted=weighted)
        return np.mean(preds == self.y_test)
    
    def evaluate_over_k(self, k_values, plot=True, title=None):
        train_acc, test_acc = [], []
        for k in k_values:
            D_tr = self.eucl_dist_matrix(self.X_train, self.X_train)
            idx_tr = self.row_topk_indices(D_tr, k)              # bevat self-neighbor
            y_pred_tr = self.vote_majority(self.y_train[idx_tr])
            train_acc.append(np.mean(y_pred_tr == self.y_train))
            test_acc.append(self.score(k=k, weighted=False))

        if plot:
            plt.figure(figsize=(7,4.5))
            plt.plot(k_values, test_acc, marker='o', label='Test accuracy')
            plt.plot(k_values, train_acc, marker='s', label='Train accuracy')
            plt.xlabel('k'); plt.ylabel('Accuracy')
            if title is None:
                title = f'KNN accuracy vs k (trainsplit={self.trainsplit})'
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
        return np.array(train_acc), np.array(test_acc)


# ===================== k-NN REGRESSOR =====================

class KNNRegressor(KNN):
    """k-NN regression: computes the (weighted) average of neighbor targets."""
    def predict(self, X_test=None, k=None, weighted=False):
        if X_test is None:
            if self.X_test is None:
                raise RuntimeError("No test set. Call test_train_split() first.")
            X_test = self.X_test
        if k is None:
            k = self.k

        D = self.eucl_dist_matrix(X_test, self.X_train)          # (n_test, n_train)
        idx = self.row_topk_indices(D, k)                        # (n_test, k)
        neigh_targets = self.y_train[idx].astype(float)

        if weighted:
            eps = 1e-12
            rows = np.arange(D.shape[0])[:, None]
            w = 1.0 / (D[rows, idx] + eps)                       # inverse-distance weights
            preds = (neigh_targets * w).sum(axis=1) / w.sum(axis=1)
        else:
            preds = neigh_targets.mean(axis=1)
        return preds

    def mse(self, y_true, y_pred):
        """Mean Squared Error (MSE)."""
        return float(((y_true - y_pred) ** 2).mean())

    def r2(self, y_true, y_pred):
        """Coefficient of determination R^2."""
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot

    def score(self, k=None, weighted=False):
        """Return -MSE on the held-out test set (higher is better, for consistency with accuracy)."""
        preds = self.predict(k=k, weighted=weighted)
        return -self.mse(self.y_test, preds)

    def evaluate_over_k(self, k_values, weighted=False, plot=True, title=None):
        """
        Evaluate performance for a list of k values.
        Returns (train_mse, test_mse).
        
        Note: for training MSE we exclude the self-neighbor (set diagonal = inf),
        otherwise k=1 would trivially give zero error.
        """
        train_mse, test_mse = [], []

        # Train-vs-train distance matrix with diagonal excluded
        D_tr = self.eucl_dist_matrix(self.X_train, self.X_train)
        D_tr = D_tr.copy()
        np.fill_diagonal(D_tr, np.inf)
        ntr = D_tr.shape[0]
        rows = np.arange(ntr)[:, None]

        for k in k_values:
            idx_tr = self.row_topk_indices(D_tr, k)
            neigh_targets_tr = self.y_train[idx_tr].astype(float)

            if weighted:
                eps = 1e-12
                w_tr = 1.0 / (D_tr[rows, idx_tr] + eps)
                preds_tr = (neigh_targets_tr * w_tr).sum(axis=1) / w_tr.sum(axis=1)
            else:
                preds_tr = neigh_targets_tr.mean(axis=1)
            train_mse.append(self.mse(self.y_train, preds_tr))

            preds_te = self.predict(self.X_test, k=k, weighted=weighted)
            test_mse.append(self.mse(self.y_test, preds_te))

        train_mse = np.array(train_mse)
        test_mse = np.array(test_mse)

        if plot:
            plt.figure(figsize=(7,4.5))
            plt.plot(k_values, test_mse, marker='o', label='Test MSE')
            plt.plot(k_values, train_mse, marker='s', label='Train MSE')
            plt.xlabel('k'); plt.ylabel('Mean Squared Error (MSE)')
            if title is None:
                title = f'KNN regression MSE vs k (trainsplit={self.trainsplit})'
            plt.title(title)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return train_mse, test_mse

