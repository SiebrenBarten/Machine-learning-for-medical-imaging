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
        """
        Splits the dataset into training and testing sets based on the trainsplit ratio. The train and test data are stored in self.X_train, self.y_train, self.X_test, self.y_test.
        """
        if self.trainsplit < 1.0:
            split_index = int(self.trainsplit * self.X.shape[0])
            self.X_train = self.X[:split_index, :]
            self.y_train = self.y[:split_index].reshape(-1)  # ensure it's a 1D array
            self.X_test = self.X[split_index:, :]
            self.y_test = self.y[split_index:].reshape(-1)  # ensure it's a 1D array
        else:
            raise ValueError("trainsplit should be a float between 0 and 1.")
        
    def normalize_train_test(self):
        """
        Calibrates the training and test set to have zero mean and unit variance. Formula: X_normalized = (X - mu) / sigma
        """
        mu = self.X_train.mean(axis=0)
        sigma = self.X_train.std(axis=0, ddof=0)
        sigma[sigma == 0.0] = 1.0
        self.X_train = (self.X_train - mu) / sigma
        self.X_test  = (self.X_test  - mu) / sigma
            
    def eucl_dist_matrix(self, X_test, X_train):
        """
        Computes the Euclidean distance matrix between the X_test vector and the X_train vector. The result should be a matrix D of shape (X_test.shape[0], X_train.shape[0]).
        """
        # compute the distance matrix D
        D = sp.spatial.distance.cdist(X_test, X_train, 'euclidean')
        
        # check shape of the D matrix
        shape = D.shape
        if shape[0] != X_test.shape[0] or shape[1] != X_train.shape[0]:
            raise ValueError("The shape of the distance matrix is incorrect.")
        return D
    
    def select_k_neighbors(self, D, k):
        """
        Return indices of k smallest per row, sorted by distance.
        """
        # depending on input k, the indices of the k smallest distances are are stored in a n_train x k matrix
        idx_mat = np.argpartition(D, kth=k-1, axis=1)[:, :k]
        
        # sort the k indices by distance
        rows = np.arange(D.shape[0])[:, None]
        dist_mat = D[rows, idx_mat]
        order_in_k = np.argsort(dist_mat, axis=1)
        idx_sorted = idx_mat[rows, order_in_k]
        return idx_sorted

    def vote_majority(self, neigh_labels):
        """
        Vote for the most common class among the k neighbors. Note that this works optimal if k is odd.
        """
        n_rows = neigh_labels.shape[0]
        n_classes = int(neigh_labels.max()) + 1
        
        # for each row, count the occurrences of each class (in this case 1 or 0) and return the class with the highest count
        pred_array = np.empty(n_rows, dtype=np.int64)
        for i in range(n_rows):
            pred_array[i] = np.bincount(neigh_labels[i], minlength=n_classes).argmax()
        
        return pred_array

    def predict(self, X_test=None, k=None):
        """
        Predict labels for X_test (defaults to held-out test set).
        """
        # add option to predict on arbitrary data X_test
        if X_test is None:
            if self.X_test is None:
                raise RuntimeError("No test set. Call test_train_split() first.")
            X_test = self.X_test
        
        # add option to predict with arbitrary k
        if k is None:
            k = self.k

        # compute distance matrix
        D = self.eucl_dist_matrix(X_test, self.X_train)
        
        # find the k nearest neighbors for every test sample
        idx = self.select_k_neighbors(D, k)
        
        # get the labels of the training samples of the k nearest neighbors
        neigh_labels = self.y_train[idx]

        # vote for the most common class among the k neighbors
        y_pred = self.vote_majority(neigh_labels)
        
        return y_pred

    def score(self, k=None):
        """
        Accuracy on the held-out test set.
        """
        preds = self.predict(k=k)
        return np.mean(preds == self.y_test)
    
def plot_knn_accuracy(X, y, k_values, trainsplit):
    train_acc, test_acc = [], []

    for k in k_values:
        # Create a new KNN instance for each k to avoid side effects
        knn = KNN(X, y, k, trainsplit)
        knn.test_train_split()
        knn.normalize_train_test()
        train_preds = knn.predict(knn.X_train, k)
        test_preds = knn.predict(knn.X_test, k)
        train_acc.append(np.mean(train_preds == knn.y_train))
        test_acc.append(np.mean(test_preds == knn.y_test))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, train_acc, marker='o', label='Train Accuracy')
    plt.plot(k_values, test_acc, marker='s', label='Test Accuracy')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('k-NN Accuracy for different k')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Best k: {k_values[int(np.argmax(test_acc))]} (Test accuracy: {max(test_acc):.4f})")
        


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

