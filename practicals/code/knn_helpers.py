import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class KNN:
    def __init__(self, X, y, trainsplit, k=None):
        self.k = k
        self.X = X  
        self.y = y
        
        # The trainsplit parameter specifies the fraction of data to be used for training (e.g., 0.8 means 80% train, 20% test)
        self.trainsplit = trainsplit
        
    def test_train_split(self):
        """
        Splits the dataset into training and testing sets based on the trainsplit ratio. The train and test data are stored in self.X_train, self.y_train, self.X_test, self.y_test.
        """
        if self.trainsplit < 1.0:
            split_index = int(self.trainsplit * self.X.shape[0])
            self.X_train = self.X[:split_index, :]
            self.y_train = self.y[:split_index].reshape(-1)  # Ensure y is a 1D array
            self.X_test = self.X[split_index:, :]
            self.y_test = self.y[split_index:].reshape(-1)  # Ensure y is a 1D array
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
        # Compute the distance matrix D
        D = sp.spatial.distance.cdist(X_test, X_train, 'euclidean')
        
        # Check shape of the D matrix
        shape = D.shape
        if shape[0] != X_test.shape[0] or shape[1] != X_train.shape[0]:
            raise ValueError("The shape of the distance matrix is incorrect.")
        return D
    
    def select_k_neighbors(self, D, k=None):
        """
        Return indices of k smallest per row, sorted by distance.
        """
        # Depending on input k, the indices of the k smallest distances are are stored in a n_train x k matrix
        if k is None:
            k = self.k
            if k is None:
                raise RuntimeError("k is not set. Provide k as function argument or set k as class attribute.")
        idx_mat = np.argpartition(D, kth=k-1, axis=1)[:, :k]
        
        # Sort the k indices by distance
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
            pred_array[i] = np.bincount(neigh_labels[i].astype(int), minlength=n_classes).argmax()
        
        return pred_array
    
    def vote_mean(self, neigh_targets):
        """
        For regression: return the mean of the neighbor targets.
        """
        return neigh_targets.mean(axis=1, keepdims=True)

    def predict(self, X_test=None, k=None, classification=True):
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
            if k is None:
                raise RuntimeError("k is not set. Provide k as function argument or set k as class attribute.")

        # compute distance matrix
        D = self.eucl_dist_matrix(X_test, self.X_train)
        
        # find the k nearest neighbors for every test sample
        idx = self.select_k_neighbors(D, k)
        
        # get the labels of the training samples of the k nearest neighbors
        neigh_labels = self.y_train[idx]

        # vote for the most common class among the k neighbors
        if classification:
            y_pred = self.vote_majority(neigh_labels)
        else:
            y_pred = self.vote_mean(neigh_labels)
        
        return y_pred
    
    def mse(self, y_true, y_pred):
        """Mean Squared Error (MSE)."""
        return float(((y_true - y_pred) ** 2).mean())

    def r2(self, y_true, y_pred):
        """Coefficient of determination R^2."""
        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot
    
    def score(self, k=None):
        """
        Accuracy on the held-out test set.
        """
        preds = self.predict(k=k)
        return self.mse(self.y_test, preds)
    
def plot_knn_stats(X, y, k_values, trainsplit, metric, normalize=True, plot=False):
    """
    Plot k-NN stats for a range of k values.
    """
    
    knn = KNN(X, y, trainsplit)
    knn.test_train_split()
    
    # Optionally normalize the data (this is important for distance-based methods like k-NN classification but not for regression)
    if normalize:
        knn.normalize_train_test()

    if metric == 'accuracy':
        train_acc, test_acc = [], []
        for k in k_values:
            # Create a new KNN instance for each k to avoid side effects
            train_preds = knn.predict(knn.X_train, k)
            test_preds = knn.predict(knn.X_test, k)
            train_acc.append(np.mean(train_preds == knn.y_train))
            test_acc.append(np.mean(test_preds == knn.y_test))
            
        if plot:
            # Plot accuracy vs k
            plt.figure(figsize=(8, 5))
            plt.plot(k_values, train_acc, marker='o', label='Train Accuracy')
            plt.plot(k_values, test_acc, marker='s', label='Test Accuracy')
            
            # Plot a vertical line at the best k
            best_k = k_values[int(np.argmax(test_acc))]
            plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k = {best_k}')
            
            # Set labels and title
            plt.xlabel('k')
            plt.ylabel('Accuracy')
            plt.title('k-NN Accuracy for different k')
            plt.legend()
            plt.grid(True)
            plt.show()
        
    elif metric == 'mse':
        # Train-vs-train distance matrix with diagonal excluded
        D_tr = knn.eucl_dist_matrix(knn.X_train, knn.X_train)
        D_tr = D_tr.copy()
        np.fill_diagonal(D_tr, np.inf)
        ntr = D_tr.shape[0]

        train_mse, test_mse = [], []
        for k in k_values:
            idx_tr = knn.select_k_neighbors(D_tr, k)
            neigh_targets_tr = knn.y_train[idx_tr].astype(float)

            preds_tr = neigh_targets_tr.mean(axis=1)
            train_mse.append(knn.mse(knn.y_train, preds_tr))

            # For test set, use the same regression logic (mean of neighbor targets)
            D_te = knn.eucl_dist_matrix(knn.X_test, knn.X_train)
            idx_te = knn.select_k_neighbors(D_te, k)
            neigh_targets_te = knn.y_train[idx_te].astype(float)
            preds_te = neigh_targets_te.mean(axis=1)
            test_mse.append(knn.mse(knn.y_test, preds_te))

        train_mse = np.array(train_mse)
        test_mse = np.array(test_mse)

        if plot:
            # Plot MSE vs k
            plt.figure(figsize=(8,5))
            plt.plot(k_values, test_mse, marker='o', label='Test MSE')
            plt.plot(k_values, train_mse, marker='s', label='Train MSE')
            
            # Plot a vertical line at the best k
            best_k = k_values[int(np.argmin(test_mse))]
            plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k = {best_k}')
            
            # Set labels and title
            plt.xlabel('k'); plt.ylabel('Mean Squared Error (MSE)')
            plt.title(f'KNN regression MSE vs k (trainsplit={trainsplit})')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return train_mse, test_mse


class RegressionMetrics:
        def __init__(self, y_true, y_pred):
            self.y_true = np.array(y_true)
            self.y_pred = np.array(y_pred)

        def mse(self):
            """
            Mean Squared Error
            """
            return np.mean((self.y_true - self.y_pred) ** 2)

        def r2_score(self):
            """
            R2 score (coefficient of determination)
            """
            ss_res = np.sum((self.y_true - self.y_pred) ** 2)
            ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        def roc_curve_numpy(y_true, y_score):
            thresholds = np.unique(y_score)[::-1]
            tpr = []
            fpr = []
            for thresh in thresholds:
                y_pred = (y_score >= thresh).astype(int)
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
                fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            return np.array(fpr), np.array(tpr), thresholds

        def auc_numpy(fpr, tpr):
            order = np.argsort(fpr)
            return np.trapz(tpr[order], fpr[order])

