#________________________________________
#This is my algorithms.py
#________________________________________

import numpy as np
import pandas as pd
import math
import random
from collections import Counter
from scipy.linalg import solve

#_________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#____________________
#LOGISTIC REGRESSION 
#____________________


np.random.seed(42)

#_______________Helper Functions_______________
def add_bias(X):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    return X_b, m, n

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_hat):
    m = y.shape[0]
    loss = - (y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))
    return float(np.mean(loss))

def predict(X, theta, threshold=0.5):
    X_b = np.c_[np.ones((len(X), 1)), X]
    linear = np.dot(X_b, theta)
    probabilities = sigmoid(linear)
    y_pred = (probabilities >= threshold).astype(int)
    return probabilities, y_pred

#_______________Logistic Regression Class_______________
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_epochs=10, mini_batch_size=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.theta = None
        self.losses = []

    def fit(self, X, y):
        X_b, m, n = add_bias(X)
        y = y.reshape(-1, 1)
        self.theta = np.random.randn(n + 1, 1)
        n_batches = m // self.mini_batch_size

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start = i * self.mini_batch_size
                end = start + self.mini_batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                linear = np.dot(X_batch, self.theta)
                predictions = sigmoid(linear)
                errors = y_batch - predictions

                gradients = (-1.0 / self.mini_batch_size) * np.dot(X_batch.T, errors)
                self.theta -= self.learning_rate * gradients

            # Track loss after each epoch
            linear = np.dot(X_b, self.theta)
            preds = sigmoid(linear)
            loss = compute_loss(y, preds)
            self.losses.append(loss)
            #print(f"Epoch {epoch + 1}/{self.n_epochs}: Loss = {loss:.4f}")

    def predict(self, X):
        _, y_pred = predict(X, self.theta)
        return y_pred
    
#_______________One-vs-Rest (Multiclass Wrapper)______________________________
class OVRLogisticRegression:
    def __init__(self, base_lr_class=LogisticRegression, n_classes=10):
        self.n_classes = n_classes
        self.models = [base_lr_class() for _ in range(n_classes)]

    def fit(self, X, y):
        for i in range(self.n_classes):
            print(f"\nTraining OvR classifier for class {i}...")
            y_binary = (y == i).astype(int)
            self.models[i].fit(X, y_binary)

    def predict(self, X):
        all_probs = []
        for model in self.models:
            probs, _ = predict(X, model.theta)
            all_probs.append(probs)
        all_probs = np.hstack(all_probs)
        return np.argmax(all_probs, axis=1)


#HYPER PARAM TUNING FOR LOG REGRESSION.(THIS IS TAKEN DIRECTLY FROM ASSIGNMENT 2)


#Start changing first with learningrate, keeping others same.
#learning_rate=, n_epochs=, mini_batch_size=,val_acc
#0.1,15,100,83.59
#0.2,15,100,86.15
#0.3,15,100,86.87
#0.4,15,100,87.39--------->1st
#0.5,15,100,86.99--------->2nd

#Now change the n_epochs.
#learning_rate=, n_epochs=, mini_batch_size=,val_acc
#0.4,15,100,87.39
#0.4,20,100,87.03
#0.4,25,100,86.19
#0.4,30,100,87.31
#0.4,35,100,87.6
#0.4,40,100,88.12--------->BEST
#0.4,45,100,87.31
       

#Now for mini_batch
#0.4,40,100,88.12--------->BEST
#0.4,40,128,87.76

#When i tested for 0.5,45,150 i got 88.32---------->BEST



#FINAL 
#learning_rate=0.5, n_epochs=45, mini_batch_size=150


#__________________________________________________________________________________________________________________________________________________________________________________________________________________

#_____________________________Multiclass Logistic Regression_____________________________
def stable_softmax(logits):
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_onehot):
    eps = 1e-12
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


class SoftmaxRegression:
    def __init__(self, lr=0.1, n_epochs=200, num_classes=10):
        self.lr = lr
        self.n_epochs = n_epochs
        self.num_classes = num_classes
        self.W = None    #
        self.b = None    

    def _one_hot(self, y):
        n = y.shape[0]
        K = self.num_classes
        oh = np.zeros((n, K))
        oh[np.arange(n), y.astype(int)] = 1
        return oh

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self.num_classes


        self.W = np.random.randn(n_features, K) * 0.01
        self.b = np.zeros(K)

        y_onehot = self._one_hot(y)

        for epoch in range(self.n_epochs):
            logits = X.dot(self.W) + self.b  
            probs = stable_softmax(logits)   

            loss = cross_entropy_loss(probs, y_onehot)


            grad_logits = (probs - y_onehot) / n_samples 
            grad_W = X.T.dot(grad_logits)                 
            grad_b = np.sum(grad_logits, axis=0)          

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        logits = X.dot(self.W) + self.b
        return stable_softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
#_______________Softmax Regression Utility (Testing)_______________

def test_softmax_regression(X_train, y_train, X_val, y_val, lr=0.1, epochs=50):
    start = time.time()
    print("\n--- Testing SoftmaxRegression (Full Batch) ---")
    model = SoftmaxRegression(lr=lr, n_epochs=epochs, num_classes=10)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    end = time.time()
    acc = np.mean(preds == y_val)
    print(f"Softmax Regression Accuracy: {acc*100:.2f}%")
    print(f"Training time: {end-start:.3f} seconds\n")
    return acc


#________________________________________________________________________________________________________________________________________________

#Mini-batch Softmax Regression 
class SoftmaxRegressionMiniBatch:
    def __init__(self, lr=0.1, n_epochs=50, batch_size=256, num_classes=10):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.W = None
        self.b = None

    def _one_hot(self, y):
        n = y.shape[0]
        K = self.num_classes
        oh = np.zeros((n, K))
        oh[np.arange(n), y.astype(int)] = 1
        return oh

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = self.num_classes

        self.W = np.random.randn(n_features, K) * 0.01
        self.b = np.zeros(K)
        y_onehot = self._one_hot(y)

        n_batches = max(1, n_samples // self.batch_size)

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = X_batch.dot(self.W) + self.b   
                probs = stable_softmax(logits)

                grad_logits = (probs - y_batch) / X_batch.shape[0]
                grad_W = X_batch.T.dot(grad_logits)
                grad_b = np.sum(grad_logits, axis=0)

                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        logits = X.dot(self.W) + self.b
        return stable_softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
#_______________Mini-Batch Softmax Regression Utility_______________

def test_softmax_minibatch(X_train, y_train, X_val, y_val,
                           lr=0.1, epochs=40, batch=256):
    start = time.time()
    print("\n--- Testing SoftmaxRegressionMiniBatch ---")
    model = SoftmaxRegressionMiniBatch(
        lr=lr,
        n_epochs=epochs,
        batch_size=batch,
        num_classes=10
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    end = time.time()
    acc = np.mean(preds == y_val)
    print(f"Mini-Batch Softmax Accuracy: {acc*100:.2f}%")
    print(f"Training time: {end-start:.3f} seconds\n")
    return acc


#_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#____________________________________________________________
# RANDOM FOREST 
#____________________________________________________________

import random
import numpy as np
import pandas as pd
from collections import Counter
import time

random.seed(42)
np.random.seed(42)

#_______________Decision Tree Node_______________
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


#_______________Decision Tree Classifier_______________
class DecisionTreeClassifier:
    def __init__(self, max_depth=10, min_samples_split=2, feature_indices=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_indices = feature_indices
        self.root = None

    def fit(self, X, y):
        if not self.feature_indices:
            self.feature_indices = list(np.arange(len(X[0])))
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples = len(y)
        num_classes = len(set(y))

        if num_samples == 0:
            return None

        if (depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y)

        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionTreeNode(value=leaf_value)

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return DecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = 0
        split_idx, split_thresh = None, None

        for feat_idx in self.feature_indices:
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                gain = self._gini_gain(y, X[:, feat_idx], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _gini_gain(self, y, feature_column, threshold):
        parent_gini = self._gini(y)
        left_idx = feature_column <= threshold
        right_idx = feature_column > threshold

        if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(y[left_idx]), len(y[right_idx])
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        child_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        return parent_gini - child_gini

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - sum(p ** 2 for p in probabilities if p > 0)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node.is_leaf_node():
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)


#_______________Random Forest Classifier_______________
class RandomForest:
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        self.max_features = self.max_features or int(np.sqrt(X.shape[1]))

        for i in range(self.n_trees):
            print(f"Building tree {i+1}/{self.n_trees}")
            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_indices = random.sample(range(X.shape[1]), self.max_features)
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          feature_indices=feature_indices)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = np.random.randint(0, n_samples, n_samples)
        return X[indices], y[indices]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([self._most_common_label(preds) for preds in tree_preds])

    def _most_common_label(self, labels):
        return Counter(labels).most_common(1)[0][0]
    
    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        n_samples = X.shape[0]
        n_classes = 10
        probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            counts = Counter(tree_preds[i])
            for c in range(n_classes):
                probs[i, c] = counts.get(c, 0)
            probs[i] /= self.n_trees

        return probs


#_______________Utility Functions_______________
def read_data(filepath):
    df = pd.read_csv(filepath)
    feature_cols = [col for col in df.columns if col != 'label']
    X = (df[feature_cols] / 255.0).values
    y = df['label'].values  # 0–9 multiclass labels

    print(f"Loaded {filepath} → X: {X.shape}, y: {y.shape}")
    return X, y

def my_accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_multiclass(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm



# -------------------- Hyperparameter Tuning Notes(from assignment2) --------------------
# n_trees=, max_depth=, max_features=, val_acc
# 20,10,50 → 93.32%
# 20,10,60 → 93.96% (BEST)
# 20,10,70 → 94.08%
# 25,10,50 → 94.24%
# 20,15,60 → 94.36%
# FINAL CONFIG: n_trees=20, max_depth=10, max_features=60, val_acc=93.96%

#_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

#____________________________________________________________
# XGBOOST (Assignment2)
#____________________________________________________________

import random
import numpy as np
import pandas as pd
from collections import Counter
import time

random.seed(42)
np.random.seed(42)


#_______________Decision Tree Regressor_______________
class DecisionTreeRegressor:
    def __init__(self, max_depth=10, feature_indices=None, reg_lambda=1.0, gamma=0.0, n_bins=10):
        self.max_depth = max_depth
        self.root = None
        self.feature_indices = feature_indices
        self.reg_lambda = reg_lambda  
        self.gamma = gamma            
        self.n_bins = n_bins

    def fit(self, X, g, h):
        if not self.feature_indices:
            self.feature_indices = list(np.arange(len(X[0])))
        self.X_data = X
        self.g_data = g
        self.h_data = h
        all_indices = np.arange(X.shape[0])
        self.root = self._build_tree(all_indices, depth=0)

    def _build_tree(self, indices, depth):
        g = self.g_data[indices]
        h = self.h_data[indices]
        if (depth >= self.max_depth or len(indices) < 2):
            leaf_value = self._calculate_leaf_value(g, h)
            return DecisionTreeNode(value=leaf_value)

        best_gain, best_feat, best_thresh = self._best_split(indices)
        if best_feat is None or best_gain <= self.gamma:
            leaf_value = self._calculate_leaf_value(g, h)
            return DecisionTreeNode(value=leaf_value)

        X_node = self.X_data[indices]
        left_mask = X_node[:, best_feat] <= best_thresh
        right_mask = X_node[:, best_feat] > best_thresh
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        if left_indices.shape[0] == 0 or right_indices.shape[0] == 0:
            leaf_value = self._calculate_leaf_value(g, h)
            return DecisionTreeNode(value=leaf_value)

        left = self._build_tree(left_indices, depth + 1)
        right = self._build_tree(right_indices, depth + 1)
        return DecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _calculate_leaf_value(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        return -G / (H + self.reg_lambda)

    def _calculate_gain(self, G_L, H_L, G_R, H_R):
        term1 = (G_L**2) / (H_L + self.reg_lambda)
        term2 = (G_R**2) / (H_R + self.reg_lambda)
        term3 = (G_L + G_R)**2 / (H_L + H_R + self.reg_lambda)
        return 0.5 * (term1 + term2 - term3)

    def _best_split(self, indices):
        X_node = self.X_data[indices]
        g_node = self.g_data[indices]
        h_node = self.h_data[indices]
        best_gain, split_idx, split_thresh = -1, None, None

        for feat_idx in self.feature_indices:
            x_col = X_node[:, feat_idx]
            quantiles = np.linspace(0, 1, self.n_bins + 2)[1:-1]
            thresholds = np.unique(np.quantile(x_col, quantiles))
            if len(thresholds) == 0:
                continue

            for t in thresholds:
                left_mask = x_col <= t
                right_mask = x_col > t
                g_left, g_right = g_node[left_mask], g_node[right_mask]
                h_left, h_right = h_node[left_mask], h_node[right_mask]

                if h_left.shape[0] == 0 or h_right.shape[0] == 0:
                    continue

                G_L, H_L = np.sum(g_left), np.sum(h_left)
                G_R, H_R = np.sum(g_right), np.sum(h_right)
                gain = self._calculate_gain(G_L, H_L, G_R, H_R)

                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, t

        return best_gain, split_idx, split_thresh

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node.is_leaf_node():
            return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

#_______________XGBoost Classifier_______________
class XGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0, n_bins=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda  
        self.gamma = gamma      
        self.n_bins = n_bins
        self.trees = []
        self.base_prediction = 0
        self.n_features_ = 0
        
    def sigmoid_func(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, self.n_features_ = X.shape
        pos_ratio = np.mean(y)
        eps = 1e-15 
        pos_ratio = np.clip(pos_ratio, eps, 1 - eps)
        self.base_prediction = np.log(pos_ratio / (1 - pos_ratio))
        raw_predictions = np.full(y.shape, self.base_prediction)

        self.train_losses, self.val_losses = [], []

        for i in range(self.n_estimators):
            probabilities = self.sigmoid_func(raw_predictions)
            gradients = probabilities - y
            hessians = probabilities * (1 - probabilities)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                n_bins=self.n_bins
            )

            tree.fit(X, gradients, hessians)
            predictions = tree.predict(X)
            raw_predictions += self.learning_rate * predictions
            self.trees.append(tree)

            # Print training/validation loss during training
            # if (i + 1) % 10 == 0:
            #     train_loss = -np.mean(y*np.log(probabilities + 1e-9) + (1 - y)*np.log(1 - probabilities + 1e-9))
            #     self.train_losses.append(train_loss)
            #     if X_val is not None and y_val is not None:
            #         val_probs = self.predict_proba(X_val)
            #         val_loss = -np.mean(y_val*np.log(val_probs + 1e-9) + (1 - y_val)*np.log(1 - val_probs + 1e-9))
            #         self.val_losses.append(val_loss)
            #     print(f"Tree {i+1}/{self.n_estimators}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    def predict(self, X):
        """Returns the binary class (0 or 1)"""
        probabilities = self.predict_proba(X)
        return np.round(probabilities).astype(int)

    def predict_proba(self, X):
        """Returns the raw probabilities (0.0 to 1.0)"""
        raw_predictions = np.full(X.shape[0], self.base_prediction)
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X)
        probabilities = self.sigmoid_func(raw_predictions)
        return probabilities


#_______________One-vs-Rest (Multiclass Wrapper for XGBoost)_______________
class OVRXGBoost:
    """
    One-vs-Rest wrapper for multiclass classification using the binary XGBoost implementation.
    Trains one XGBoost model per class (0–9) and predicts by selecting the class
    with the highest probability.
    """

    def __init__(self, base_xgb_class=XGBoost, n_classes=10):
        self.n_classes = n_classes
        self.models = [base_xgb_class() for _ in range(n_classes)]

    def fit(self, X, y):
        print(f"\n[INFO] Starting One-vs-Rest XGBoost training for {self.n_classes} classes...\n")
        for i in range(self.n_classes):
            print(f"Training classifier for class {i} vs rest...")
            y_binary = (y == i).astype(int)
            self.models[i].fit(X, y_binary)
        print("\n[INFO] OvR XGBoost training completed.\n")

    def predict(self, X):
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(X).reshape(-1, 1) 
            all_probs.append(probs)
        
        all_probs = np.hstack(all_probs)
        return np.argmax(all_probs, axis=1)
    
    def predict_proba(self, X):
        all_probs = []
        for model in self.models:
            probs = model.predict_proba(X).reshape(-1, 1)
            all_probs.append(probs)
        return np.hstack(all_probs)

    

#_______________ HYPERPARAMETER TUNING NOTES_______________
# Step 1: n_estimators tuning (20–150) → best = 100
# Step 2: learning_rate tuning (0.1–0.5) → best = 0.5
# Step 3: max_depth tuning (3–5) → best = 4
# Final Tuned Parameters: n_estimators=100, learning_rate=0.5, max_depth=4, reg_lambda=1.0, gamma=0.0
# Final Validation Accuracy: 97.16%
# Training time ≈ 180 sec
# Observations:
# - Increasing estimators beyond 100 gave minimal gain but doubled runtime
# - Increasing depth beyond 4 led to overfitting
# - XGBoost outperformed Logistic Regression and Random Forest on both accuracy and recall(for assignment2)
#____________________________________________________________________________________________________________________________________________________________________________________

#____________________________________________________XGBoostMulticlass____________________________________________________


class XGBoostMulticlass:
    def __init__(self, n_estimators=10, learning_rate=0.2, max_depth=3, reg_lambda=1.0, gamma=0.0, n_bins=10, num_classes=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.n_bins = n_bins
        self.num_classes = num_classes

        self.trees = []  
        self.base_score = None  

    def _softmax(self, raw_preds):
        z = raw_preds - np.max(raw_preds, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        K = self.num_classes


        pos_counts = np.bincount(y, minlength=K).astype(float) + 1e-8
        pos_ratio = pos_counts / np.sum(pos_counts)
        self.base_score = np.log(pos_ratio)  


        raw_preds = np.tile(self.base_score, (n_samples, 1))  

        self.train_losses = []
        self.val_losses = []

        for t in range(self.n_estimators):

            probs = self._softmax(raw_preds) 

            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(n_samples), y.astype(int)] = 1

            grads = probs - y_onehot           
            hess = probs * (1.0 - probs)      


            trees_this_round = []
            for k in range(K):

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                    n_bins=self.n_bins
                )

                tree.fit(X, grads[:, k], hess[:, k])
                trees_this_round.append(tree)


                pred_k = tree.predict(X) 
                raw_preds[:, k] += self.learning_rate * pred_k

            self.trees.append(trees_this_round)


            if (t + 1) % 1 == 0:
                probs_train = self._softmax(raw_preds)
                eps = 1e-12
                train_loss = -np.mean(np.sum(y_onehot * np.log(probs_train + eps), axis=1))
                self.train_losses.append(train_loss)

                if X_val is not None and y_val is not None:
                    val_raw = np.tile(self.base_score, (X_val.shape[0], 1))
                    for trees_round in self.trees:
                        for k_idx, tr in enumerate(trees_round):
                            val_raw[:, k_idx] += self.learning_rate * tr.predict(X_val)
                    val_probs = self._softmax(val_raw)
                    yv_onehot = np.zeros_like(val_probs)
                    yv_onehot[np.arange(X_val.shape[0]), y_val.astype(int)] = 1
                    val_loss = -np.mean(np.sum(yv_onehot * np.log(val_probs + eps), axis=1))
                    self.val_losses.append(val_loss)

                print(f"Round {t+1}/{self.n_estimators} - train_loss: {train_loss:.4f}")

        return self

    def predict_proba(self, X):
        raw = np.tile(self.base_score, (X.shape[0], 1)).astype(float)
        for trees_round in self.trees:
            for k_idx, tr in enumerate(trees_round):
                raw[:, k_idx] += self.learning_rate * tr.predict(X)
        return self._softmax(raw)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
#_______________Multiclass XGBoost Utility_______________

def test_xgboost_multiclass(X_train, y_train, X_val, y_val,
                            n_estimators=10, lr=0.2, depth=3, bins=8):
    start = time.time()
    print("\n--- Testing XGBoostMulticlass ---")
    model = XGBoostMulticlass(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=depth,
        reg_lambda=1.0,
        gamma=0.0,
        n_bins=bins,
        num_classes=10
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    end = time.time()
    acc = np.mean(preds == y_val)
    print(f"Multiclass XGBoost Accuracy: {acc*100:.2f}%")
    print(f"Training time: {end-start:.3f} seconds\n")
    return acc




#________________________________________________________________________________________________________________________________________________________________________________


#________________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________
# K-NEAREST NEIGHBOURS (KNN) – curiosity baseline
#____________________________________________________________

import numpy as np
from collections import Counter

class KNNClassifier:
    """
    Simple KNN classifier implemented from scratch.
    Stores the entire training set and performs distance-based lookup.
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Just stores the training data."""
        self.X_train = X
        self.y_train = y
        print(f"KNN stored {X.shape[0]} samples for lookup (k={self.k}).")

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        preds = []

        for i, x in enumerate(X):
            # Compute distances to all training points
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

            # Pick the k nearest
            idx = np.argsort(dists)[:self.k]
            nearest_labels = self.y_train[idx]

            # Majority vote
            pred = Counter(nearest_labels).most_common(1)[0][0]
            preds.append(pred)

            # Print occasionally so student sees progress
            if (i + 1) % 500 == 0:
                print(f"Processed {i+1}/{len(X)} samples...")

        return np.array(preds)

    def predict_proba(self, X):
        """
        Probability = fraction of neighbor votes for each class.
        """
        n_samples = X.shape[0]
        n_classes = 10
        prob_matrix = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            idx = np.argsort(dists)[:self.k]
            nearest_labels = self.y_train[idx]

            counts = Counter(nearest_labels)
            for c in range(n_classes):
                prob_matrix[i, c] = counts.get(c, 0) / self.k

        return prob_matrix


#_______________ Utility tester _______________
def test_knn(X_train, y_train, X_val, y_val, k=3):
    print(f"\n===== Testing KNN Classifier (k={k}) =====")
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = np.mean(preds == y_val)

    print(f"KNN Accuracy: {acc*100:.2f}%")
    return acc


#__________________________________________________________________________________________________________________________________________________________________________________________________________________
#________________________________________
#ENSEMBLE
#________________________________________

class EnsembleClassifier:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1]*len(models)

    def predict_proba(self, X):
        total_weight = sum(self.weights)
        prob_sum = None

        for w, model in zip(self.weights, self.models):

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)

            else:
                raise ValueError("Model lacks probability output")

            prob_sum = probs * w if prob_sum is None else prob_sum + probs * w

        return prob_sum / total_weight

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
