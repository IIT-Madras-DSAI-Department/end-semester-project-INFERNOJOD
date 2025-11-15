#________________________________________
#This is my main.py
#Using: XGBoostMulticlass (Final Best Model)
#________________________________________

import time
import numpy as np
from algorithms import (
    read_data,
    my_accuracy_score,
    XGBoostMulticlass
)
from collections import Counter

#_______________Weighted F1 Score_______________
def weighted_f1_score(y_true, y_pred):
    classes = np.unique(y_true)
    f1_total = 0
    total_samples = len(y_true)

    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        weight = np.sum(y_true == cls) / total_samples
        f1_total += weight * f1

    return f1_total


#_________________________________________________________________________
# MAIN STARTS
#_________________________________________________________________________



X_train, y_train = read_data("/Users/krishyadav/Desktop/lab endsem/MNIST_train.csv")
X_val, y_val = read_data("/Users/krishyadav/Desktop/lab endsem/MNIST_validation.csv")

print("\n__________TRAINING FINAL MODEL: XGBoost Multiclass__________\n")

start_time = time.time()

# Tuned parameters from hyperparameter search
model = XGBoostMulticlass(
        n_estimators=25,      
        learning_rate=0.4,
        max_depth=4,
        reg_lambda=1.0,
        gamma=0.0,
        n_bins=8,
        num_classes=10
    )

model.fit(X_train, y_train)

print("\n__________EVALUATION ON VALIDATION SET__________\n")

preds = model.predict(X_val)

acc = my_accuracy_score(y_val, preds)
f1 = weighted_f1_score(y_val, preds)

end_time = time.time()

print(f"Accuracy: {acc * 100:.4f}%")
print(f"Weighted F1 Score: {f1:.4f}")
print(f"Training + Inference Time: {end_time - start_time:.2f} seconds\n")

print("__________END__________")
