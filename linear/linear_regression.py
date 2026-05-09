import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
import time
from metric import mae,mse
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/example/rumah.csv")

start_time = time.perf_counter()

X = df.drop(columns=["harga"]).to_numpy(dtype=float)
y = df["harga"].to_numpy(dtype=float)


w = np.zeros(X.shape[1])
b = 0
lr = 0.0001
epoch = 100
verbose = 10


def predict(X, w,b):
    return X @ w + b

def compute_error(y_pred, target):
    return y_pred-target

def compute_gradient(X,err):
    n = len(err)
    grad_w = (2 / n) * (X.T @ err)
    grad_b = (2 / n) * err.sum()
    return (grad_w,grad_b)

def update_weight(grad_w,grad_b,lr,w,b):
    w -= lr * grad_w
    b -= lr * grad_b
    return b

for train in range(epoch):
    y_pred = predict(X,w,b)
    error = compute_error(y_pred,y)
    grad_w,grad_b = compute_gradient(X,error)
    b = update_weight(grad_w,grad_b,lr,w,b)
    y_pred = predict(X,w,b)
    if (train+1) % verbose == 0:
        print(f"[{train+1}] MAE : {mae(y_pred,y)} | MSE : {mse(y_pred,y)} | RMSE : {np.sqrt(mse(y_pred,y))}")
print(f"Final Score, MAE : {mae(y_pred,y)} | MSE : {mse(y_pred,y)} | RMSE : {np.sqrt(mse(y_pred,y))}")
end_time = time.perf_counter()

print(f"Runtime: {end_time - start_time:.6f} seconds")