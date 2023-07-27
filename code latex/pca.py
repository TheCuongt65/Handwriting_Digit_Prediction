import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import idx2numpy

X_train_dir = './data/train-images.idx3-ubyte'
y_train_dir = './data/train-labels.idx1-ubyte'
X_val_dir = './data/t10k-images.idx3-ubyte'
y_val_dir = './data/t10k-labels.idx1-ubyte'

X_train = idx2numpy.convert_from_file(X_train_dir).reshape(60000, -1)
y_train = idx2numpy.convert_from_file(y_train_dir)
X_val = idx2numpy.convert_from_file(X_val_dir).reshape(10000, -1)
y_val = idx2numpy.convert_from_file(y_val_dir)


print("X-train shape:", X_train.shape)
print("y-train shape:", y_train.shape)
print("X-val shape:", X_val.shape)
print("y-val shape:", y_val.shape)

def myPCA(X, k = 3):
    X = X.T
    N  = X.shape[1]
    X_bar = X.mean(axis=1).reshape(-1, 1)
    X_hat = X - X_bar
    S = (X_hat @ X_hat.T) / N
    SVD = np.linalg.svd(S)
    U = SVD[0]
    lambd = SVD[1]
    Z = U[:, :k].T @ X_hat
    z = Z.T
    return z

THRESHOLD = 1000
X = X_train[:THRESHOLD]
z = myPCA(X, k = 3)
pc1 = z[:, 0]
pc2 = z[:, 1]
pc3 = z[:, 2]

plt.axes(projection = "3d").scatter3D(pc1, pc2, pc3, c = y_train[:THRESHOLD], s = 6)