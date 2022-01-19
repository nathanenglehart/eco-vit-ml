import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import sys

def targetvector(t,Xnew,lambda_value,D,scale):
    xval=Xnew
    X = X_build(Xnew, D)
    QR = qr(X)
    I = np.identity(D + 1)
    lam = I * lambda_value
    lam[0][0] = 0
    theta = np.matmul(np.matmul(np.linalg.inv(I + lam), QR[0].T), t)
    w = np.matmul(np.linalg.inv(QR[1]), theta)
    return np.array(w)

def qr(X):

    """ Computes Q and R for X matrix.

	Args:

		X::[Numpy Array]
                	Matrix for which to compute the QR. Note: should not have column of 1s at column index 0
    
    """

    return np.linalg.qr(X)
