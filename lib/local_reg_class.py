import numpy as np

from lib.utils import append_ones

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

class LocalRidge():

  def __init__(self, lam=1.0):
      
      """ Ridge regression class based on sklearn functionality (for modular use)
		
		Args:
			
			lam::[Float]
				Weight penalty for high order polynomials

      """

      self.lam = lam

  def fit(self, X, t):

      """ Fits ridge regression model with given train matrix and target vector

		Args:
			
			X::[Numpy Array]
				Train matrix (build before putting into function)
			
			t::[Numpy Array]
				Target vector

      """

      X = append_ones(X)
      I = np.identity(X.shape[1])
      I[0, 0] = 0
      lam_matrix = self.lam * I
      theta = np.linalg.inv(X.T.dot(X) + lam_matrix).dot(X.T).dot(t)
      self.theta = theta
      return self

  def predict(self, X):
      
      """ Generates predictions for the given matrix based on model.

      		Args:
			
			X::[Numpy Array]
				Test matrix (build before putting into function)

      """

      theta = self.theta
      preds = append_ones(X)
      self.predictions = preds.dot(theta)

      return self.predictions
