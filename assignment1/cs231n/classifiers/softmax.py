import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_training = X.shape[0]
  C = W.shape[1]

  for i in range(num_training):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    probabilities = exp_scores / exp_scores.sum()
    correct_class_prob = probabilities[y[i]]
    loss += -np.log(correct_class_prob)

    dLdfyi = -1
    dfyidWyi = X[i] * dLdfyi
    dW[:, y[i]] += dfyidWyi
    for j in range(C):
      dLdefj = 1 / exp_scores.sum()
      defjdWj = exp_scores[j] * X[i] * dLdefj
      dW[:, j] += defjdWj


  loss /= num_training
  loss += reg * np.sum(W * W)
  dW /= num_training
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_training = X.shape[0]

  scores = X.dot(W)
  exp_scores = np.exp(scores)
  sum = exp_scores.sum(axis=1)
  probabilities = (exp_scores.T / sum).T

  loss += np.sum(-np.log(probabilities[np.arange(num_training), y]))
  loss /= num_training
  loss += reg * np.sum(W * W)

  # each i-th element of the column of the indicator matrix is coefficient of x_i
  # in the sum of all training examples
  indicator = probabilities.copy()
  indicator[np.arange(num_training), y] -= 1

  dW += X.T.dot(indicator)
  dW /= num_training
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

