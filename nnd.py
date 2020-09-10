""" Neural Network Deconvolution (NND) is a toolkit that unmixes 
  bulk tumor samples. It aims to solve the following problem:
    
    Given a non-negative bulk RNA expression matrix B \in R_+^{m x n}, 
    where each row i is a gene, each column j is a tumor sample, 
    our goal is to infer an expression profile matrix C \in R_+^{m x k}, 
    where each column l is a cell community, 
    and a fraction matrix F \in R_+^{k x n}, such that:
      B ~= C F.

  main APIs:
    compress_module: integrate gene module knowledge to reduce noise
    estimate_number: estimate the number of cell populations automatically
    estimate_clones: utilize core NND algorithm to unmix the cell populations
    estimate_marker: estimate other biomarkers of cell populations given bulk marker data

"""

#TODO: enable random seed as input param

import random

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import cvxopt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import mask_mse, torch_mask_mse, heuristic_learning_rate

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

__author__ = "Yifeng Tao"


def compress_module(B, module):
  """ compress raw gene expression into module expression
  
  Parameters
  ----------
  B: 2D array of non-negative float
    bulk gene data, each row a sample, each column a gene
  module: list of list of int
    each sublist contains indices of genes in the same gene module
    
  Returns
  -------
  B_M: 2D array of float
    compressed module-level bulk data
  
  """
  
  B_M = np.array([np.mean([B[idx] for idx in m],axis=0) for m in module], dtype=float)
  
  return B_M


class NND(nn.Module):
  """ NND model for deconvolution.
  """

  def __init__(self, n_features, n_samples, n_components, learning_rate, weight_decay):
    """
    Initialize the hyperparameters of the NND model.

    Parameters
    ----------
    n_features : int
      number of features (genes/gene modules)
    n_samples : int
      number of tumor samples
    n_components : int
      number of cell populations
    learning_rate : float
      learning rate for optimization
    weight_decay : TYPE
      coefficient of the l2-regularization term

    Returns
    -------
    None.

    """

    super(NND, self).__init__()
    
    self.epsilon = 1e-10 #1e-4

    self.n_features = n_features#dim_m
    
    self.n_samples = n_samples #dim_n
    
    self.n_components = n_components#dim_k
    
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay


  def build(self):
    """ Define modules of the model.

    """

    self.C = torch.nn.Parameter(
        data=torch.Tensor(self.n_features, self.n_components), requires_grad=True)
    self.C.data.uniform_(0, 1)

    self.F = torch.nn.Parameter(
        data=torch.Tensor(self.n_components, self.n_samples), requires_grad=True)
    self.F.data.uniform_(0, 1)

    self.optimizer = optim.Adam(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay)


  def forward(self):
    """
    Predict the output bulk data using estimated self.C and self.F
    
    Returns
    -------
    B_prd : 2D torch float array
      predicted bulk data
    F_abs : 2D torch float array
      estimated fraction matrix F
    C_abs : 2D torch float array
      estimated expression matrix C

    """
    
    F_abs = torch.abs(self.F)
    F_abs = F.normalize(F_abs, p=1, dim=0)

    C_abs = torch.abs(self.C)
    B_prd = torch.mm(C_abs, F_abs)

    return B_prd, F_abs, C_abs


  def train(self, B, M, max_iter=None, inc=1, verbose=False):
    """ Train the matrix factorization using gradient descent.

    Parameters
    ----------
    B: 2D numpy matrix
      bulk data, each column a sample, each row a gene module.
    M: numpy 0/1 mask matrix
      same size of B, positions of 1 mean seen data, otherwise unseen.
    max_iter: int
      max iterations of training.
    inc: int
      intervals to evaluate the training.
    verbose: boolen
      whether print too much itermediat results.

    Returns
    -------
    C: 2D numpy array
      Deconvolved matrix C
    F: 2D numpy array
      Deconvolved matrix F

    """

    B = Variable(torch.FloatTensor(B))
    M = Variable(torch.FloatTensor(M))

    previous_error = 1e10

    for iter_train in range(0, max_iter+1):
      
      B_prd, F, C = self.forward()
      
      self.optimizer.zero_grad()
      
      loss = torch_mask_mse(B, B_prd, M)

      loss.backward()
      self.optimizer.step()

      if iter_train % inc == 0:
        loss = 1.0*loss.data.numpy()
        if verbose:
          print( "iter=%d, l2_loss=%.2e"% (iter_train,loss) )

        if (previous_error - loss) / previous_error < self.epsilon:
          break
        
        previous_error = loss

    if iter_train >= max_iter-2*inc:
      warnings.warn("Not well optimized, please increase learning_rate or max_iter")

    C, F = np.array(C.data.numpy(), dtype=float), np.array(F.data.numpy(), dtype=float)
    
    return C, F
  
  
def estimate_clones(
    B, k, M=None, learning_rate=None, max_iter=1000, weight_decay=0, verbose=True):
  """
  Estimate the C and F using bulk data B.

  Parameters
  ----------
  B : 2D numpy array of float
    bulk data, each row a gene, each column a sample
  k : int
    number of cell populations/clones
  M : 2D numpy array of 0/1, optional
    if the element is 1, the corresponding element in B is used for optimization. 
    The default is None, and the whole B is used for optimization.
  learning_rate : float, optional
    learning rate of gradient descent during optimization. 
    The default is None, and the learning rate is determined heuristically.
  max_iter : int, optional
    max number of optimization iterations. The default is 1000.
  weight_decay : float, optional
    coefficient of the l2-regularization term during optimization. The default is 0.
  verbose : boolen, optional
    whether to output intermediate results during optimization. The default is True.

  Returns
  -------
  C : 2D numpy array of float
    estimated expression profiles of each cell clones
  F : 2D numpy array of float
    estimated fractions of each cell clones.

  """
    
  if M is None:
    M = np.ones(B.shape)
    
  if learning_rate is None:
    learning_rate = heuristic_learning_rate(B)

  nnd = NND(B.shape[0], B.shape[1], k, learning_rate, weight_decay)
  nnd.build()
  
  C, F = nnd.train(B, M, max_iter=max_iter, verbose=verbose)
  
  return C, F


# the coordinate descent phase borrows code from cvxopt:
# https://cvxopt.org/userguide/coneprog.html
def _cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
  """ solve the QP problem of
  
    minimize_x 1/2 x^T P x + q^T x
    subject to G x <= h
               A x = b
  
  """
  
  P = .5 * (P + P.T)  # make sure P is symmetric
  args = [cvxopt.matrix(P), cvxopt.matrix(q)]
  if G is not None:
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
      args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
  cvxopt.solvers.options["show_progress"] = False
  sol = cvxopt.solvers.qp(*args)
  if "optimal" not in sol["status"]:
    return None
  
  return np.array(sol["x"]).reshape((P.shape[1],))


def _quad_prog_BF2C(B, F, M, max_val=2**19):
  """ Solve the QP problem of
  
    minimize_C ||B - C F||_2^2
    subject to C >= 0
               C <= max_val
               
  """

  num_gene = B.shape[0]
  num_comp = F.shape[0]

  Gl = np.diag([-1.0]*num_comp)
  hl = np.zeros(num_comp).reshape((num_comp,))

  Gu = np.diag([1.0]*num_comp)
  hu = np.array([max_val]*num_comp).reshape((num_comp,))

  G=np.vstack([Gl, Gu])
  h=np.hstack([hl, hu])

  C = []

  for i in range(num_gene):
    m = M[i,:]
    Fm = [F[:,j] for j, v in enumerate(m) if v != 0]
    Fm = np.array(Fm).T
    P = np.dot(Fm, Fm.T)
    bm = np.array([B[i,j] for j, v in enumerate(m) if v != 0])
    q = -np.dot(Fm, bm)

    ci = _cvxopt_solve_qp(P, q, G, h)

    if ci is None:
      return None

    C.append(ci)

  C = np.vstack(C)

  return C


def estimate_marker(B, F, M=None, max_val=2**19):
  """ estimate biomarkers of individual components
  
    B_P, F -> C_P or
    B, F -> C
  
  """
  
  if M is None:
    M = np.ones(B.shape)
  
  C = _quad_prog_BF2C(B, F, M, max_val=max_val)
  
  return C


def estimate_number(B, max_comp=10, n_splits=20, plot_cv_error=True):
  """ Cross-validation of matrix factorization.

  Parameters
  ----------
  B: matrix
    bulk data to be deconvolved.
  n_comp: list int
    numbers of population component.
  n_splits: int
    fold of cross-validation.

  Returns
  -------
  results: dict
    numbers of components, training errors and test errors.
  """
  
  n_comp = [i+1 for i in range(max_comp)]

  results = {
      "n_comp":n_comp,
      "test_error":[[] for _ in range(len(n_comp))],
      "train_error":[[] for _ in range(len(n_comp))]
      }

  rng = [(idx, idy) for idx in range(B.shape[0]) for idy in range(B.shape[1])]
  random.Random(2020).shuffle(rng)

  kf = KFold(n_splits=n_splits)

  idx_fold = 0
  for train_index, test_index in kf.split(rng):
    idx_fold += 1

    rng_train = [rng[i] for i in train_index]
    rng_test = [rng[i] for i in test_index]

    M_test = np.zeros(B.shape)
    for r in rng_test:
      M_test[r[0],r[1]] = 1.0
    M_train = np.zeros(B.shape)
    for r in rng_train:
      M_train[r[0],r[1]] = 1.0

    for idx_trial in range(len(n_comp)):
      dim_k = results["n_comp"][idx_trial]

      C, F = estimate_clones(B, dim_k, M=M_train, verbose=False)

      l2_train = mask_mse(B, M_train, C, F)
      l2_test = mask_mse(B, M_test, C, F)
      results["train_error"][idx_trial].append(l2_train)
      results["test_error"][idx_trial].append(l2_test)

      #print("fold=%3d/%3d, dim_k=%2d, train=%.2e, test=%.2e"%(idx_fold, n_splits, dim_k, l2_train, l2_test))
      
  k = n_comp[np.argmin(np.mean(results["test_error"], axis=1))]
  
  if plot_cv_error:
    plot_cv(results,inputstr="test_error",deno=np.sum(np.multiply(B, B))/B.shape[0]/B.shape[1])

  return k


def plot_cv(results, inputstr="test_error", deno=1.0):
  """ Plot the cross-validation results.

  Parameters
  ----------
  results: dict
  
  """

  size_label = 18
  size_tick = 18
  sns.set_style("darkgrid")

  fig = plt.figure(figsize=(5,4))
  M_rst = []
  n_comp = results["n_comp"]
  M_test_error = np.asarray(results[inputstr]/deno)
  for idx, k in enumerate(n_comp):
    for v in M_test_error[idx]:
      M_rst.append([k, v])

  df = pd.DataFrame(
      data=M_rst,
      index=None,
      columns=["# comp", inputstr])
  avg_test_error = M_test_error.mean(axis=1)
  ax = sns.lineplot(x="# comp", y=inputstr, markers=True, data=df)

  idx_min = np.argmin(avg_test_error)
  
  #print("min:k=%d,mse=%f"%(n_comp[idx_min], avg_test_error[idx_min]))

  if inputstr == "test_error":
    yl = "Normalized CV MSE"
  else:
    yl = "Normalized train MSE"
  plt.ylabel(yl, fontsize=size_label)
  plt.xlabel("# components (k)", fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  plt.xlim([1, 4])#TODO
  #plt.ylim([0.55,0.95])

  plt.show()
  
  