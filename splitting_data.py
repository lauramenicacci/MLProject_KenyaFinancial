"""2. Import Dependencies

Python ≥3.5 and Scikit-Learn ≥0.20 are required for this project. Additional packages are imported and a seed is set to make this
notebook's output stable across runs.
"""

# run "pip install pickle-mixin" in anaconda prompt or wherever else you do pip install

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# import necessary dependencies
import numpy as np
import pickle
import pandas as pd

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# set seed to make this notebook's output stable across runs
np.random.seed(42)

from sklearn.model_selection import train_test_split


def training_set(x_path, y_path):
  with open(x_path, 'rb') as ifile:
    Xd =pickle.load(ifile)
  with open(y_path, 'rb') as ifile:
    Yd =pickle.load(ifile)

  X_train_val, X_test, y_train_val, y_test = train_test_split(
    Xd, Yd, test_size=0.2, random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42)
  
  return (X_train, X_val, X_test, y_train, y_val, y_test)
