from pathlib import Path
import pandas as pd
import numpy as np
import os
import sys

directory = Path(__file__).parent.parent
directory_path = os.path.abspath(directory)
# !! If "module auxiliary not found" error appears run the following code: !!
sys.path.append(directory_path)

from auxiliary import cross_validation
from src.matrix_factorization import sgd_factorization, als_factorization

num_eigenvalues = 5
max_iterations = 1000
train_size = 0.9
sys.path.append(directory_path)
def main():
    DATA_PATH = directory_path + '/data/data_train.csv'
    data_pd = pd.read_csv(DATA_PATH)

    cross_validation.cross_validate_factorization(data_pd, sgd_factorization)

if __name__ == "__main__":
    main()