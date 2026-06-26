import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import train_test_split 

def set_data(filename, p, n):
    df = pd.read_csv(filename, header=0)             
    data = df.to_numpy()
    data_subset = data[0:n]                          
    Y = data_subset[:, 0]
    X = np.ones((len(Y), p+3))                      

    X[:, [0, 1]] = data_subset[:, [1, 2]]
    for i in range(p):                               
        X[:, [i+3]] = data_subset[:, [i+3]]

    X_data, X_test_data, Y_data, Y_test_data = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_sub_data = X_data[0:len(Y_data), 2:(p+3)]    
    n = len(Y_data)

    return X_data, X_sub_data, Y_data, X_test_data, Y_test_data 