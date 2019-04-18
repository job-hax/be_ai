import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder 

# Append the column y to the right of the 2-d dataset X and write the result to .csv file
def merge_to_file(X, y):
    f = open('../data/admission_pediction.csv', 'w')  
    try:
        writer = csv.writer(f)    
        m = len(y)                  # the number of rows to write
        n = len(X[0])               # the number of columns in X
        # make and write the header
        row = []
        for j in range(n):
            row.append(X[0, j])
        row.append('admission prediction')
        writer.writerow(row)

        # make and write each row
        for i in range(m):
            row = []
            for j in range(n):
                row.append(X[i+1, j])
            row.append(y[i])
            writer.writerow(row)
    finally:
        f.close()

def convert_categorical(X):
    # Convert non-numeric data X array to numeric data and return the converted data
    le = LabelEncoder()
    n_features = len(X[0])
    X_new = le.fit_transform(X[:,0])  # Take the first column
    for i in range (1, n_features): # add remaining columns
        Xi = le.fit_transform(X[:,i]) 
        X_new = np.column_stack((X_new, Xi))
    return X_new


