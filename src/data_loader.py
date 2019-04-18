import csv
import numpy as np

def load_data(filename,data_type):
    # load the data from a .csv file named as filename
    a = []
    with open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        for row in data_file:
            a.append(row)
    # convert to numpy array
    arr = np.array(a)
    # load the training data
    if data_type=='train':
    # separate the example data X (without ID column) and the label data y
        X = arr[1:, 1:-1]
        y = arr[1:, -1]
        return X, y
    elif data_type=='test':    
        # the partial dataset Y (w/o the ID and label columns)
        Y = arr[1:, 1:-1]
        # return the entire dataset arr and the partial dataset Y
        return arr, Y

   
