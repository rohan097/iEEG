#********************************Random Forest Implementation for Classification between Interical and Preictal iEEG signals********************************

import scipy.io
import numpy as np
from numpy import genfromtxt, savetxt
import pydot
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random
import sklearn.preprocessing
import csv

try:

    from itertools import izip
except ImportError:
    izip = zip

# This function is used to transpose the contents of the CSV file.
def transpose():

#    a = izip(*csv.reader(open("ProcessedData.csv", "rb")))
    a = izip(*csv.reader(open("ProcessedData.csv","rb")))
    csv.writer(open("Output.csv", "wb")).writerows(a)

# Converts the generated decision trees from a .DOT file to .PNG file
def toPNG():

    for i in range(0,3000):
        graph = pydot.graph_from_dot_file('tree_%d.dot' %i)
            graph.write_png('tree_%i.png' %i)

# Returns the Fast Fourier Transform of the raw data.
def fft(data):

    return np.log10(np.absolute(np.fft.rfft(data, axis = 0)[1:41,:]))

# Produces an upper right triangle matrix
def upper_right_triangle(matrix):
        accum = []
        for i in range(matrix.shape[0]):
            for j in range(i+1, matrix.shape[1]):
                accum.append(matrix[i, j])
                return np.array(accum)

# This function produces a concatenated array of the correlation coefficient and eigen values in the time domain.
def freq_corr(data):

    scaled = sklearn.preprocessing.scale(data, axis = 0)
    corr_matrix = np.corrcoef(scaled)
    eigenvalues = np.absolute(np.linalg.eig(corr_matrix)[0])
    eigenvalues.sort()
    corr_coefficients = upper_right_triangle(corr_matrix)
    return np.concatenate((corr_coefficients, eigenvalues))


# The main function used for processing the data.
def Processing():

    # Data processing for interictal segments.
    label = 1

    for i in range(1,1153):

        try:

            mat = scipy.io.loadmat('/home/rohan/train_1/1_%d_0.mat' %i)
            df = pd.DataFrame(mat['dataStruct'][0]['data'][0][:,:])

            data_fft = np.reshape(fft(df),(16,40))
            data_corr_eigen = freq_corr(data_fft)
            matrix = np.concatenate((data_fft.ravel(), data_corr_eigen))
            matrix = np.insert(matrix, 0, label, axis = 0)
            matrix = matrix.reshape(658,1)

            print ('Successfully processed file: 1_%d_0.mat' % i)
            if i == 1:
               final_matrix = matrix
            else:
                final_matrix = np.concatenate((final_matrix,matrix), axis = 1)

        except ValueError:
            print ('There was an error in file: 1_%d_0.mat' %i)
            pass

    # Data processing for preictal segments
    label = 2

    for j in range(1,151):

        try:

            mat = scipy.io.loadmat('/home/rohan/train_1/1_%d_1.mat' %j)
            df = pd.DataFrame(mat['dataStruct'][0]['data'][0][:,:])

            data_fft = np.reshape(fft(df),(16,40))
            data_corr_eigen = freq_corr(data_fft)
            matrix = np.concatenate((data_fft.ravel(), data_corr_eigen))
            matrix = np.insert(matrix, 0, label, axis = 0)
            matrix = matrix.reshape(658,1)
            final_matrix = np.concatenate((final_matrix,matrix), axis = 1)

            print ('Successfully processed file: 1_%d_1.mat' %j)

        except ValueError:
            print ('There was an error in file: 1_%d_1.mat' %j)

    # Writing the processed data to a .CSV file and then taking the transpose of the file
    np.savetxt('ProcessedData.csv', final_matrix, delimiter = ",")
    transpose()

# used to calculate the accuracy of the predictions.
def accuracy(predictions, test_labels):

    counter = 0.0
    total = 0.0

    for i in range(len(test_labels)):
        if  test_labels[i] == predictions[i]:
            counter += 1
        total += 1

    accuracy = counter/total*100

    print ('Accuracy = ', accuracy)

# Used to split the entire data into training and test data with a ratio corresponding to 75:25
def DataSplit():

    data = genfromtxt('Output.csv', delimiter = ',', dtype = 'f8')

    # Generate the shuffled training data along with its corresponing labels
    temp1 = data[0:950,:]
    temp2 = data[1127:1232,:]
    train = np.concatenate((temp1, temp2))
    random.shuffle(train)
    targets = train[:,0]
    train = train[:,1:]

    # Generate the testing data along with the true values, which is to be used later for determining accuracy.
    temp3 = data[950:1127,:]
    temp4 = data[1232:,:]
    test = np.concatenate((temp3, temp4))
    true_values = test[:,0]
    test = test[:,1:]

    return train, targets, test, true_values

# Training the Random Forest model
def RandomForest():

    train, target, test, true_values = DataSplit()

    rf = RandomForestClassifier(n_estimators = 3000, n_jobs = -1)
    rf.fit(train, target)

    print ('*************** Training Complete ***************')

    predictions = rf.predict(test)

    savetxt('/home/rohan/iEEG/Predictions.csv',predictions, delimiter = ',', fmt = '%f')
    accuracy(predictions, true_values)

    # This is used for visualising the decision trees generated by the model
    '''
    for i, tree in enumerate(rf.estimators_):
        with open('tree_' + str(i) + '.dot', 'w') as dotfile:
            sklearn.tree.export_graphviz(tree, dotfile, filled = True, rounded = True, class_names = ['1','2'])
    toPNG()
    '''

# Converts the generated decision trees from a .DOT file to .PNG file
def toPNG():

    for i in range(0,3000):
        graph = pydot.graph_from_dot_file('tree_%d.dot' %i)
        graph.write_png('tree_%i.png' %i)


def main():

    Processing()
    RandomForest()


if __name__ == '__main__':
    main()

