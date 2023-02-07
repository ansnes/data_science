"""
In this file, I am implementing some classification modules that can be used for prediction.

LDA: Linear Discriminant Analysis.
    Builds a geometric model that will predict a measurement into any of n defined groups by minimizing distance to
    group center. Borders between the groups will be (n - 1)-dimensional hyperplanes. Lets user specify Euclidean
    distance, otherwise the more general Mahalanobis distance is used. Can be easily expanded to other distance metrics.
"""

import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


class Classify(object):

    def __init__(self):
        pass

    def prop_data(self, col_a, col_b):
        if (col_a + col_b).any() == 0:
            return 0
        else:
            return col_a / (col_a + col_b)
    def order_data(self, filepath, id_name, response):
        data = pd.read_csv(filepath)    # Reads csv file into a pandas dataframe
        y = data[[response]]            # Makes a separate pandas dataframe for the response column
        data = data[[col for col in data if col not in [id_name, response]]]  # Removes id and response columns
        data['prop_cancellations'] = self.prop_data(data['no_of_previous_cancellations'],
                                                   data['no_of_previous_bookings_not_canceled'])
        data = data.fillna(0)
        return data, y

    def make_numpy_arrays(self, data, y):
        X = pd.concat([data[[col]]
                       if data[[col]].dtypes.item() != "object"
                       else pd.get_dummies(data[[col]])
                       for col in data],
                      axis=1, join="inner").to_numpy().astype(float)
        G = pd.get_dummies(y).to_numpy().astype(float)
        return X, G

    def wrangle(self, filepath, id_name, response):
        """
        Converts data from csv file to np.arrays when id and response columns are specified. Handles categorical
        variables by converting them to one dummy variable for each level.

        :param filepath: path of csv data file
        :param id_name:  id column name (str)
        :param response: response column name (str)
        :return: data matrix and response vector
        """
        data, y = self.order_data(filepath, id_name, response)
        X, G = self.make_numpy_arrays(data, y)
        return X, G

    def distance(self, X, G, muG, d):
        """
        Gives the distance (Mahalanobis or Euclidean) for each measurement in X to group centers in muG.

        :param X: Data matrix from which function finds distances to group centers
        :param G: Matrix of group memberships (dummy)
        :param muG: Group centers
        :param d: distance metric. Euclidean or Mahalanobis
        :return: d2 - square distances of all measurements in X to all group centers.

        """
        n = len(X)  # Number of measurements

        #  Centering the measurements
        Xc = np.array([X[k] - muG[0] if G[k][0] == 1 else X[k] - muG[1] for k in range(n)])

        #  Finding squared distances
        if d == "Mahalanobis":
            Sinv = np.linalg.pinv(Xc.T @ Xc)  # Inverse of scaled covariance matrix
            return [[(X[i] - muG[c]) @ Sinv @ (X[i] - muG[c]).T for c in range(self.K)] for i in range(n)]
        elif d == "Euclidean":
            return [[(X[i] - muG[c]) @ (X[i] - muG[c]).T for c in range(self.K)] for i in range(n)]

    def confusion_matrix(self, G, Ghat):
        """
        Constructs the confusion matrix for a classification. Returns matrix, number of correct classifications, and
        proportion of correct classifications.

        :param G: Vector of group memberships, dummy variable form
        :param d2: Matrix of distances from group centers for every point that is to be predicted
        :return: conf_mat: the confusion matrix
                 ncc: number of correct classifications
                 pcc: proportion of correct classifications
        """
        confmat = np.zeros([self.K, self.K])    # Init of confusion matrix
        Gr = np.argmax(G, axis=1)               # Actual group memberships
        n = len(G)                              # Number of measurements
        for i in range(n):
            j = Gr[i]                           # Defines each row in confmat to represent actual group
            k = Ghat[i]                         # Defines each column in confmat to represent predicted group
            confmat[j][k] = confmat[j][k] + 1   # Updates confmat
        ncc = np.trace(confmat)                 # Number of correct classifications
        pcc = ncc / n                           # Proportion of correct classifications

        return confmat, ncc, pcc

    def LDA(self, X, G, d="Mahalanobis"):
        """
        :param X: Measurement matrix.
               G: Group matrix dummy form
               d: Distance metric. By default Mahalanobis, Euclidean can be requested by user.
        :return: pcc, fraction of correct classifications.
        """
        self.K = len(G[0])                                      # Number of groups
        muG = np.linalg.lstsq(G.T @ G, G.T @ X, rcond=None)[0]  # Group centers
        d2 = self.distance(X, G, muG, d)                        # Calculating group center distances to each point
        Ghat = np.argmin(d2, axis=1)                            # Predicted group memberships
        # Getting confusion matrix and proportion and number of correct classifications
        confmat, ncc, pcc = self.confusion_matrix(G, Ghat)
        return confmat, ncc, pcc

    def lin_reg(self, X, G):
        self.K = len(G[0])
        n = len(X)
        X = np.c_[np.ones(n), X]
        Ghat = np.argmax(X @ np.linalg.lstsq(X.T @ X, X.T @ G, rcond=None)[0], axis=1)
        Gr = np.argmax(G, axis=1)
        confmat, ncc, pcc = self.confusion_matrix(G, Ghat)
        return confmat, ncc, pcc

    def test_train(self, X, G, p, d="Mahalanobis"):
        n = len(X)
        self.K = len(G[0])
        d2 = np.zeros([n, self.K])
        sample_size = int(np.round(n * p))
        training_idc = rd.sample([k for k in range(n)], sample_size)
        test_idc = [k for k in range(n) if k not in training_idc]
        Xtrain = X[training_idc]
        Gtrain = G[training_idc]
        Xtest = X[test_idc]
        Gtest = G[test_idc]
        muG_train = np.linalg.lstsq(Gtrain.T @ Gtrain, Gtrain.T @ Xtrain, rcond=None)[0]
        d2 = self.distance(Xtest, Gtest, muG_train, d)
        Ghat = np.argmin(d2, axis=1)
        confmat, ncc, pcc = self.confusion_matrix(Gtest, Ghat)
        return confmat, ncc, pcc

    def remove_one_variable(self, filepath, id_name, response, model):
        data, y = self.order_data(filepath, id_name, response)
        removed_variables = ['type_of_meal_plan',
                             'arrival_month',
                             'no_of_adults',
                             'no_of_children',
                             'arrival_date']
        data = data[[col for col in data if col not in removed_variables]]
        X, G = self.make_numpy_arrays(data, y)
        variable_removed = "none"
        if model == "LDA":
            _, _, pcc = self.LDA(X, G)

        elif model == "lin_reg":
            _, _, pcc = self.lin_reg(X, G)

        for col in data:
            red_data = data[[col for col in data]]
            removed_variable = red_data.pop(col).name
            X, G = self.make_numpy_arrays(red_data, y)
            if model == "LDA":
                confmat, ncc, red_pcc = self.LDA(X, G)
            elif model == "lin_reg":
                confmat, ncc, red_pcc = self.lin_reg(X, G)
            if red_pcc > pcc:
                pcc = red_pcc
                removed_variables.append(removed_variable)
                break
        print(pcc)
        print(removed_variables[-1])
        print(confmat)




if __name__ == "__main__":
    ins = Classify()
    file = "data/hotels.csv"

    X, G = ins.wrangle(file, 'Booking_ID', 'booking_status')

    ins.remove_one_variable(file, 'Booking_ID', 'booking_status', "lin_reg")

    # print(X)
    # print(G)

    # rd.seed(42)
    # confmat, ncc, pcc = ins.test_train(X, G, .4)
    # print(confmat)
    # print(ncc)
    # print(pcc)

    # confmat, ncc, pcc = ins.LDA(X, G)
    # print(confmat)
    # print(pcc)

    # confmat, ncc, pcc = ins.lin_reg(X, G)
    # print(confmat)
    # print(pcc)