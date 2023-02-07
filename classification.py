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

    def confusion_matrix(self, G, d2):
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
        Ghat = np.argmin(d2, axis=1)            # Indices of least square distance, i.e. predicted group memberships
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
        # Getting confusion matrix and proportion and number of correct classifications
        confmat, ncc, pcc = self.confusion_matrix(G, d2)
        return confmat, pcc, muG

    def lin_reg(self, X, G):
        K = 2
        n = len(X)
        X = np.c_[np.ones(n), X]
        Ghat = np.argmax(X @ np.linalg.lstsq(X.T @ X, X.T @ G, rcond=None)[0], axis=1)
        Gr = np.argmax(G, axis=1)
        confmat = np.zeros([K, K])

        for i in range(n):
            j = Gr[i]                           # Defines each row in confmat to represent actual group
            k = Ghat[i]                         # Defines each column in confmat to represent predicted group
            confmat[j][k] = confmat[j][k] + 1   # Updates confmat
        ncc = np.trace(confmat)                 # Number of correct classifications
        pcc = ncc / n                           # Proportion of correct classifications
        print(confmat)
        print(pcc)


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
        confmat, ncc, pcc = self.confusion_matrix(Gtest, d2)
        return confmat, ncc, pcc


if __name__ == "__main__":
    ins = Classify()
    file = "data/hotels.csv"
    data = pd.read_csv(file)

    # Wrangling. Removing the ID column and recoding the categorical variables
    meals = pd.get_dummies(data.type_of_meal_plan)
    rooms = pd.get_dummies(data.room_type_reserved)
    segment_mapping = {"Offline": 0, "Online": 1}
    cancelled = pd.get_dummies(data.booking_status)
    # Numeric data, market_segment_type removed as it produced nan-values.
    n_data = pd.concat([data.no_of_adults,
                        data.no_of_children,
                        data.no_of_weekend_nights,
                        data.no_of_week_nights,
                        meals,
                        data.required_car_parking_space,
                        rooms,
                        data.lead_time,
                        data.arrival_year,
                        data.arrival_month,
                        data.arrival_date,
                        data.repeated_guest,
                        data.no_of_previous_cancellations,
                        data.avg_price_per_room,
                        data.no_of_special_requests],
                       axis=1,
                       join="inner")
    X = n_data.to_numpy().astype(float)
    G = cancelled.to_numpy().astype(float)
    # rd.seed(42)
    # confmat, ncc, pcc = ins.test_train(X, G, .4)
    # print(confmat)
    # print(ncc)
    # print(pcc)

    # confmat, pcc, muG = ins.LDA(X, G)
    # print(confmat)
    # print(pcc)

    ins.lin_reg(X, G)




