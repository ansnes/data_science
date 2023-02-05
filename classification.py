"""
In this file, I am implementing some classification modules that can be used for prediction.

LDA: Linear Discriminant Analysis.
    Builds a geometric model that will predict a measurement into any of n defined groups by minimizing distance to
    group center. Borders between the groups will be (n - 1)-dimensional hyperplanes. Lets user specify Euclidean
    distance, otherwise the more general Mahalanobis distance is used. Can be easily expanded to other distance metrics.
"""

import numpy as np
import pandas as pd
from random import sample


class Classify(object):

    def __init__(self):
        pass

    def confusion_matrix(self, G, d2):
        """

        :param G: Vector of group memberships, dummy variable form
        :param d2: Matrix of distances from group centers for every point that is to be predicted
        :return: conf_mat: the confusion matrix
                 ncc: number of correct classifications
                 pcc: proportion of correct classifications
        """
        confmat = np.zeros([self.K, self.K])
        Ghat = np.argmin(d2, axis=1)
        Gr = np.argmax(G, axis=1)
        n = len(G)
        for i in range(n):
            j = Gr[i]
            k = Ghat[i]
            confmat[j][k] = confmat[j][k] + 1
        ncc = np.trace(confmat)
        pcc = ncc / len(Gr)

        return confmat, ncc, pcc

    def LDA(self, X, G, d="Mahalanobis"):
        """
        :param X: Measurement matrix.
               G: Group matrix dummy form
               d: Distance metric. By default Mahalanobis, Euclidean can be requested by user.
        :return: pcc, fraction of correct classifications.
        """
        n = len(X)
        K = len(G[0])
        d2 = np.zeros([n, K])
        confmat = np.zeros([K, K])
        muG = np.linalg.lstsq(G.T @ G, G.T @ X, rcond=None)[0]
        Xc = np.zeros([n, len(X[0])])
        for k in range(n):
            if G[k][0] == 1:
                Xc[k] = X[k] - muG[0]
            else:
                Xc[k] = X[k] - muG[1]
        if d == "Mahalanobis":
            Sinv = np.linalg.pinv(Xc.T @ Xc)
            for i in range(n):
                for c in range(K):
                    d2[i, c] = (X[i] - muG[c]) @ Sinv @ (X[i] - muG[c]).T
        elif d == "Euclidean":
            for i in range(n):
                for c in range(K):
                    d2[i, c] = (X[i] - muG[c]) @ (X[i] - muG[c]).T

        Ghat = np.argmin(d2, axis=1)
        Gr = np.argmax(G, axis=1)
        for i in range(n):
            j = Gr[i]
            k = Ghat[i]
            confmat[j][k] = confmat[j][k] + 1
        pcc = np.trace(confmat) / n
        return confmat, pcc, muG

    def test_train(self, X, G, p):
        n = len(X)
        self.K = len(G[0])
        d2 = np.zeros([n, self.K])
        sample_size = int(np.round(n * p))
        training_idc = sample([k for k in range(n)], sample_size)
        test_idc = [k for k in range(n) if k not in training_idc]
        Xtrain = X[training_idc]
        Gtrain = G[training_idc]
        Xtest = X[test_idc]
        Gtest = G[test_idc]
        muG_train = np.linalg.lstsq(Gtrain.T @ Gtrain, Gtrain.T @ Xtrain, rcond=None)[0]

        Xc_train = np.zeros([sample_size, len(X[0])])
        for k in range(sample_size):
            if G[k][0] == 1:
                Xc_train[k] = Xtrain[k] - muG_train[0]
            else:
                Xc_train[k] = Xtrain[k] - muG_train[1]

        Sinvtrain = np.linalg.pinv(Xc_train.T @ Xc_train)
        for i in range(n - sample_size):
            for c in range(self.K):
                d2[i, c] = (Xtest[i] - muG_train[c]) @ Sinvtrain @ (Xtest[i] - muG_train[c]).T

        conf_mat, ncc, pcc = self.confusion_matrix(Gtest, d2)
        print(conf_mat)
        print(ncc)
        print(pcc)


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

    ins.test_train(X, G, .4)

  #  confmat, pcc, muG = ins.LDA(X, G)
  #  print(confmat)
  #  print(pcc)
