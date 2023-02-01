"""
In this file, I am implementing some classification modules that can be used for prediction.

LDA: Linear Discriminant Analysis.
    Builds a geometric model that will predict a measurement into any of n defined groups by minimizing distance to
    group center. Borders between the groups will be (n - 1)-dimensional hyperplanes. Lets user specify Euclidean
    distance, otherwise the more general Mahalanobis distance is used. Can be easily expanded to other distance metrics.
"""

import numpy  as np
import pandas as pd


class Classify(object):

    def __init__(self):
        pass





    def LDA(self, X, G, d="Mahalanobis"):
        """
        :param d: Distance metric. By default Mahalanobis, Euclidean can be requested by user.
        :return: pcc, fraction of correct classifications.
        """
        n = len(X)
        K = len(G[0])
        d2 = np.zeros([n, K])
        confmat = np.zeros([K, K])
        muG = np.linalg.lstsq(G.T@G, G.T@X, rcond=None)[0]
        #print(type(muG))
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
                    d2[i, c] = (X[i] - muG[c])@Sinv@(X[i] - muG[c]).T
        if d == "Euclidean":
            for i in range(n):
                for c in range(K):
                    d2[i, c] = (X[i] - muG[c])@(X[i] - muG[c]).T

        Ghat = np.argmin(d2, axis=1)
        Gr = np.argmax(G, axis=1)
        for i in range(n):
            j = Gr[i]
            k = Ghat[i]
            confmat[j][k] = confmat[j][k] + 1
        pcc = np.trace(confmat)/n
        return confmat, pcc



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


    confmat, pcc = ins.LDA(X, G)
    print(confmat)
    print(pcc)

