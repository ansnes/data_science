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
        n = len(self.X)
        K = len(self.G[0])
        d2 = np.zeros([n, K])
        confmat = np.zeros([K, K])
        muG = np.linalg.lstsq(self.G.T@self.G, self.G.T@self.X)
        print(muG)


if __name__ == "__main__":
    ins = Classify()
    file = "data/hotels.csv"
    data = pd.read_csv(file)


    # Wrangling. Removing the ID column and recoding the categorical variables
    meals = pd.get_dummies(data.type_of_meal_plan)
    rooms = pd.get_dummies(data.room_type_reserved)
    segment_mapping = {"Offline": 0, "Online": 1}
    data = data.assign(market_segment_type=data.market_segment_type.map(segment_mapping))
    cancelled = pd.get_dummies(data.booking_status)
    # Numeric data
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
                        data.market_segment_type,
                        data.repeated_guest,
                        data.no_of_previous_cancellations,
                        data.avg_price_per_room,
                        data.no_of_special_requests],
                       axis=1,
                       join="inner")

    print(n_data.dtypes)