'''
Created on 25.05.2019
@author: Mostafa Abdelrashied
'''

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from data_cleaning import retrieving_data

FIGWIDTH = 6.3 * 2
FIGHIGHT = FIGWIDTH / 3
FIGSIZE = [FIGWIDTH, FIGHIGHT]
FIGSIZE_carpet = [16, 16 / 3]


def prediction(df):
    """
    The prediction model function
    :param df:  Data Input
    """
    # df = df[df['cnt'] > 300]
    # df['cnt'] = (df['cnt'] - df['cnt'].min()) / (df['cnt'].max() - df['cnt'].min())
    # df = df[df['cnt'] < 0.4]
    features = ['temp', 'hum', 'season', 'holiday', 'weekday', 'weathersit',
                'windspeed']  # , 'season', 'holiday', 'weekday', 'weathersit','windspeed']
    targets = ['cnt']  # casual, registered
    # df = df.loc[:,['cnt','hum_value','temp_value']]
    x = df.loc[:, features]
    y = df.loc[:, targets]

    def poly_pred():
        """
        Creating a polynomial regression model
        """
        poly = PolynomialFeatures(degree=3)
        x_poly = poly.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_poly, y, test_size=0.2, random_state=12)

        model = LinearRegression()
        # model = GridSearchCV(model,param_grid,cv=5)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        # cv_results = cross_val_score(model, x, y, cv=10)
        # print(cv_results)
        # print(np.mean(cv_results))
        print(model.score(x_test, y_test))
        print(metrics.mean_squared_error(y_test, prediction))

        def actual_pred():
            """
            PLotting the actual vs prediction values
            """
            plt.scatter(y_test, prediction)
            plt.xlabel('Actual cnt')
            plt.ylabel('Predicted cnt')
            plt.show()

        actual_pred()

        def feature_coef():
            """
            Features selction using Ridge, LASOO and plotting its graphs (Ridge was removed)
            """
            fig, ax = plt.subplots(1, figsize=FIGSIZE)
            model = LinearRegression()
            lasso = Lasso(alpha=0.1)
            lasso_coef = lasso.fit(x_train, y_train).coef_
            sns.lineplot(x=features, y=lasso_coef[1:])
            # plt.xticks(range(len(features)), features, rotation=60)
            plt.ylabel('Coefficients')
            plt.show()

        feature_coef()

        def plot_pred():
            """
            PLotting the final prediction graph of two features (humidity and temperature) against prediction
            """
            fig, ax = plt.subplots(1, figsize=FIGSIZE)

            plt.scatter(x.loc[:, 'temp'].values, y.values, color='blue', alpha=0.7, label='Temperature')
            plt.scatter(x.loc[:, 'hum'].values, y.values, color='green', alpha=0.2, label='Humidity')
            # plt.scatter(x.loc[:,'hum'].values, y.values, color = 'green')
            plt.plot(np.linspace(df['temp'].min(), df['temp'].max(), len(prediction)), prediction, color='red',
                     label='Prediction')
            plt.legend()
            plt.title('Polynomial Regression')
            plt.xlabel('Variable')
            plt.ylabel('cnt')
            plt.text(x=0.02, y=0.7,
                     s='R^2: {}'.format(round(model.score(x_test, y_test), 2)),
                     backgroundcolor='white',
                     horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            plt.show()

        plot_pred()

    def mad(data, axis=None):
        """
        Calculating mean absolute deviation for all columns
        """
        out_list = []
        cols = data.columns
        for col in cols:
            val = np.mean(np.absolute(data[col] - np.mean(data[col], axis)), axis)
            out_list.append(val)
        return out_list

    mad_values = mad(df)
    poly_pred()
    print(mad_values)
    print(df.columns)
    # print("coef_pval:\n", stats.coef_pval(model, x, y))


if __name__ == "__main__":
    file_name = 'Bike-Sharing-Dataset/input_files/hour.csv'
    df = retrieving_data(file_name)
    prediction(df)
