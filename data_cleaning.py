'''
Created on 25.05.2019
@author: Mostafa Abdelrashied
'''

import pandas as pd


def init_datetime(df: object) -> object:
    """
    Adding date and time in a single column and set it as an Index "Timezone Aware"
    :param df: Unformatted DataFrame
    :return: Formatted DataFrame with Datetime Index
    """
    df['hour'] = ["%02d" % x for x in df['hour']]
    df['date'] = df['date'].astype(str) + ' ' + df['hour'].astype(str)
    df.drop(columns=['instant', 'yr', 'mnth', 'workingday', 'hour'], inplace=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S.%f')
    df.date = df.date.dt.tz_localize('UTC').dt.tz_convert('Europe/Lisbon')
    df.set_index('date', inplace=True)
    return df


def cate_cols(df: object) -> object:
    """
    Reformatting the dtypes of the DataFrame columns for plotting
    :param df: DataFrame with integers as category indicators
    :return: DataFrame with strings as category indicators
    """
    dow_dict = {1: 'monday', 2: 'tuesday', 3: 'wednesday ', 4: 'thursday', 5: 'friday', 6: 'saturday', 0: 'sunday'}
    holiday_dict = {0: 'false', 1: 'true'}
    weather_dict = {1: 'clear', 2: 'misty', 3: 'light', 4: 'heavy'}
    month_to_season_dct = {
        1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn',
        12: 'winter'}
    df.replace({'weathersit': weather_dict, 'holiday': holiday_dict, 'weekday': dow_dict}, inplace=True)
    df['season'] = [month_to_season_dct.get(t_stamp.month) for t_stamp in df.index]
    df.loc[:, ['season', 'holiday', 'weekday', 'weathersit']] = df.loc[:,
                                                                ['season', 'holiday', 'weekday', 'weathersit']].astype(
        'category')
    return df


def retrieving_data(fname):
    """
    Reading Data from a CSV file to a pandas DataFrame
    :param fname: The absolute path for the input file
    :return: Dataframe with modified columns as needed for prediction and plotting
    """
    init_df = pd.read_csv(fname)
    init_df.rename(columns={'dteday': 'date', 'hr': 'hour'}, inplace=True)
    df = init_datetime(init_df)
    # df = cate_cols(df)  # Comment this line out when you need to include categories in prediction
    return df


if __name__ == "__main__":
    pass
