'''
Created on 25.05.2019
@author: Mostafa Abdelrashied
'''

import matplotlib.dates as md
import matplotlib.pyplot as plt
import seaborn as sns
from genpy.plottools import carpet

from data_cleaning import retrieving_data

FIGWIDTH = 6.3 * 2
FIGHIGHT = FIGWIDTH / 3
FIGSIZE = [FIGWIDTH, FIGHIGHT]
FIGSIZE_carpet = [16, 16 / 3]


def carpet_plot(df):
    """
    Heatmap plot for a whole year indicating the behaviour for each hour for each day
    :param df: Input Data
    """
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    df = df.loc['2012', 'cnt'].values
    carpet(df, zlabel='cnt', sampling_step_width_sec=3600, savepath=None)


def sns_plots(data, xvalue, target, cat_col):
    """
    plotting multiple graphs for different targets using line plotting or bar plotting
    :param data: Input DataFrame
    :param xvalue: X-axis parameter
    :param target: Y-axis parameter
    :param cat_col: categorical column
    """
    # sns.barplot(x=df.index,y=target,hue=cat_col,data = df)
    fig, ax = plt.subplots(len(target), sharex=True, sharey=True, figsize=FIGSIZE)
    for num, item in enumerate(target):
        sns.lineplot(x=xvalue, y=item, hue=cat_col, data=data, ax=ax)

        # sns.boxplot(x=xvalue, y=item, hue=cat_col , data=df, ax=ax)

    plt.show()
    # sns.boxplot(x=cat_col, y=target, data=df)


def melt_plot(df):
    """
    melting a dataframe to be able to categorize casual and registered
    :param df: Data Input
    """
    fig, ax = plt.subplots(1, figsize=FIGSIZE)
    df = df.melt(id_vars=['season', 'holiday', 'weekday', 'weathersit', 'temp', 'hum', 'windspeed'],
                 value_vars=['casual', 'registered', 'cnt'],
                 var_name='type', value_name='count')
    # sns.barplot(data=df, x='season', y='count', hue='type')
    sns.barplot(data=df, x='holiday', y='count', hue='type')
    # myFmt = mdates.DateFormatter('%b-%y')
    # ax.xaxis.set_major_formatter(myFmt)
    plt.show()


def hetmap_corr(df):
    """
    Pearson correlation graph
    :param df: Data Input
    """
    fig, ax = plt.subplots(1, figsize=FIGSIZE)
    # sns.pairplot(data=df)
    df = df.loc[:, ['temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']]
    # df2 = df.melt(id_vars='season', 
    #           value_vars=['temp', 'hum','windspeed'], 
    #           var_name='variable', value_name='count')
    print(df.corr())
    # sns.lineplot(df.loc[:,['temp','hum','windspeed','cnt']])
    # scatter_matrix(df, diagonal='kde', alpha=0.1,figsize=(12,12))    
    sns.heatmap(df.corr(), cmap='RdYlGn', ax=ax, square='auto', vmin=-1, vmax=1)
    # carpet(df['cnt'], zlabel='Count', cmap='jet', sampling_step_width_sec=3600,savepath=None)
    plt.show()


def step_plot(df):
    """
    Step curve for hourly behaviour
    :param df: Input Data
    """
    fig, ax = plt.subplots(1, figsize=FIGSIZE)

    df_win = df['15-01-2011']
    df_sum = df['15-07-2011']
    df_spr = df['15-04-2011']
    df_fal = df['15-11-2011']
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    sns.lineplot(x=df_win.index, y='cnt', data=df_win, drawstyle='steps-post', ax=ax, label='Winter')
    sns.lineplot(x=df_win.index, y='cnt', data=df_sum, drawstyle='steps-post', ax=ax, label='Summer')
    sns.lineplot(x=df_win.index, y='cnt', data=df_spr, drawstyle='steps-post', ax=ax, label='Spring')
    sns.lineplot(x=df_win.index, y='cnt', data=df_fal, drawstyle='steps-post', ax=ax, label='Autumn')
    plt.legend()
    xfmt = md.DateFormatter('%H')
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Hour')
    plt.show()


if __name__ == "__main__":
    file_name = 'Bike-Sharing-Dataset/input_files/hour.csv'
    df = retrieving_data(file_name)
    hetmap_corr(df)
    # x = df.loc['2011','cnt']
    # y = df.loc['2012','cnt']
    # percentage = (np.mean(y)-np.mean(x))/np.mean(x)
    # print(np.mean(x),np.mean(y),percentage)
    # sns_plots(data=df,xvalue='weathersit',target=['count'], cat_col='type')
    # sns_plots(data=df,xvalue='holiday',target=['cnt'], cat_col='season')
