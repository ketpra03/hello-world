# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# The dict statistics has a lot of information and I figure it might be easier
# for me to include what it's keys entails. As a quick reminder, to see what 
# keys a dict has, such as statistics, just enter statistics.keys()
# 
# Keys of statistics:
#
# Returns - This key has the monthly, quarterly, and yearly returns for each
#           ETF as a DataFrame
# Volatility - This key has monthly, quarterly, and yearly volatilities for
#              each ETF as a DataFrame
# Skewness - This key has monthly, quarterly, and yearly skewness for
#            each ETF as a DataFrame
# Kurtosis - This key has monthly, quarterly, and yearly kurtosis for
#            each ETF as a DataFrame
# Correlations - This key has monthly, quarterly, and yearly correlations for
#                each ETF pair as a DataFrame
# Regressions - This key has monthly, quarterly, and yearly regressions for 
#               each ETF pair
# Differentials - This key has monthly, quarterly, and yearly return 
#                 differentials for each ETF pair as a DataFrame
# Differentials Vol - This key has monthly, quarterly, and yearly volatility of
#                     the return differentials for each ETF pair as a DataFrame
# Differentials Skew - This key has monthly, quarterly, and yearly skewness of
#                     the return differentials for each ETF pair as a DataFrame
# Differentials Kurt - This key has monthly, quarterly, and yearly kurtosis of
#                     the return differentials for each ETF pair as a DataFrame
# Max_Min_Mean_Abs_Diff - This key has monthly, quarterly, and yearly max, min,
#                         and mean of the absolute value for each ETF pair 
#                         as a DataFrame
# Positive - This key has the monthly, quarterly, and yearly, max, min, and 
#            mean of the positive differentials for each ETF pair as a 
#            DataFrame
# Negative - This key has the monthly, quarterly, and yearly, max, min, and 
#            mean of the negative differentials for each ETF pair as a 
#            DataFrame
# Histograms - This key plots histograms of monthly, quarterly, and yearly 
#              positive and negative return differentials for each ETF pair
# Predictive Averages - This key has the one period ahead, whether it be 
#                       monthly, quarterly, and yearly, average return value
#                       for each ETF pair differential as a DataFrame
# Predictive Histograms - This key has the one period ahead, whether it be 
#                         monthly, quarterly, and yearly, average return value
#                         of each ETF pair return differential for both 
#                         positive and negative return values, which are keys 
#                         of the ETF differentials
# Decile Buckets - The key has the decile buckets of monthly, quarterly, and
#                  yearly return differentials for each ETF pair for both
#                  positive and negative return values as an array (length 11,
#                  including borders)
# Decile Counts - The key has the decile counts of monthly, quarterly, and
#                 yearly return differentials for each ETF pair for both
#                 positive and negative return values as an array (length 11,
#                 including borders)
# Correlations Strat - This key has the average returns, volatility, risk
#                      adjusted return, bin counts, and bins at monthly, 
#                      quarterly, and yearly for each correlation decile of 
#                      each ETF pair as a DataFrame
# Differentials Vol Strat - This key has the average returns, volatility, risk
#                           adjusted return, bin counts, and bins at monthly, 
#                           quarterly, and yearly for each volatility decile of
#                           each ETF return differential pair as a DataFrame
# Differentials Skew Strat - This key has the average returns, volatility, risk
#                            adjusted return, bin counts, and bins at monthly, 
#                            quarterly, and yearly for each skewness decile of
#                            each ETF return differential pair as a DataFrame
# Differentials Kurt Strat - This key has the average returns, volatility, risk
#                            adjusted return, bin counts, and bins at monthly, 
#                            quarterly, and yearly for each kurtosis decile of
#                            each ETF return differential pair as a DataFrame
# Volatility Buckets - This key has the volatility buckets at each frequency,
#                      monthly, quarterly, and yearly for each ETF as an array
# Skewness Buckets - This key has the skewness buckets at each frequency,
#                    monthly, quarterly, and yearly for each ETF as an array
# Kurtosis Buckets - This key has the kurtosis buckets at each frequency,
#                    monthly, quarterly, and yearly for each ETF as an array
# Volatility Buckets Backtest - This key gives the results of performing the
#                               volatility buckets backtest at each frequency,
#                               monthly, quarterly, and yearly, which compares
#                               the individual volatility buckets of each ETF
#                               pair
# Skewness Buckets Backtest - This key gives the results of performing the
#                             skewness buckets backtest at each frequency,
#                             monthly, quarterly, and yearly, which compares
#                             the individual skewness buckets of each ETF pair
# Kurtosis Buckets Backtest - This key gives the results of performing the
#                             kurtosis buckets backtest at each frequency,
#                             monthly, quarterly, and yearly, which compares
#                             the individual kurtosis buckets of each ETF pair
# Volatility Buckets Test - This key gives the results of performing the
#                           volatility buckets test at each frequency, monthly,
#                           quarterly, and yearly, which compares the
#                           individual volatility buckets of each ETF pair, 
#                           incorporating information from the backtest                  
# Skewness Buckets Test - This key gives the results of performing the skewness
#                         buckets test at each frequency, monthly, quarterly, 
#                         and yearly, which compares the individual skewness 
#                         buckets of each ETF pair, incorporating information
#                         from the backtest
# Kurtosis Buckets Test - This key gives the results of performing the kurtosis
#                         buckets backtest at each frequency, monthly, 
#                         quarterly, and yearly, which compares the individual 
#                         kurtosis buckets of each ETF pair, incorporating 
#                         information from the backtest
# The next set of strategies have the following theme: The first set of words 
# contains the statistic as to how the ETFs were grouped into deciles, such as
# Differentials Vol Strat would mean that we looked at the volatility of the
# spreads and grouped based on that. After the word Strat, the next set of 
# words describes the type of overlay I used. The last word is either Backtest
# or Test.

import numpy as np
from datetime import datetime, date, timedelta
import calendar
from yahoo_finance import Share
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as ss
import itertools as it
import matplotlib.pyplot as plt
import math
import os
import csv
from pandas_datareader import data

# First let's obtain the last trading date in December 2006 for return
# calculation

tickers = ['GLD', 'GDX', 'SLV']
data_source = 'google'

initial_date = datetime.strptime('2006-12-31', '%Y-%m-%d')

while True:
    trial = data.DataReader(tickers, 
                            data_source, 
                            initial_date.strftime('%Y-%m-%d'),
                            initial_date.strftime('%Y-%m-%d'))
    if not trial.empty:
        break
    else:
        initial_date -= timedelta(days = 1)
        
del trial

# Next let's obtain all the price data starting with daily

relevant_columns = ['Close']
price_data = pd.DataFrame()

for ticker in tickers:
    temp = data.DataReader(ticker,
                           data_source,
                           initial_date.strftime('%Y-%m-%d'),
                           '2016-12-31')['Close']
    temp = temp.rename(ticker)
    if price_data.shape[0] > 1:
        price_data = pd.concat([price_data, temp], axis = 1)
    elif price_data.shape[0] == 1:
        price_data = pd.merge(price_data, temp, on = 'Date')
    else:
        price_data = temp

del temp, ticker, relevant_columns

price_data.loc[datetime.strptime('2008-12-05', '%Y-%m-%d'), 'SLV'] = 9.40
price_data.to_csv('Daily Prices.csv')
price_data.reset_index(drop = False, inplace = True)

# Extracting out monthly, quarterly, and annual price data

price_data['Year'] = price_data['Date'].dt.year
price_data['Quarter'] = ((price_data['Date'].dt.month - 1) // 3 + 1)
price_data['Month'] = price_data['Date'].dt.month
price_data['Day'] = price_data['Date'].dt.day          
                
monthly_grouping = pd.DataFrame(price_data.groupby(
    ['Year', 'Month'])['Day'].max()).reset_index()
monthly_grouping = monthly_grouping.rename(columns = {'Day': 'Max Day'})
monthly_prices = pd.merge(price_data, monthly_grouping, on = ['Year', 'Month'])
monthly_prices = monthly_prices[
    monthly_prices['Day'] == monthly_prices['Max Day']]
monthly_prices = monthly_prices.drop(['Year', 'Quarter', 'Day', 'Max Day'], 1)
quarterly_prices = monthly_prices[(monthly_prices['Month'] % 3) == 0]
yearly_prices = quarterly_prices[quarterly_prices['Month'] == 
                                 max(quarterly_prices['Month'])]
del monthly_grouping

price_frequencies = [monthly_prices, quarterly_prices, yearly_prices]
[df.drop('Month', axis = 1, inplace = True) for df in price_frequencies]

# Saving a copy of monthly, quarterly, and annual price data

monthly_prices.to_csv('Monthly Prices.csv')
quarterly_prices.to_csv('Quarterly Prices.csv')
yearly_prices.to_csv('Yearly Prices.csv')
# Would have done this programmatically but referring to variable name is not
# recommended

#Calculating returns
def price_returns(dataframe, column_returns, column_sort = False, 
                  column_to_sort = None, ascending = True):
    # Calculate returns of a dataframe - first should any particular column
    # be sorted? - second we calculate returns
    #
    # dataframe: calculate returns from here
    # column_returns: columns to calculate returns from - these will be
    #                 mutually exclusive of the columns_to_sort
    # column_sort: binary variable if a column will be sorted or not - default
    #              is False
    # column_to_sort: columns which will be sorted before returns are 
    #                 calculated
    # ascending: columns to sort in ascending order or not
    #
    # Haven't tested with multiple column_to_sort, but asides from Date can't
    # think of anything else right away 
    # I am also aware this function is not preserving columns that are not
    # going to be used for return calculation or the basis for sorting
    # Error-Checking
    if dataframe.shape[1] == 0:
        print('There has to be some data to calculate returns.')
        return
    if dataframe.shape[0] <= 1:
        print('There has to be more than 1 data point to calculate returns.')
        return
    if column_sort:
        if set(column_to_sort).intersection(column_returns):
            print('The columns specified to be sorted should not be used ' +
                  'to calculate returns.')
            return
        if set(column_to_sort).intersection(dataframe.columns.values):
            dataframe = dataframe.sort_values(column_to_sort, 
                                              ascending = ascending)
        else:
            print('The columns specified to be sorted are not columns of ' +
                  'given dataframe.')
            return
    else:
        if column_to_sort != None:
            print('Sorting of columns will not be done, as column_sort was ' +
                  'either left out, having a default value of False or was ' +
                  'set as False.')
            
    if (set(column_returns).intersection(dataframe.columns.values) != 
          set(column_returns)):
        print('The columns specified for return calculation were not found ' +
              'in given dataframe.')
        return
    answer = dataframe[column_returns].pct_change(1)
    if column_sort:
        answer[column_to_sort] = dataframe[column_to_sort]
        answer = answer[answer.columns.tolist()[-len(column_to_sort):] + 
                        answer.columns.tolist()[:-len(column_to_sort)]]
    return answer.iloc[1:]

#Calculating higher order moments for analysis
daily_returns = price_returns(price_data, tickers, True, 
                              ['Date', 'Year', 'Quarter', 'Month'], True)


statistics = {}

m_functions = [lambda x: np.prod(1 + x) - 1, np.std, ss.skew, ss.kurtosis]
m_arguments = [{}, {'ddof':1}, {'bias':False}, {'fisher':True, 'bias':False}]
frequencies = ['Month', 'Quarter', 'Year']

for label, function, argument in zip(['Returns', 'Volatility', 'Skewness', 
                                      'Kurtosis'], m_functions, m_arguments):
    statistics[label] = {}
    for index, prices in zip(range(3, 0, -1), price_frequencies):
        data = daily_returns.groupby(frequencies[::-1][:index]).agg(
            function, **argument).reset_index().drop(frequencies[::-1][index:], 
            axis = 1, inplace = False)
        data['Date'] = prices['Date'][1:].iloc[::-1].reset_index()['Date']
        data = data.drop(frequencies[::-1][:index], axis = 1, inplace = False)
        data = data[['Date', 'GLD', 'GDX', 'SLV']]
        statistics[label].update({frequencies[::-1][index - 1]: data})
del label, function, argument, index, prices, data

statistics['Correlations'] = {}

#Obtaining Correlations
for frequency, index in zip(frequencies, np.arange(3, 0, -1)):
    statistics['Correlations'].update({frequency: {}})
    for ticker_pair in list(it.combinations(tickers, 2)):
        a_name = '_'.join(ticker_pair)
        statistics['Correlations'][frequency][a_name] = pd.DataFrame(columns =
            ('Date', 'Correlation_' + a_name))
        for name, group in daily_returns.groupby(frequencies[::-1][:index]):
            statistics['Correlations'][frequency][a_name] = statistics[
                'Correlations'][frequency][a_name].append({
                'Date': max(group['Date']), 
                'Correlation_' + a_name: ss.pearsonr(group[ticker_pair[0]], 
                                                     group[ticker_pair[1]])[
                                                         0]}, 
                    ignore_index = True)
del frequency, index, a_name, name, group

#Performing Regressions and Differntials
labels = ['Regressions', 'Differentials', 'Differentials Vol', 
          'Differentials Skew', 'Differentials Kurt', 'Max_Min_Mean_Abs_Diff', 
          'Positive', 'Negative', 'Histograms', 'Predictive Averages', 
          'Predictive Histograms', 'Decile Buckets', 'Decile Counts', 
          'Correlations Strat', 'Differentials Vol Strat', 
          'Differentials Skew Strat', 'Differentials Kurt Strat']

for label in labels:
    statistics[label] = {}
    for frequency in frequencies:
        statistics[label].update({frequency: {}})
del label, frequency

backtest_period = datetime.strptime('01/01/2013', '%m/%d/%Y')
backtest_test = {'start': datetime.strptime('02/01/2007', '%m/%d/%Y'),
                 'end': datetime.strptime('02/01/2013', '%m/%d/%Y')}
decile_bins = 10
decile_labels = ['Decile ' + x for x in np.char.mod(
    '%d', np.arange(1, decile_bins + 1))]
one_period_ahead = 1

#quintile_bins = 5
#quintile_labels = ['Quintile ' + x for x in np.char.mod(
#    '%d', np.arange(1, quintile_bins + 1))]

ticker_pair_names = ['_'.join(tick) 
    for tick in list(it.combinations(tickers, 2))]

signs = ['Positive', 'Negative']

for frequency, index in zip(frequencies, np.arange(len(frequencies), 0, -1)):
    for ticker_pair in list(it.combinations(tickers, 2)):
        the_data = statistics['Returns'][frequency]
        statistics['Regressions'][frequency][
            ' ~ '.join(ticker_pair)] = smf.ols(
                formula = ' ~ '.join(ticker_pair), 
                                    data = the_data).fit().summary()
        ticker_pair_name = '_'.join(ticker_pair)
        statistics['Differentials'][frequency].update({ticker_pair_name: 
            pd.DataFrame({
                'Date': the_data['Date'], 
                '_minus_'.join(ticker_pair): the_data[ticker_pair[0]] - 
                    the_data[ticker_pair[1]]})})
        relevant_data = statistics['Differentials'][frequency][
            ticker_pair_name]
        daily_differential = pd.DataFrame({
            'Year': daily_returns['Year'], 
             'Quarter': daily_returns['Quarter'], 
             'Month': daily_returns['Month'], 
             '_minus_'.join(ticker_pair): daily_returns[ticker_pair[0]] -
                 daily_returns[ticker_pair[1]]})
        for diff, func, args in zip(['Differentials Vol', 'Differentials Skew',
                                     'Differentials Kurt'], m_functions[1:], 
                                    m_arguments[1:]):
            statistics[diff][frequency].update({
                ticker_pair_name: pd.DataFrame({
                    'Date': the_data['Date'], 
                    '_minus_'.join(ticker_pair): daily_differential.groupby(
                        frequencies[::-1][:index])[
                            '_minus_'.join(ticker_pair)].apply(func, 
                                **args).reset_index().drop(frequencies[::-1][
                                    :index], axis = 1, inplace = False)[
                                        '_minus_'.join(ticker_pair)]})})
        relevant_series = relevant_data['_minus_'.join(ticker_pair)]
        statistics['Max_Min_Mean_Abs_Diff'][frequency][
            ticker_pair_name] = pd.DataFrame({
                'Max': max(relevant_series), 
                'Min': min(relevant_series), 
                'Mean_Absolute': relevant_series.abs().mean()}, index = [0])
        for a_key, a_series in zip(signs, 
                                   [relevant_series[relevant_series >= 0],
                                    relevant_series[relevant_series < 0]]):
            statistics[a_key][frequency][ticker_pair_name] = pd.DataFrame({
                'Max': max(a_series),
                'Min': min(a_series),
                'Mean': a_series.mean()}, index = np.arange(1))
        hist_plot, axis = plt.subplots()
        backtest_series = relevant_data.loc[
            relevant_data['Date'] < backtest_period][
                '_minus_'.join(ticker_pair)]
        (positive_counter, positive_bins) = axis.hist(backtest_series[
            backtest_series >= 0], bins = decile_bins, color = 'blue')[:2]
        (negative_counter, negative_bins) = axis.hist(backtest_series[
            backtest_series < 0], bins = decile_bins, color = 'red')[:2]
        axis.set_title('Histogram of ' + ' minus '.join(ticker_pair) + ' ' + 
                       frequency + 'ly Returns from 2007-2012')
        axis.set_xlabel(' minus '.join(ticker_pair))
        axis.set_ylabel('Frequency of Returns')
        statistics['Histograms'][frequency][ticker_pair_name] = hist_plot
        statistics['Predictive Averages'][frequency][ticker_pair_name] = {
            a_key: pd.DataFrame(0, index = np.arange(1), 
                                columns = decile_labels) for a_key in signs}
        predictive_data = statistics['Predictive Averages'][frequency][
                ticker_pair_name]
        for s_label in ['Predictive Histograms', 'Decile Buckets', 
                        'Decile Counts']:
            statistics[s_label][frequency][ticker_pair_name] = {}
        for a_key, a_counter, a_bin in zip(signs, 
                                           [positive_counter, 
                                            negative_counter], 
                                           [positive_bins, negative_bins]):
            for decile in np.arange(decile_bins):
                if decile != (decile_bins - 1):
                    relevant_indices = (backtest_series[
                        (backtest_series >= a_bin[decile]) & 
                        (backtest_series < a_bin[decile + 1])]).index
                else:
                    relevant_indices = (backtest_series[
                        (backtest_series >= a_bin[decile]) &
                        (backtest_series <= a_bin[decile + 1])]).index 
                relevant_indices += one_period_ahead
                predictive_data[a_key]['Decile ' + str(decile + 1)] = sum(
                    relevant_series[relevant_indices])
            statistics['Decile Buckets'][frequency][ticker_pair_name][
                a_key] = a_bin
            statistics['Decile Counts'][frequency][ticker_pair_name][
                a_key] = pd.DataFrame([a_counter], columns = decile_labels, 
                                    index = np.arange(1))
            predictive_data[a_key] /= a_counter
            predictive_data[a_key].replace({math.inf: float('NaN')},
                                           inplace = True)
            predictive_data[a_key].replace({-math.inf: float('NaN')},
                                           inplace = True)
            bar_plot, bar_axis = plt.subplots()
            bar_axis.bar(np.arange(decile_bins), 
                         predictive_data[a_key].loc[0], 0.3, color = 'y', 
                         align = 'center')
            bar_axis.set_xticks(np.arange(decile_bins))
            bar_axis.set_xticklabels(decile_labels)
            bar_axis.set_xlabel(a_key + ' Return Buckets')
            bar_axis.set_ylabel('Average Future ' + frequency + ' Return')
            bar_axis.set_title('Average Future ' + frequency + ' Return for ' +
                               a_key + ' Return Buckets of 2007 - 2012 for ' +
                               ' minus '.join(ticker_pair))
            statistics['Predictive Histograms'][frequency][
                ticker_pair_name].update({a_key: bar_plot})
        # Extra Rules developed
        for pred in ['Correlations', 'Differentials Vol', 'Differentials Skew',
                     'Differentials Kurt']:
            relevant_stat = statistics[pred][frequency][ticker_pair_name]
            stat_plot, stat_axis = plt.subplots()
            backtest_stat = relevant_stat.loc[
                relevant_stat['Date'] < backtest_period][[col for col in 
                    relevant_stat.columns.values if col not in ['Date']]]
            (stat_counter, stat_bins) = stat_axis.hist(
                backtest_stat.iloc[:, 0], bins = decile_bins)[:2]
            statistics[pred + ' Strat'][frequency][ticker_pair_name] = {
                a_key: pd.DataFrame(0, index = np.arange(1), 
                                    columns = decile_labels) for a_key in [
                                    'Average Returns', 'Volatility', 
                                    'Risk Adjusted Return']}
            for decile in np.arange(decile_bins):
                if decile != (decile_bins - 1):
                    stat_indices = np.where((backtest_stat[
                        backtest_stat.columns.values] >= stat_bins[decile]) &
                        (backtest_stat[backtest_stat.columns.values] < 
                            stat_bins[decile + 1]))[0]
                else:
                    stat_indices = np.where((backtest_stat[
                        backtest_stat.columns.values] >= stat_bins[decile]) &
                        (backtest_stat[backtest_stat.columns.values] <= 
                            stat_bins[decile + 1]))[0]
                stat_indices += one_period_ahead
                statistics[pred + ' Strat'][frequency][ticker_pair_name][
                    'Average Returns']['Decile ' + str(decile + 1)] = sum(
                        relevant_series[stat_indices])
                statistics[pred + ' Strat'][frequency][ticker_pair_name][
                    'Volatility']['Decile ' + str(decile + 1)] = np.std(
                        relevant_series[stat_indices], ddof = 1)
            statistics[pred + ' Strat'][frequency][ticker_pair_name][
                'Average Returns'] /= stat_counter
            statistics[pred + ' Strat'][frequency][ticker_pair_name][
                'Risk Adjusted Return'] = statistics[pred + ' Strat'][
                    frequency][ticker_pair_name][
                        'Average Returns'] / statistics[pred + ' Strat'][
                            frequency][ticker_pair_name]['Volatility']
            for a_key in ['Average Returns', 'Volatility', 
                          'Risk Adjusted Return']:
                statistics[pred + ' Strat'][frequency][ticker_pair_name][
                    a_key].replace({math.inf: float('NaN')}, inplace = True)
                statistics[pred + ' Strat'][frequency][ticker_pair_name][
                    a_key].replace({-math.inf: float('NaN')}, inplace = True)
            statistics[pred + ' Strat'][frequency][ticker_pair_name][
                'Bin Counts'] = pd.DataFrame([(stat_counter)], 
                    index = np.arange(1), columns = decile_labels)
            statistics[pred + ' Strat'][frequency][ticker_pair_name][
                'Bins'] = stat_bins
        for ind_stat in ['Volatility', 'Skewness', 'Kurtosis']:
            relevant_stat = statistics[ind_stat][frequency].loc[
                statistics[ind_stat][frequency]['Date'] < backtest_period]
            relevant_stat = relevant_stat[[ticker_pair[0], ticker_pair[1]]]
            stat_plot, stat_axis = plt.subplots()
            (stat_counter, stat_bins) = stat_axis.hist(
                backtest_stat.iloc[:, 0], bins = decile_bins)[:2]

del a_bin, a_counter, a_key, a_series, args, backtest_series, backtest_stat
del decile, diff, frequency, index, negative_bins, negative_counter, s_label
del positive_bins, positive_counter, pred, predictive_data, relevant_data
del relevant_series, relevant_stat, stat_bins, stat_counter, stat_indices
del ticker_pair, ticker_pair_name, backtest_test, the_data

# for better presentation of the results in statistics dict
def label_creater(levels_per_layer):
    # This function creates a list of lists, in which all the inner lists
    # have the same length and is to be used for multi-index dataframes, for
    # the very purpose of creating multi-indexes.
    # levels_per_layer: a list structure that should not be empty or have any 
    #                   non-positive numbers, it expressly states how many 
    #                   levels a layer has, starting from the outermost going 
    #                   to innermost
    #                   
    # Error-checking:
    if len(levels_per_layer) == 0:
        print('There are no layers and so no levels for layers either.')
        return
    if len(np.where(np.array(levels_per_layer)[0] <= 0)) == 0:
        print('All layers must have positive levels')
        return
    if not all(isinstance(x, int) for x in levels_per_layer):
        print('All levels must be integers.')
        return
    answer = []
    for index in np.arange(len(levels_per_layer)):
        if index == 0:
            temp = sorted(list(np.arange(levels_per_layer[index])) * np.prod(
                levels_per_layer[(index + 1):]))
        elif index == (len(levels_per_layer) - 1):
            temp = np.prod(levels_per_layer[:index]) * list(np.arange(
                levels_per_layer[index]))
        else:
            temp = np.prod(levels_per_layer[:index]) * sorted(list(
                np.arange(levels_per_layer[index])) * np.prod(levels_per_layer[
                    (index + 1):]))
        answer.append(temp)
    return answer

#Important Result
predictive_results = {}

decile_indices = pd.MultiIndex(levels = [frequencies, ticker_pair_names, 
                                         signs], 
                               labels = label_creater([len(frequencies),
                                                       len(ticker_pair_names),
                                                       len(signs)]))

for label, file_name in zip(['Predictive Averages', 'Decile Counts'], 
                            ['Average Decile Returns', 'Average Count']):
    list_holder = []
    [list_holder.append(statistics[label][f_key][t_key][s_key]) 
        for f_key in statistics[label].keys() 
        for t_key in statistics[label][f_key].keys()
        for s_key in statistics[label][f_key][t_key].keys()]
    data_frame = pd.concat(list_holder, keys = decile_indices)
    predictive_results[label] = pd.DataFrame(
        data_frame.values, index = data_frame.index.droplevel(3), 
        columns = data_frame.columns)
    predictive_results[label].index.names = ['Frequency', 'Ticker_Pairs', 
                                             'Signs']
    predictive_results[label].to_csv(file_name + '.csv', index = True, 
                                     header = True)
del label, file_name, list_holder, data_frame

#Important Result
moment_results = {}

table_stats = ['Risk Adjusted Return', 'Bin Counts']

strategies = ['Correlations Strat', 'Differentials Vol Strat', 
              'Differentials Skew Strat', 'Differentials Kurt Strat']

decile_indices = pd.MultiIndex(levels = [strategies, frequencies, 
                                           ticker_pair_names],
                                 labels = label_creater([len(strategies),
                                                         len(frequencies),
                                                         len(ticker_pair_names)
                                                        ]))

for label, file_name in zip(table_stats, 
                            ['Strategy Returns', 'Strategy Counts']):
    list_holder = []
    [list_holder.append(statistics[strategy][f_key][t_key][label]) 
        for strategy in strategies
        for f_key in statistics[strategy].keys()
        for t_key in statistics[strategy][f_key].keys()]
    data_frame = pd.concat(list_holder, keys = decile_indices)
    moment_results[label] = pd.DataFrame(data_frame.values, 
                                         index = data_frame.index.droplevel(3),
                                         columns = data_frame.columns)
    moment_results[label].index.names = ['Strategy', 'Frequency', 
                                         'Ticker_Pair']
    moment_results[label].to_csv(file_name + '.csv', index = True, 
                                 header = True)
del label, file_name, list_holder, data_frame 

#Different Trading Rules
# I have established trading rules based on the following: 
# For the predictive returns in both both negative and positive decile buckets,
# I wanted to see a high return and with good frequency associated with it.
# Obviously, with longer time frames, this frequency measure is difficult to
# come by, but returns certainly have a chance to be much higher than compared
# to shorter time frames. 
# For monthly returns, I first look to see if they are higher than 1% and if
# the frequency is about greater than 5% of the backtested data of 72 months,
# so greater than or equal to 7 months. Since there are more months, there
# is a greater chance for variety of returns to go into different buckets, as
# compared to quarterly or yearly. For quarterly, I look to see if returns
# generated were higher than 2.5% and if frequency is greater than or equal to 
# 4 months. Compouneded returns for months in a quarter is certainly going to 
# be greater than 2%, but there is more of a chance for volatility to be there,
# moderating the effects of returns. Additionally, with lesser data, a higher
# fraction of total backtested returns in a bucket is warranted. Admittedly,
# there are 20 buckets, so concentration of returns in a bucket with a high
# or low returns can be be informative of a signal. For yearly returns, I look
# to see if returns generated were greater than 8% with frequency of 1 or
# greater. For yearly returns, there are only 6 years in the backtest, so it
# doesn't make much sense to look for concentration, as there are 20 available
# buckets.
rel_pred_average = {'Month': 0.01, 'Quarter': 0.025, 'Year': 0.08}
rel_pred_counts = {'Month': 7, 'Quarter': 4, 'Year': 1}
test_set_returns = {}
for frequency in frequencies:
    test_set_returns[frequency] = {}
    for ticker_pair in ticker_pair_names:
        test_set_returns[frequency][ticker_pair] = {}
        for sign in signs:
            test_set_returns[frequency][ticker_pair][sign] = {}

for frequency in frequencies:
    for ticker_pair in ticker_pair_names:
        for sign in signs:
            large_returns_index = np.where(predictive_results[
                'Predictive Averages'].loc[[(
                    frequency, ticker_pair, sign)]].abs() >= rel_pred_average[
                        frequency])[1]
            large_counts_index = np.where(predictive_results[
                'Decile Counts'].loc[[(
                    frequency, ticker_pair, sign)]] >= rel_pred_counts[
                        frequency])[1]
            index_intersect = list(set(large_returns_index).intersection(
                large_counts_index))
            relevant_data = statistics['Differentials'][frequency][ticker_pair]
            relevant_data = relevant_data.loc[
                relevant_data['Date'] > backtest_period]
            relevant_series = relevant_data[
                ticker_pair.replace('_', '_minus_')]
            if index_intersect:
                for index in index_intersect:
                    relevant_indices = (relevant_series[
                        (relevant_series >= statistics['Decile Buckets'][
                            frequency][ticker_pair][sign][index]) & 
                        (relevant_series < statistics['Decile Buckets'][
                                frequency][ticker_pair][sign][
                                    index + 1])]).index
                    relevant_indices += one_period_ahead
                    relevant_indices = relevant_indices[
                        relevant_indices <= max(relevant_series.index)]
                    test_set_returns[frequency][ticker_pair][sign][
                        'Decile ' + str(index + 1)] = float((np.prod(
                            relevant_series[
                                relevant_indices] + 1) - 1) * np.sign(
                                    predictive_results[
                                        'Predictive Averages'].loc[[(
                                            frequency, ticker_pair, 
                                            sign)], 'Decile ' + str(index + 1)]
                    ))

#Important Result
#test_set_returns - keys are frequency, ticker_pair, and sign in that order

m_test_rar = {'Month': 1/math.sqrt(12), 'Quarter': 0.5, 'Year': 1}
m_test_returns = {}
for strat in ['Correlations Strat', 'Differentials Vol Strat', 
              'Differentials Skew Strat', 'Differentials Kurt Strat']:
    m_test_returns[strat] = {}
    for frequency in frequencies:
        m_test_returns[strat][frequency] = {}
        for ticker_pair in ticker_pair_names:
            m_test_returns[strat][frequency][ticker_pair] = {}

for strat in ['Correlations Strat', 'Differentials Vol Strat', 
              'Differentials Skew Strat', 'Differentials Kurt Strat']:
    for frequency in frequencies:
        for ticker_pair in ticker_pair_names:
            large_returns_index = np.where(moment_results[
                'Risk Adjusted Return'].loc[[(
                    strat, frequency, ticker_pair)]].abs() >= m_test_rar[
                        frequency])[1]
            large_counts_index = np.where(moment_results[
                'Bin Counts'].loc[[(
                    strat, frequency, ticker_pair)]] >= rel_pred_counts[
                        frequency])[1]
            index_intersect = list(set(large_returns_index).intersection(
                large_counts_index))
            relevant_data = statistics[strat.replace(' Strat', '')][frequency][
                ticker_pair]
            relevant_data = relevant_data.loc[
                relevant_data['Date'] > backtest_period]
            if strat == 'Correlations Strat':
                relevant_series = relevant_data[
                    strat.replace('s Strat', '_' + ticker_pair)]
            else:
                relevant_series = relevant_data[
                    ticker_pair.replace('_', '_minus_')]
            if index_intersect:
                for index in index_intersect:
                    relevant_indices = (relevant_series[
                        (relevant_series >= statistics[strat][frequency][
                            ticker_pair]['Bins'][index]) & 
                        (relevant_series < statistics[strat][frequency][
                            ticker_pair]['Bins'][index + 1])]).index
                    relevant_indices += one_period_ahead
                    relevant_indices = relevant_indices[
                        relevant_indices <= max(relevant_series.index)]
                    m_test_returns[strat][frequency][ticker_pair][
                        'Decile ' + str(index + 1)] = float((np.prod(
                            statistics['Differentials'][frequency][
                                ticker_pair][ticker_pair.replace(
                                    '_', '_minus_')][
                                        relevant_indices] + 1) - 1) * np.sign(
                            moment_results['Risk Adjusted Return'].loc[[(
                                strat, frequency, ticker_pair)], 
                                'Decile ' + str(index + 1)]))

#Important Result
#m_test_returns - keys are strat, frequency, and ticker_pair in that order

#Creating Buckets for each of Volatility, Skewness, and Kurtosis - individual
#ETFs

for stat in ['Volatility', 'Skewness', 'Kurtosis']:
    statistics[stat + ' Buckets'] = {}
    for frequency in frequencies:
        statistics[stat + ' Buckets'][frequency] = {}
        relevant_data = statistics[stat][frequency]
        relevant_data = relevant_data.loc[
            relevant_data['Date'] < backtest_period]
        for ticker in tickers:
            relevant_series = relevant_data[ticker]
            stat_plot, stat_axis = plt.subplots()
            (stat_counter, stat_bins) = stat_axis.hist(
                relevant_series, bins = decile_bins)[:2]
            statistics[stat + ' Buckets'][frequency][ticker] = stat_bins
            '_'.join(list(it.combinations(tickers, 2))[0])

#Backtest on ETF pairs based on comparison of their respective buckets of their
#statistics - 
for bucket in ['Volatility Buckets', 'Skewness Buckets', 'Kurtosis Buckets']:
    statistics[bucket + ' Backtest'] = {}
    statistics[bucket + ' Test'] = {}
    for frequency in frequencies:
        statistics[bucket + ' Backtest'][frequency] = {}
        statistics[bucket + ' Test'][frequency] = {}
        for ticker_pair in list(it.combinations(tickers, 2)):
            relevant_full_data = statistics['Differentials'][
                frequency]['_'.join(ticker_pair)]
            relevant_data = relevant_full_data.loc[
                relevant_full_data['Date'] < backtest_period]
            temp_dict = {tick_pair: statistics[bucket][frequency][
                tick_pair].searchsorted(statistics[
                    bucket.replace(' Buckets', '')][frequency].loc[:, 
                    tick_pair]) for tick_pair in ticker_pair}
            statistics[bucket + ' Backtest'][frequency][
                '_'.join(ticker_pair)] = np.prod(relevant_full_data.loc[
                    one_period_ahead:len(relevant_data), 
                    '_minus_'.join(ticker_pair)] * np.sign(temp_dict[
                        ticker_pair[1]] - temp_dict[ticker_pair[0]])[
                            0:len(relevant_data)] + 1) - 1
            statistics[bucket + ' Test'][frequency][
                '_'.join(ticker_pair)] = (np.prod(relevant_full_data.loc[
                    (len(relevant_full_data) - len(relevant_data) + 2):len(
                        relevant_full_data), 
                    '_minus_'.join(ticker_pair)] * np.sign(temp_dict[
                        ticker_pair[1]] - temp_dict[ticker_pair[0]])[
                            (len(relevant_full_data) - len(
                                relevant_data) + 1):(len(
                                relevant_full_data) - 1)] + 1) - 1) * np.sign(
                statistics[bucket + ' Backtest'][frequency]['_'.join(
                    ticker_pair)])

#Overlaying pair strategies of Correlation, Differentials Volatility,
#Differentials Skewness, and Differentials Kurtosis with relativisitic
#comparative bucket performance of each leg of the long-short pair
for strat in ['Correlations Strat', 'Differentials Vol Strat', 
              'Differentials Skew Strat', 'Differentials Kurt Strat']:
    for overlay, bucket in zip(['Vol Overlay', 'Skew Overlay', 'Kurt Overlay'],
                               ['Volatility', 'Skewness', 'Kurtosis']):
        statistics[strat + ' ' + overlay + ' Backtest'] = {}
        statistics[strat + ' ' + overlay + ' Test'] = {}
        for frequency in frequencies:
            statistics[strat + ' ' + overlay + ' Backtest'][frequency] = {}
            statistics[strat + ' ' + overlay + ' Test'][frequency] = {}
            for ticker_pair in list(it.combinations(tickers, 2)):
                statistics[strat + ' ' + overlay + ' Backtest'][frequency][
                    '_'.join(ticker_pair)] = {}
                statistics[strat + ' ' + overlay + ' Test'][frequency][
                    '_'.join(ticker_pair)] = {}
                relevant_full_data = statistics[strat.replace(' Strat', '')][
                    frequency]['_'.join(ticker_pair)]
                relevant_columns = [col for col in 
                    relevant_full_data.columns.values if col not in ['Date']]
                relevant_series = relevant_full_data.loc[:, relevant_columns]
                large_count_deciles = np.array(statistics[strat][frequency][
                    '_'.join(ticker_pair)]['Bin Counts'])
                large_count_deciles = np.where(
                    large_count_deciles >= rel_pred_counts[frequency])[1]
                if large_count_deciles.size:
                    for decile in large_count_deciles:
                        if decile < (decile_bins - 1):
                            relevant_indices = np.where(
                                (relevant_series >= statistics[strat][
                                    frequency]['_'.join(ticker_pair)]['Bins'][
                                        decile]) & 
                                (relevant_series < statistics[strat][
                                    frequency]['_'.join(ticker_pair)]['Bins'][
                                        decile + 1]))[0]
                        else:
                            relevant_indices = np.where(
                                (relevant_series >= statistics[strat][
                                    frequency]['_'.join(ticker_pair)]['Bins'][
                                        decile]) & 
                                (relevant_series <= statistics[strat][
                                    frequency]['_'.join(ticker_pair)]['Bins'][
                                        decile + 1]))[0]
                        backtest_indices = relevant_full_data.loc[
                                relevant_full_data[
                                    'Date'] < backtest_period].index
                        backtest_indices = list(set(
                            backtest_indices).intersection(relevant_indices))
                        test_indices = relevant_full_data.loc[
                                relevant_full_data[
                                    'Date'] > backtest_period].index
                        test_indices = list(set(test_indices).intersection(
                                relevant_indices))
                        if backtest_indices and test_indices:
                            test_indices = [x for x in test_indices if x < max(
                                relevant_full_data.index)]
                            temp_dict[overlay] = {tick_pair: 
                                statistics[bucket + ' Buckets'][frequency][
                                    tick_pair].searchsorted(statistics[
                                        bucket][frequency].loc[:, tick_pair]) 
                                        for tick_pair in ticker_pair}
                            backtest_returns = statistics['Differentials'][
                                frequency]['_'.join(ticker_pair)].loc[[(
                                    x + one_period_ahead) for x in 
                                    backtest_indices], '_minus_'.join(
                                        ticker_pair)]
                            backtest_overlay = np.sign(temp_dict[overlay][
                                ticker_pair[1]][backtest_indices] - \
                                temp_dict[overlay][ticker_pair[0]][
                                    backtest_indices])
                            statistics[strat + ' ' + overlay + ' Backtest'][
                                    frequency]['_'.join(ticker_pair)][
                                        'Decile ' + str(decile + 1)] = np.prod(
                                            np.array(backtest_returns * \
                                            backtest_overlay) + 1) - 1
                            test_returns = statistics['Differentials'][
                                frequency]['_'.join(ticker_pair)].loc[[
                                    (x + one_period_ahead) for x in 
                                    test_indices], '_minus_'.join(ticker_pair)]
                            test_overlay = np.sign(temp_dict[overlay][
                                ticker_pair[1]][test_indices] - \
                                temp_dict[overlay][ticker_pair[0]][
                                    test_indices])
                            backtest_sign = np.sign(statistics[
                                strat + ' ' + overlay + ' Backtest'][
                                    frequency]['_'.join(ticker_pair)][
                                        'Decile ' + str(decile + 1)])
                            statistics[strat + ' ' + overlay + ' Test'][
                                frequency]['_'.join(ticker_pair)][
                                    'Decile ' + str(decile + 1)] = \
                                    np.prod(np.array(test_returns * \
                                                     test_overlay * \
                                                     backtest_sign) + 1) - 1

# This created 24 new keys. To look at each strategy in detail would be very
# time-consuming and so I just summarize how each decile does with each ticker
# pair.

overlay_strategies = list(statistics.keys())[-24:]
backtest_strategy = []
test_strategy = []
[backtest_strategy.append(
    pd.DataFrame(dict(pd.DataFrame(statistics[strat][f_key]).mean()), 
                 index = np.arange(1)))
    if 'Back' in strat
    else test_strategy.append(
    pd.DataFrame(dict(pd.DataFrame(statistics[strat][f_key]).mean()), 
                 index = np.arange(1)))
    for strat in overlay_strategies
    for f_key in frequencies]

backtest_names = [strat for strat in overlay_strategies if 'Back' in strat]
test_names = [strat for strat in overlay_strategies if 'Back' not in strat]
overlay_results = {'Backtest': pd.DataFrame(), 'Test Set': pd.DataFrame()}

for a_strat, a_name, a_df, a_file in zip([backtest_strategy, test_strategy], 
                                         [backtest_names, test_names],
                                         overlay_results.keys(),
                                         ['Backtest Overlay', 'Test Overlay']):
    temp_index = pd.MultiIndex(levels = [a_name, frequencies],
                               labels = label_creater([len(a_name),
                                                       len(frequencies)]))
    temp_df = pd.concat(a_strat, keys = temp_index)
    overlay_results[a_df] = pd.DataFrame(temp_df.values, 
                                         index = temp_df.index.droplevel(2), 
                                         columns = temp_df.columns.values)
    overlay_results[a_df].index.names = ['Strategy', 'Frequency']
    overlay_results[a_df].to_csv(a_file + '.csv')

#Important Result
#overlay_results - keys are Backtest and Test Set at the same level

# Now I examine how many times we traded in the test set and were right
# directionally. This means excluding all NaNs and 0 returns. The NaNs usually
# occur for Yearly frequency due to lack of observations. The 0 returns are
# deciles that are not traded due to the fact that both the long and short ETF
# had the same statistic. Since there are 108 observations in this table, I 
# have to deduct the amount of NaNs and 0s.

# For the whole table
p_value_all = ss.binom_test(len(np.where(overlay_results['Test Set'] > 0)[0]), 
    108 - len(
        np.where((
            overlay_results['Test Set'] == 0) | np.isnan(overlay_results[
                'Test Set']))[0]))

overall_strategy = pd.DataFrame(columns = ['Directional P-Value',
                                           'Magnitude P-Value'], 
                                index = np.arange(0))

for frequency in frequencies:
    indices = [v for i, v in enumerate(
        overlay_results['Test Set'].index.values) if v[1] == frequency]
    rel_subset = overlay_results['Test Set'].loc[indices]
    total_count = np.prod(rel_subset.shape)
    positive_count = len(np.where(rel_subset > 0)[0])
    negative_count = len(np.where(rel_subset < 0)[0])
    missing_count = len(np.where((rel_subset == 0) | np.isnan(rel_subset))[0])
    p_value_direction = ss.binom_test(
        positive_count, total_count - missing_count)
    all_values = list(it.chain.from_iterable(rel_subset.values))
    positive_values = list(filter(lambda x: x > 0, all_values))
    negative_values = list(filter(lambda x: x < 0, all_values))
    degrees_of_freedom = positive_count + negative_count - 2
    pooled_var = ((positive_count - 1) * np.var(positive_values, ddof = 1) + \
                 (negative_count - 1) * np.var(negative_values, ddof = 1)) / \
                 (degrees_of_freedom)
    # since the mean of the negative values are going to be negative and we
    # are only interested in seeing the differences in magnitude, subtracting
    # a negative number, yields the same as adding a positive number of the
    # same magnitude
    t_statistic = (np.mean(positive_values) + np.mean(negative_values)) / \
                  (np.sqrt(pooled_var * (
                      1 / positive_count + 1 / negative_count)))
    p_value_magnitude = ss.t.sf(t_statistic, degrees_of_freedom) * 2
    overall_strategy = overall_strategy.append({
        'Directional P-Value': p_value_direction, 
        'Magnitude P-Value': p_value_magnitude}, ignore_index = True)
    
overall_strategy.index = frequencies

#Important Result
# overall_strategy