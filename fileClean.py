import configparser
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read('config.ini')
config = config['default']
filepath_raw = config['dataset_filepath']
dataset_clean_file = config['dataset_clean_file']
correlation_target_file = config['correlation_target_file']


# dropping columns that are 99% or more similar
def filter_feature_file():
    dataset_file = pd.read_csv(filepath_or_buffer=filepath_raw, delimiter=',')
    dataset_file = dataset_file.replace('?', np.NaN)

    selector = VarianceThreshold(threshold=0.01)
    selector.fit(dataset_file)

    feature_drop = [column for column in dataset_file.columns
                    if column not in dataset_file.columns[selector.get_support()]]

    dataset_file.drop(feature_drop, axis=1, inplace=True)
    dataset_file.to_csv(dataset_clean_file, index=False, sep=',')

    print('Number of columns with 99% or more similar values: ' + str(len(feature_drop)))
    return


# replace all fields where values are Nan with mean of that column
def mean_result():
    num_mean_column = 0
    dataset_file = pd.read_csv(filepath_or_buffer=dataset_clean_file, delimiter=',')

    for column in dataset_file:
        if dataset_file[column].isnull().any():
            mean_column = np.nanmean(dataset_file[column].values)
            dataset_file[column].replace(np.NaN, round(mean_column, 6), inplace=True)
            num_mean_column += 1
    dataset_file.to_csv(dataset_clean_file, index=False, sep=',')
    print('Number of columns where NaN is replaced: ' + str(num_mean_column))
    return


# calculate correlation between all columns
def correlation_target():
    dataset_file = pd.read_csv(filepath_or_buffer=dataset_clean_file, delimiter=',')

    calculate_correlation = dataset_file.corr()
    calculate_correlation.to_csv(correlation_target_file, float_format='%.6f', index=False, sep=',')
    return


# scaling all dataset
def data_standardization():
    dataset_file = pd.read_csv(filepath_or_buffer=dataset_clean_file, delimiter=',')

    for column in dataset_file.loc[:, ~dataset_file.columns.isin(['result'])]:
        all_values_unscale = dataset_file[column].values
        scalar = StandardScaler()
        scaled_data = scalar.fit_transform(all_values_unscale.reshape(-1, 1))
        dataset_file[column].replace(all_values_unscale, scaled_data, inplace=True)
    dataset_file.to_csv(dataset_clean_file, float_format='%.6f', index=False, sep=',')
    return
