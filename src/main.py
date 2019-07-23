
import gc
import pandas as pd
import numpy as np 
from datetime import datetime as dt

from sklearn.preprocessing import LabelEncoder

def _get_merged_data(t_data_path, i_data_path):
    """
    """
    t_data = pd.read_csv(t_data_path)
    i_data = pd.read_csv(i_data_path)

    t_data = t_data.merge(right=i_data, how='outer', right_on='TransactionID', left_on='TransactionID')

    del i_data
    gc.collect()

    return t_data

def _get_column_types(data):
    """
    """
    # Build categorical columns
    cat_cols = set(['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo'])
    cat_cols.update(['card' + str(i) for i in range(1, 7)])
    cat_cols.update(['M' + str(i) for i in range(1, 10)])
    cat_cols.update(['id_' + str(i) for i in range(12, 39)])

    # Build numerical columns
    num_cols = set([x for x in data.columns.values if x not in cat_cols])

    # remove id and predictor (handle date/time values separately)
    num_cols.remove('TransactionID')
    num_cols.remove('isFraud')

    return cat_cols, num_cols

def _dropna_columns(data, columns_for_test=None):
    """
    Let's keep it simple for the time being, drop all columns that have
    a 80%+ NaN value.
    """
    if columns_for_test: # for processing test data
        return data.drop(columns_for_test, axis=1)

    before_columns = data.columns.values
    data = data.dropna(thresh=(0.8 * len(data)), axis=1)
    dropped_columns = set([x for x in data.columns if x not in before_columns])
    return data, dropped_columns

def _fillna_values(data):
    """
    Keep it simple, fill with the most frequent.
    """
    return data.fillna(data.mode().iloc[0])

def _encode_cyclical(data, column, max_value):
    data[column + '_sin'] = np.sin(2 * np.pi * data[column] / max_value)
    data[column + '_cos'] = np.cos(2 * np.pi * data[column / max_value])
    return data

def _one_hot_encode(data, columns):
    for col in columns:
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
    return data

def _normalize(data, columns):
    for col in columns:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data

def _label_encode(train_data, test_data, cat_cols):
    for col in cat_cols:
        if col in train_data.columns:
            le = LabelEncoder()
            le.fit(list(train_data[col].astype(str).values) + list(test[col].astype(str).values))
            train_data[col] = le.transform(list(train_data[col].astype(str).values))
            test_data[col] = le.transform(list(test_data[col].astype(str).values))

    return train_data, test_data

if __name__ == '__main__':
    start_overall = dt.now()
    print('[{0}] Starting IEEE Fraud Detection Solution...'.format(dt.now()))

    print('[{0}] Reading in training data for processing...'.format(dt.now()))
    data = _get_merged_data('../data/train_transaction.csv', '../data/train_identity.csv')
    print('[{0}] Data read in and merged together...'.format(dt.now()))

    values = data['isFraud'].value_counts().keys().tolist()
    counts = data['isFraud'].value_counts(normalize=True).tolist()
    print('[{0}] Distribution of predictor features (isFraud): {1}: {2}%, {3}: {4}%'.format(
        dt.now(),
        'Not Fraud',
        round(counts[0] * 100, 2),
        'Fraud',
        round(counts[1] * 100, 2)
    ))

    cat_cols, num_cols = _get_column_types(data)
    assert (len(cat_cols) + len(num_cols) == len(data.columns.values) - 2)

    print('[{0}] Handling NaN values in the training data...'.format(dt.now()))
    data, dropped_columns = _dropna_columns(data)
    data = _fillna_values(data)



    # print('[{0}] Reading in testing data for processing...'.format(dt.now()))
    # test_data = _get_merged_data('../data/test_transaction.csv', '../data/test_transaction.csv')
    # print('[{0}] Test data read in and merged together...'.format(dt.now()))

    # values = test_data['isFraud'].value_counts().keys().tolist()
    # counts = test_data['isFraud'].value_counts(normalize=True).tolist()
    # print('[{0}] Distribution of predictor features (isFraud): {1}: {2}%, {3}: {4}%'.format(
    #     dt.now(),
    #     'Not Fraud',
    #     round(counts[0] * 100, 2),
    #     'Fraud',
    #     round(counts[1] * 100, 2)
    # ))

    # t_cat_cols, t_num_cols = _get_column_types(test_data)
    # assert (len(t_cat_cols) + len(t_num_cols) == len(test_data.columns.values) - 2)
    # assert (len(data.columns.values) == len(test_data.columns.values))

    # print('[{0}] Handling NaN values in the testing data...'.format(dt.now()))
    # test_data = _dropna_columns(test_data, dropped_columns)
    # test_data = _fillna_values(test_data)



    
