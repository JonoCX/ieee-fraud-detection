
import gc, calendar
import pandas as pd
import numpy as np 
from datetime import datetime as dt

from sklearn.preprocessing import LabelEncoder

class Preprocessing:

    def get_merged_data(self, t_data_path, i_data_path):
        """
        """
        t_data = pd.read_csv(t_data_path)
        i_data = pd.read_csv(i_data_path)

        t_data = t_data.merge(right=i_data, how='outer', right_on='TransactionID', left_on='TransactionID')

        del i_data
        gc.collect()

        return t_data

    def get_column_types(self, data):
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

    def dropna_columns(self, data, columns_for_test=None):
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

    def fillna_values(self, data):
        """
        Keep it simple, fill with the most frequent.
        """
        return data.fillna(data.mode().iloc[0])

    def handle_timedelta(self, data, epochal_date):
        data['TransactionDT'] = pd.to_timedelta(data['TransactionDT'], unit='seconds')
        data['timestamp'] = data['TransactionDT'].apply(lambda x: epochal_date + x)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data

class FeatureEngineering:

    def _encode_cyclical(self, data, columns):
        for col, max_value in columns.items():
            data[col + '_sin'] = np.sin(2 * np.pi * data[col] / max_value)
            data[col + '_cos'] = np.cos(2 * np.pi * data[col] / max_value)
        return data

    def _normalize(self, data, columns):
        for col in columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        return data

    def _label_encode(self, data, cat_cols):
        for col in cat_cols:
            if col in train_data.columns:
                le = LabelEncoder()
                le.fit(list(data[data['source'] == 'train'][col].astype(str).values) + list(test_data[data['source'] == 'test'][col].astype(str).values))
                train_data[col] = le.transform(list(data[data['source'] == 'train'][col].astype(str).values))
                test_data[col] = le.transform(list(data[data['source'] == 'test'][col].astype(str).values))

        return data

    def _one_hot_encode(self, data, columns):
        for col in columns:
            data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)
        return data

    def _time_of_day(self, hour):
        if hour in range(5, 13): return 'morning'
        elif hour in range(13, 18): return 'afternoon'
        elif hour in range(18, 23): return 'evening'
        else: return 'night'

    def handle_categorical(self, data, cat_cols):
        # label encode, one hot encode
        data = self._label_encode(data, cat_cols)
        data = self._one_hot_encode(data, cat_cols)
        return data

    def handle_numerical(self, data, num_cols):
        # normalize columns
        return self._normalize(data, num_cols)

    def handle_cyclical(self, data, cyc_cols):
        # encode columns
        return self._encode_cyclical(data, cyc_cols)

    def create_time_features(self, data):
        data['time_hour'] = data['timestamp'].dt.hour 
        data['time_minute'] = data['timestamp'].dt.minute
        data['time_day'] = data['timestamp'].dt.day
        data['time_month'] = data['timestamp'].dt.month
        
        data['inside_business_hours'] = data['timestamp'].map(
            lambda x: (x.hour < 9 and x.hour > 17) and (calendar.day_name[x.weekday()] and calendar.day_name[x.weekday()])
        )

        data['time_of_day'] = data['time_hour'].map(self._time_of_day)

        weekend_days = set(['Saturday', 'Sunday'])
        data['weekday'] = data['timestamp'].map(lambda x: calendar.day_name[x.weekday()] in weekend_days)

        return data

if __name__ == '__main__':
    start_overall = dt.now()
    print('[{0}] Starting IEEE Fraud Detection Solution...'.format(start_overall))

    preprocessing = Preprocessing()
    feature_engineering = FeatureEngineering()

    print('[{0}] Reading in training data for processing...'.format(dt.now()))
    train_data = preprocessing.get_merged_data('../data/train_transaction.csv', '../data/train_identity.csv')
    print('[{0}] Data read in and merged together...'.format(dt.now()))

    print('[{0}] Reading in testing data for processing...'.format(dt.now()))
    test_data = preprocessing.get_merged_data('../data/test_transaction.csv', '../data/test_identity.csv')
    print('[{0}] Data read in and merged together...'.format(dt.now()))

    train_data['source'] = 'train'
    test_data['source'] = 'test'

    print('[{0}] Concatenating training and testing data to process together...'.format(dt.now()))
    data = pd.concat([train_data, test_data], ignore_index=True, sort=False)

    del train_data
    del test_data
    gc.collect()

    values = data[data['source'] == 'train']['isFraud'].value_counts().keys().tolist()
    counts = data[data['source'] == 'train']['isFraud'].value_counts(normalize=True).tolist()
    print('[{0}] Distribution of predictor features (isFraud): {1}: {2}%, {3}: {4}%'.format(
        dt.now(),
        'Not Fraud',
        round(counts[0] * 100, 2),
        'Fraud',
        round(counts[1] * 100, 2)
    ))

    cat_cols, num_cols = preprocessing.get_column_types(data)
    assert (len(cat_cols) + len(num_cols) == len(data.columns.values) - 2)

    print('[{0}] Handling NaN values in the training data...'.format(dt.now()))
    data, dropped_columns = preprocessing.dropna_columns(data)
    data = preprocessing.fillna_values(data)
    print('[{0}] NaN values filled...'.format(dt.now()))
    
    epochal_date = dt(1970, 1, 1)
    print('[{0}] Processing timedelta feature, using {1} as epochal date...'.format(dt.now(), epochal_date))
    data = preprocessing.handle_timedelta(data, epochal_date)

    print('[{0}] Creating time-based features from the timestamp...'.format(dt.now()))
    data = feature_engineering.create_time_features(data)

    cyclical_columns = {
        'time_hour': 23,
        'time_minute': 59,
        'time_day': 6,
        'time_month': 12
    }

    cat_cols.update([
        'inside_business_hours',
        'time_of_day',
        'weekday'
    ])

    num_cols.update([
        'time_hour_cos', 'time_hour_sin', 'time_minute_cos', 
        'time_minute_sin', 'time_day_cos', 'time_day_sin',
        'time_month_sin', 'time_month_cos'
        ]
    )
    num_cols.remove('V207')
    num_cols.remove('V250')

    # Handle cyclical data
    cyc_ts = dt.now()
    print('[{0}] Encoding cyclical data...'.format(cyc_ts))
    data = feature_engineering.handle_cyclical(data, cyclical_columns)
    cyc_ts_end = dt.now()
    print('[{0}] Encoded cyclical data (total time: {1})...'.format(cyc_ts_end, cyc_ts_end - cyc_ts))

    # Handle numerical data
    num_ts = dt.now()
    print('[{0}] Encoding numerical data...'.format(num_ts))
    data = feature_engineering.handle_numerical(data, num_cols)
    num_ts_end = dt.now()
    print('[{0}] Encoded numerical data (total time: {1})...'.format(num_ts_end, num_ts_end - num_ts))

    # Handle categorical data
    cat_ts = dt.now()
    print('[{0}] Encoding categorical data...'.format(cat_ts))
    data = feature_engineering.handle_categorical(data, cat_cols)
    cat_ts_end = dt.now()
    print('[{0}] Encoded categorical data (total time: {1})...'.format(cat_ts_end, cat_ts_end - cat_ts))

    print(data.columns.values)
    
