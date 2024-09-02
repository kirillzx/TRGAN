import pandas as pd
import numpy as np
import copy
import json
from typing import Union,TypeVar
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import scipy.stats as sts



def load_data(path: str, name: str) -> pd.DataFrame:
    if name == 'czech':
        data = pd.read_csv(f'{path}{name}.csv')
        
        czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
        data["datetime"] = data["date"].apply(czech_date_parser)
        
        data = data.rename(columns={"account_id": 'customer', 'tcode': 'mcc', 'datetime': 'transaction_date'})
        data = data.drop(['date', 'Unnamed: 0', 'type', 'operation', 'k_symbol', 'column_a'], axis=1)

        # idx_customer = (data['customer'].value_counts().loc[(data['customer'].value_counts() > 2) == True]).index.tolist()
        # data = data[data['customer'].isin(idx_customer)]
        
        le = LabelEncoder()
        data['mcc'] = le.fit_transform(data['mcc'])
        
        data = data.sample(100_000)
        data = data.reset_index(drop=True)
        data = data[data['amount'] < 30_000]
        # data['amount'] = np.log1p(data['amount'])
        
        data = data[data['customer'].isin(data['customer'].value_counts().index[np.where(data['customer'].value_counts() > 2)[0]])]
        
        data['transaction_date'] = pd.to_datetime(data['transaction_date'])
        data = data.sort_values(by='transaction_date')
        
        
    elif name == 'data_uk_clean':
        data = pd.read_csv('Data/data_uk_clean.csv')[['account_id', 'amount', 'balance', 'description', 'date', 'id', 'type', 'tcode']]
        # data['amount'] = data['amount'] + np.abs(data['amount'].min()) + 1
        data['date'] = pd.to_datetime(data['date'], format='mixed')
        data['date'] = pd.to_datetime(data['date'].dt.date)
        data = data[data['account_id'].isin(data['account_id'].value_counts().index[np.where(data['account_id'].value_counts() > 2)[0]])]
        data = data.sort_values(by='date')
        data = data.reset_index(drop=True)
        
    elif name == 'transaction_secret':
        data = pd.read_csv(f'{path}{name}.csv')
        
        data = data.drop(['CHANNEL', 'USED_PAY_SERVICE', 'RETAILER', 'TRAN_METHOD', 'CurrencyName', 'DEVICE_TYPE'], axis=1)
        data = data.dropna().reset_index(drop=True)
        
        data['MCC'] = data['MCC'].astype(int).astype(str)
        data['ACCOUNT_ID'] = data['ACCOUNT_ID'].astype(int).astype(str)
        data['CARD'] = data['CARD'].astype(int).astype(str)
        data['CustomerKey'] = data['CustomerKey'].astype(int).astype(str)
        data['ID'] = data['ID'].astype(int).astype(str)
        data = data.rename(columns={'IS_OWN_TERMINAL': 'ISOWNTERMINAL'})
        
        
        data = data.loc[remove_outliers(data[['AMOUNT_EQ', 'PAY_AMT']]).index]
        
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['TRANS_TIME'] = pd.to_datetime(data['TRANS_TIME'])
        
        data = data.sample(100_000)
        data = data[data['CustomerKey'].isin(data['CustomerKey'].value_counts().index[np.where(data['CustomerKey'].value_counts() > 2)[0]])]
        
        data = data.sort_values(['DATE', 'TRANS_TIME'], ascending=[True, True])
        data['HOUR'] = data['TRANS_TIME'].dt.hour
        data['MINUTE'] = data['TRANS_TIME'].dt.minute
        data['SECOND'] = data['TRANS_TIME'].dt.second
        # data['AMOUNT_EQ'] = np.log1p(data['AMOUNT_EQ'])
        
        data = data.reset_index(drop=True)
        
    elif name == 'users_spb_only':
        with open('Data/users_spb_only.json') as f:
            d = json.load(f)

        d = pd.json_normalize(d).T
        d = d.sample(13_000)

        data = pd.DataFrame()
        for i in d.index:
            temp = pd.json_normalize(d.loc[i]).T[0].apply(pd.Series)
            temp['customer'] = [i] * len(temp)
            data = pd.concat([data, temp])
            
        data = data.sample(100_000)
        data = data.reset_index(drop=True)
        data = data.loc[remove_outliers(data[['AMOUNT_EQ']]).index]
        
        data = data[data['customer'].isin(data['customer'].value_counts().index[np.where(data['customer'].value_counts() > 2)[0]])]
        
        data['TRANS_DATE'] = pd.to_datetime(data['TRANS_DATE'])
        data['TRANS_TIME'] = pd.to_datetime(data['TRANS_TIME'])
        data = data.sort_values(['TRANS_DATE', 'TRANS_TIME'], ascending=[True, True])
        data['HOUR'] = data['TRANS_TIME'].dt.hour
        data['MINUTE'] = data['TRANS_TIME'].dt.minute
        data['SECOND'] = data['TRANS_TIME'].dt.second
        
    
    return data



def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    data = copy.deepcopy(data)
    
    for col in data.columns:
        data = data[(data[col] < np.quantile(data[col], 0.75) + 3 * sts.iqr(data[col])) &
                    (data[col] > np.quantile(data[col], 0.25) - 3 * sts.iqr(data[col]))]
        
    return data