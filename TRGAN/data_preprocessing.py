import pandas as pd
import numpy as np
import copy
from typing import Union,TypeVar
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder



def load_data(path: str, name: str) -> pd.DataFrame:
    data = pd.read_csv(f'{path}{name}.csv')
    
    if name == 'czech':
        czech_date_parser = lambda x: datetime.strptime(str(x), "%y%m%d")
        data["datetime"] = data["date"].apply(czech_date_parser)
        
        data = data.rename(columns={"account_id": 'customer', 'tcode': 'mcc', 'datetime': 'transaction_date'})
        data = data.drop(['date', 'Unnamed: 0', 'type', 'operation', 'k_symbol', 'column_a'], axis=1)

        idx_customer = (data['customer'].value_counts().loc[(data['customer'].value_counts() > 20) == True]).index.tolist()
        data = data[data['customer'].isin(idx_customer)]
        
        
        data['transaction_date'] = pd.to_datetime(data['transaction_date'])
        data = data.sort_values(by='transaction_date')
        data = data.reset_index(drop=True)
    
    
        le = LabelEncoder()
        data['mcc'] = le.fit_transform(data['mcc'])
        data = data[data['customer'].isin(data['customer'].value_counts().index[np.where(data['customer'].value_counts() > 1)[0]])]
        
    
    return data