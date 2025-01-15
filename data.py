import os
import pandas as pd
import subprocess
from torch.utils.data import DataLoader, TensorDataset
from typing import Literal
import zipfile

import torch
from eda import create_time_features

def download_data():

    if os.path.exists('data'):
        print('Data already downloaded')
        return

    os.makedirs('data')
    subprocess.run(['curl', '-L', '-o', 'hourly-energy-consumption.zip', 'https://www.kaggle.com/api/v1/datasets/download/robikscube/hourly-energy-consumption'])

    with zipfile.ZipFile('hourly-energy-consumption.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

    os.remove('hourly-energy-consumption.zip')

    print('Data downloaded successfully!')

def prepare_data(
        name='PJME_hourly.csv',
        scaler=None, 
        split_date='2015-01-01', 
        method: Literal['none', 'feature', 'shift']='none',
        window=30
    ):

    df = pd.read_csv(f'data/hourly-energy-consumption/{name}', index_col=[0], parse_dates=[0])
    df.sort_index(inplace=True)

    if scaler:
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    if split_date:
        train = df[df.index <= split_date].copy()
        test = df[df.index > split_date].copy()
        
        if method == 'none':
            return train, test
        
        elif method == 'feature':
            X_train, y_train = create_time_features(train)
            X_test, y_test = create_time_features(test)

            return X_train, y_train, X_test, y_test
        
        elif method == 'shift':
            train_shift = train.copy()
            test_shift = test.copy()

            target = df.columns[0]

            for i in range(1, window+1):
                train_shift[f'lag_{i}'] = train_shift[target].shift(i)
                test_shift[f'lag_{i}'] = test_shift[target].shift(i)

            train_shift.dropna(inplace=True)
            test_shift.dropna(inplace=True)

            X_train = train_shift.drop(target, axis=1).values
            y_train = train_shift[target].values
            X_test = test_shift.drop(target, axis=1).values
            y_test = test_shift[target].values

            return X_train, y_train, X_test, y_test
    
    return df

def prepare_loader(X_train, y_train, X_test, y_test, batch_size=32, device='cpu'):

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return {
        'train': train_loader,
        'test': test_loader
    }
