import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

from data import download_data, prepare_data, prepare_loader
import dl.model as dl_model
import dl.train as dl_train
from eda import create_time_features
from eval import plot_actual_forecast, plot_future_forecast
import ml.model as ml_model
import stats.model as stat_model
import stats.transform as T

# hyperparameters
xgb_n_estimators = 1000
xgb_max_depth = 6
xgb_learning_rate = 0.3
xgb_objective = 'reg:squarederror'
xgb_early_stopping_rounds = 50

nn_batch_size = 32
nn_num_epochs = 20
nn_early_stop_patience = 5
nn_early_stop_delta = 0.00001
nn_learning_rate = 0.001
nn_window = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TimeSeriesScaler:

    def __init__(self):
        self.ts = None

    def fit_transform(self, X):
        self.ts = X
        return T.difference(T.log_transform(X))
    
    def inverse_transform(self, X):
        return T.inverse_log_transform(T.inverse_difference(X, self.ts))

def main():
    
    parser = argparse.ArgumentParser(description='Model training and testing')
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, required=True, choices=['arima', 'sarima', 'xgb', 'nn'], help='Model to train')

    test_parser = subparsers.add_parser('test', help='Test a model')
    test_parser.add_argument('--model', type=str, required=True, choices=['arima', 'sarima', 'xgb', 'nn'], help='Model to test')

    forecast_parser = subparsers.add_parser('forecast', help='Forecast using a model')
    forecast_parser.add_argument('--model', type=str, required=True, choices=['arima', 'sarima', 'xgb', 'nn'], help='Model to forecast')
    forecast_parser.add_argument('--steps', type=int, default=24, help='Number of steps to forecast')

    args = parser.parse_args()

    if args.command == 'train':
        download_data()
        
        if args.model == 'arima':
            scaler = TimeSeriesScaler()
            train, test = prepare_data(scaler=scaler, method='none')
            order, seasonal_order = stat_model.arima_params(train)
            model = stat_model.arima_fit(train, order, seasonal_order)
            stat_model.arima_save(model, 'models/arima.pkl')

        elif args.model == 'sarima':
            scaler = TimeSeriesScaler()
            train, test = prepare_data(scaler=scaler, method='none')
            order, seasonal_order = stat_model.arima_params(train)
            model = stat_model.sarima_fit(train, order, seasonal_order)
            stat_model.arima_save(model, 'models/sarima.pkl')
        
        elif args.model == 'xgb':
            X_train, y_train, X_test, y_test = prepare_data(method='feature')
            params = {
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'learning_rate': xgb_learning_rate,
                'objective': xgb_objective,
                'early_stopping_rounds': xgb_early_stopping_rounds
            }
            model = ml_model.xgb_fit(X_train, y_train, X_test, y_test, params)
            ml_model.xgb_save(model, 'models/xgb.pkl')
        
        elif args.model == 'nn':
            scaler = MinMaxScaler()
            X_train, y_train, X_test, y_test = prepare_data(scaler=scaler, method='shift', window=nn_window)
            loaders = prepare_loader(X_train, y_train, X_test, y_test, batch_size=32, device=device)

            model = dl_model.CNN_LSTM().to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=nn_learning_rate)
            model = dl_train.train_model(
                loaders,
                model,
                criterion,
                optimizer,
                device,
                num_epochs=nn_num_epochs,
                early_stop_patience=nn_early_stop_patience,
                early_stop_delta=nn_early_stop_delta
            )
            dl_train.save_model(model, 'models/nn.pth')

    elif args.command == 'test':

        if args.model == 'arima':
            scaler = TimeSeriesScaler()
            train, test = prepare_data(scaler=scaler, method='none')
            model = stat_model.arima_load('models/arima.pkl')
            forecast = stat_model.arima_forecast(model, len(test))
            forecast = scaler.inverse_transform(forecast)
            plot_actual_forecast(test, forecast)

        elif args.model == 'sarima':
            scaler = TimeSeriesScaler()
            train, test = prepare_data(scaler=scaler, method='none')
            model = stat_model.arima_load('models/sarima.pkl')
            forecast = stat_model.arima_forecast(model, len(test))
            forecast = scaler.inverse_transform(forecast)
            plot_actual_forecast(test, forecast)

        elif args.model == 'xgb':
            X_train, y_train, X_test, y_test = prepare_data(method='feature')
            model = ml_model.xgb_load('models/xgb.pkl')
            forecast = ml_model.xgb_predict(model, X_test)
            plot_actual_forecast(y_test, forecast)

        elif args.model == 'nn':
            scaler = MinMaxScaler()
            X_train, y_train, X_test, y_test = prepare_data(scaler=scaler, method='shift', window=30)
            loaders = prepare_loader(X_train, y_train, X_test, y_test, batch_size=32, device=device)

            model = dl_model.CNN_LSTM().to(device)
            model = dl_train.load_model(model, 'models/nn.pth')
            model.eval()
            forecast = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().detach().numpy().flatten()
            forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
            plot_actual_forecast(y_test, forecast)

    elif args.command == 'forecast':

        if args.model == 'arima':
            scaler = TimeSeriesScaler()
            train, test = prepare_data(scaler=scaler, method='none')
            model = stat_model.arima_load('models/arima.pkl')
            forecast = stat_model.arima_forecast(model, args.steps)
            forecast = scaler.inverse_transform(forecast)
            plot_future_forecast(train, forecast)

        elif args.model == 'sarima':
            scaler = TimeSeriesScaler()
            train, test = prepare_data(scaler=scaler, method='none')
            model = stat_model.arima_load('models/sarima.pkl')
            forecast = stat_model.arima_forecast(model, args.steps)
            forecast = scaler.inverse_transform(forecast)
            plot_future_forecast(train, forecast)

        elif args.model == 'xgb':
            X_train, y_train, X_test, y_test = prepare_data(method='feature')
            model = ml_model.xgb_load('models/xgb.pkl')
            
            new_index = pd.date_range(X_test.index[-1], periods=args.steps+1, freq='H')[1:]
            X_future = create_time_features(pd.DataFrame(index=new_index), split=False, label=False)
            y_future = ml_model.xgb_predict(model, X_future)
            y_test_sample = y_test.iloc[-args.steps:]
            plot_future_forecast(y_test_sample, y_future)

        elif args.model == 'nn':
            scaler = MinMaxScaler()
            X_train, y_train, X_test, y_test = prepare_data(scaler=scaler, method='shift', window=30)
            loaders = prepare_loader(X_train, y_train, X_test, y_test, batch_size=32, device=device)

            model = dl_model.CNN_LSTM().to(device)
            model = dl_train.load_model(model, 'models/nn.pth')

            X_future = X_test[-nn_window:].values
            X_future_tensor = torch.tensor(X_future, dtype=torch.float32).to(device)
            y_future = []

            model.eval()
            with torch.no_grad():
                for i in range(args.steps):
                    y_pred = model(X_future_tensor.reshape(1, -1))
                    y_future.append(y_pred.item())
                    X_future = np.roll(X_future, -1)
                    X_future[-1] = y_pred.item()
                    X_future_tensor = torch.tensor(X_future, dtype=torch.float32).to(device)

            y_future = scaler.inverse_transform(np.array(y_future).reshape(-1, 1)).ravel()
            y_test_sample = y_test.iloc[-args.steps:]
            y_test_sample = scaler.inverse_transform(y_test_sample.values.reshape(-1, 1)).ravel()
            plot_future_forecast(y_test_sample, y_future)
