from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    smape = 0
    for i in range(len(y_true)):
        smape += (2 * abs(y_true[i] - y_pred[i])) / (abs(y_true[i]) + abs(y_pred[i]))
    smape = (smape * 100) / len(y_true)
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')
    print(f'RMSE: {rmse}')
    print(f'SMAPE: {smape}')

def plot_actual_forecast(y_true, y_pred):
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Forecast')
    plt.legend()
    plt.show()

def plot_future_forecast(y_true, y_pred):
    y_true = y_true[-len(y_pred):]
    plt.plot(y_true, label='Actual', style='-')
    plt.plot(y_pred, label='Forecast', style='--')
    plt.legend()
    plt.show()
