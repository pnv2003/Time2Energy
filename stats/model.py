from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMA

def arima_params(data, m=12):

    model = auto_arima(data, m=m, trace=True, error_action='ignore')
    return model.order, model.seasonal_order

def arima_fit(data, order):

    model = ARIMA(data, order=order).fit()
    print(model.summary())
    return model

def sarima_fit(data, order, seasonal_order):

    model = SARIMA(data, order=order, seasonal_order=seasonal_order).fit()
    print(model.summary())
    return model

def arima_forecast(model, steps):

    return model.forecast(steps)[0]

def arima_save(model, path):

    model.save(path)

def arima_load(path):
    
    return ARIMA.load(path)
