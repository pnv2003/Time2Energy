import joblib
from xgboost import XGBRegressor, plot_importance

def xgb_fit(X_train, y_train, X_test, y_test, params):

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    plot_importance(model, height=0.9)

    return model

def xgb_predict(model, X):

    return model.predict(X)

def xgb_save(model, path):

    joblib.dump(model, path)