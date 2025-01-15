from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def time_plot(df, xlabel, ylabel, title):
    df.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def create_time_features(df, target='y', split=True, label=True):

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['day_of_year'] = df.index.dayofyear
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week

    if split:
        X = df.drop(target, axis=1)
        if label:
            y = df[target]
            return X, y
        return X

    df['day_str'] = [i.strftime('%A') for i in df.index]
    df['year_month'] = [i.strftime('%Y-%m') for i in df.index]
    return df

def seasonal_plot(df, target, ylabel, title, has_features=False):
    
    if not has_features:
        df = create_time_features(df, target=target, split=False)

    # 2x2 grid plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    # Assume that the dataset is hourly
    # 1. Hourly
    axes[0].plot(df.resample('H').mean()[target])
    axes[0].set_title('Hourly')
    axes[0].set_xlabel('Datetime')
    axes[0].set_ylabel(ylabel)

    # 2. Yearly
    yearly_data = df[['year', 'month', target]].groupby(['year', 'month']).mean()[[target]].reset_index()
    years = yearly_data['year'].unique()

    for year in years:
        axes[1].plot(yearly_data[yearly_data['year'] == year], label=year)
        axes[1].text(
            yearly_data.loc[yearly_data['year'] == year].shape[0] + 0.3,
            yearly_data.loc[yearly_data['year'] == year, target][-1:].values[0],
            str(year),
            fontsize=12
        )

    axes[1].set_title('Yearly')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel(ylabel)
    # axes[1].legend()

    # 3. Weekly
    weekly_data = df[['month', 'day_str', 'day_of_week', target]].dropna().groupby(['month', 'day_str', 'day_of_week']).mean()[[target]].reset_index()
    weekly_data = weekly_data.sort_values('day_of_week')
    months = weekly_data['month'].unique()

    for month in months:
        axes[2].plot(weekly_data[weekly_data['month'] == month][target], label=month)
        axes[2].text(
            weekly_data.loc[weekly_data['month'] == month].shape[0] - 0.9,
            weekly_data.loc[weekly_data['month'] == month, target][-1:].values[0],
            str(month),
            fontsize=12
        )

    axes[2].set_title('Weekly')
    axes[2].set_xlabel('Day of Week')
    axes[2].set_ylabel(ylabel)
    # axes[2].legend()

    # 4. Daily
    daily_data = df[['hour', 'day_str', target]].groupby(['hour', 'day_str']).mean()[[target]].reset_index()

    sns.lineplot(data=daily_data, x='hour', y=target, hue='day_str', ax=axes[3])
    axes[3].set_title('Daily')
    axes[3].set_xlabel('Hour')
    axes[3].set_ylabel(ylabel)
    # axes[3].legend()

    plt.suptitle(title)
    plt.show()

def box_plot(df, target, target_name, title, has_features=False):

    if not has_features:
        df = create_time_features(df, target=target, split=False)

    # 2x2 grid plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    # Assume that the dataset is hourly
    # 1. Target distribution
    sns.boxplot(data=df, x=target, ax=axes[0])
    axes[0].set_title(f'{target_name} Distribution')
    axes[0].set_xlabel(target_name)

    # 2. Year and month distribution
    ym_data = df[df['year'] >= 2017].reset_index().sort_values(by='Datetime').set_index('Datetime')

    sns.boxplot(data=ym_data, x='year_month', y=target, hue='year_month', ax=axes[1])
    axes[1].set_title('Year and Month Distribution')
    axes[1].set_xlabel('Year and Month')
    axes[1].set_ylabel(target_name)
    axes[1].xticks(rotation=45)

    # 3. Day of week distribution
    dow_data = df[['day_str', 'day_of_week', target]].sort_values(by='day_of_week')

    sns.boxplot(data=dow_data, x='day_str', y=target, hue='day_str', ax=axes[2])
    axes[2].set_title('Day of Week Distribution')
    axes[2].set_xlabel('Day of Week')
    axes[2].set_ylabel(target_name)

    # 4. Hour distribution
    hour_data = df[['hour', target]].sort_values(by='hour')

    sns.boxplot(data=hour_data, x='hour', y=target, hue='hour', ax=axes[3], palette='viridis')
    axes[3].set_title('Hour Distribution')
    axes[3].set_xlabel('Hour')
    axes[3].set_ylabel(target_name)

    plt.suptitle(title)
    plt.show()

def seasonal_decompose_plot(df, target, ym='2017-01'):

    df_plot = df.copy()
    if ym:
        y, m = ym.split('-')
        y, m = int(y), int(m)
        df_plot = df_plot[(df_plot.index.year == y) & (df_plot.index.month == m)]
        df_plot = df_plot.set_index('Datetime')

    sd_add = seasonal_decompose(df_plot[target], model='additive', period=24 * 7)
    sd_mul = seasonal_decompose(df_plot[target], model='multiplicative', period=24 * 7)

    plt.figure(figsize=(20, 10))
    sd_add.plot().suptitle('Additive Decomposition', fontsize=22)
    plt.xticks(rotation=45)
    sd_mul.plot().suptitle('Multiplicative Decomposition', fontsize=22)
    plt.xticks(rotation=45)
    plt.show()

def pacf_plot(y, hour_range, lags=30, alpha=0.01):

    actual = y.copy()
    
    for hour in hour_range:
        plot_pacf(actual[actual.index.hour == hour], lags=lags, alpha=alpha, title=f'PACF for Hour {hour}')
        plt.xlabel('Lags')
        plt.ylabel('Partial Autocorrelation')
        plt.show()

def forecast_plot(data, pred, xlabel, ylabel, title):
    plt.plot(data, label='Actual')
    plt.plot(pred, label='Forecast')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()