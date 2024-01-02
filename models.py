from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

import xgboost as xgb

from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from neuralprophet import NeuralProphet

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import plotly
import json

def plot_model(df,forecast):

  # Create a trace for the time series
  trace = go.Scatter(
      x = df.index, # The dates along the x-axis
      y = df['point_value'], # The values along the y-axis
      mode = 'lines', # Connect the data points with lines
      name = 'Time Series' # Name of the trace for the legend
  )

  # Perform time series forecasting using your chosen method and create trace for the forecasted line
  # For this example, let's assume we have a forecasted line stored in a separate DataFrame called `forecast`
  forecast_trace = go.Scatter(
      x = forecast.index  , # The dates along the x-axis
      y = forecast.values, # The forecasted values along the y-axis
      mode = 'lines', # Connect the data points with lines
      name = 'Forecast' # Name of the trace for the legend
  )

  # Combine the time series trace and forecast trace into one data object
  data = [trace, forecast_trace]

  # Create the layout for the plot
  layout = go.Layout(
      title = 'Time Series Plot with Forecast', # Title of the plot
      xaxis = dict(title = 'Date'), # Label for the x-axis
      yaxis = dict(title = 'Value'), # Label for the y-axis
      hovermode = 'x unified' # Show data for all traces at a given x value
  )

  # Create the figure and plot it
  fig = go.Figure(data=data, layout=layout)
  graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
  return graphJSON


def arima(train_data,test_data):
  test_result=adfuller(train_data['point_value'])
  print('ADF Statistic: %f' % test_result[0])
  print('p-value: %f' % test_result[1])
  plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

  # Original Series
  fig, axes = plt.subplots(3, 2, sharex=True)
  axes[0, 0].plot(train_data['point_value']); axes[0, 0].set_title('Original Series')
  plot_acf(train_data['point_value'], ax=axes[0, 1])

  # 1st Differencing
  axes[1, 0].plot(train_data['point_value'].diff()); axes[1, 0].set_title('1st Order Differencing')
  plot_acf(train_data['point_value'].diff().dropna(), ax=axes[1, 1])

  # 2nd Differencing
  axes[2, 0].plot(train_data['point_value'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
  plot_acf(train_data['point_value'].diff().diff().dropna(), ax=axes[2, 1])

  #plt.show()

  model=ARIMA(train_data['point_value'],order=(1,1,1))
  model_fit=model.fit()

  fc = model_fit.forecast(15, alpha=0.05)  # 95% conf\
  df_test = pd.DataFrame(test_data['point_value'])
  test =np.array(df_test.replace(to_replace=0, method='ffill'))
  #print(y_test,preds)
  mape = np.mean(np.abs((np.array(fc) - test)/test))*100
  #plt.show()
  #print("MAPE Value : ",mape)
  summary =model_fit.summary()
  graphJSON = plot_model(train_data,fc)
  #print(fc)
  return graphJSON,['ARIMA',mape], summary

def Xgb(df_xg):
  x = np.array(df_xg.index)
  y = np.array(df_xg['point_value'])
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
  dtrain_reg = xgb.DMatrix(np.vstack(X_train),np.vstack(y_train), enable_categorical=True)
  dtest_reg = xgb.DMatrix(np.vstack(X_test), np.vstack(y_test), enable_categorical=True)
  params = {"objective": "reg:squarederror", "tree_method": "hist"}
  model = xgb.train(dtrain=dtrain_reg,params=params, num_boost_round=100)
  preds = model.predict(dtest_reg)
  df_test = pd.DataFrame(y_test)
  y_test =np.array(df_test.replace(to_replace=0, method='ffill'))
  #print(y_test,preds)
  mape = np.mean(np.abs((preds- y_test)/y_test))*100
  print(mape)
  graphJSON = plot_model(df_xg,pd.Series(preds).rename(lambda x: x+len(df_xg)-1))
  return graphJSON,['XgBoost',mape], []

def ets(train_data,test_data):
  train =pd.Series(train_data['point_value']).astype('float64')
  test = test_data['point_value']
  model = ETSModel(train)
  fit = model.fit(maxiter=10000)
  #fit.fittedvalues.plot(label="statsmodels fit")
  summary = fit.summary()
  fc = fit.forecast(len(test))
  y_test =np.array(test.replace(to_replace=0, method='ffill'))
  mape = np.mean(np.abs((fc- y_test)/fc))*100
  print(mape )
  graphJSON = plot_model(train_data,fc)
  return graphJSON,['ETS Model',mape], summary


def neural_prophet(train_data,test_data):
  train = train_data
  test = test_data
  train=train.rename(columns={'point_timestamp': 'ds', 'point_value': 'y'})
  test =test.rename(columns={'point_timestamp': 'ds', 'point_value': 'y'})
  train = train[['ds','y']]
  test = test['y']
  model = NeuralProphet()
  metrics = model.fit(train, freq="D") 
  future = model.make_future_dataframe(train, periods=len(test), n_historic_predictions=len(train)) 
  forecast = model.predict(future)
  fc = forecast[-len(test):]['yhat1']
  y_test =np.array(test.replace(to_replace=0, method='ffill'))
  #print(y_test,preds)
  mape = np.mean(np.abs((fc- y_test)/fc))*100
  graphJSON = plot_model(train_data,fc) 
  return graphJSON,['Neural Prophet',mape], []

