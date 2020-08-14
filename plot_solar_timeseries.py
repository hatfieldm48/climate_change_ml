"""
Script to plot the time series data for solar data
  starting with the data found in 1601_18.46_-66.11_2016.csv
  hatfieldm

  links:
  - https://openei.org/datasets/dataset?sectors=buildings&tags=renewable+energy
  - https://openei.org/datasets/dataset/rooftop-solar-challenge-rsc-database/resource/2a27dca6-5d04-48ba-b799-2a1c4c1cf3d8
  - https://developer.nrel.gov/docs/solar/nsrdb/
  - https://developer.nrel.gov/docs/solar/nsrdb/puerto_rico_data_download/
  - https://medium.com/pursuitnotes/support-vector-regression-in-6-steps-with-python-c4569acd062d

  run commands:
  - python plot_solar_timeseries.py -f .\green-energy\1601_18.46_-66.11_2016.csv

  18.362169, -67.270845

"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, Panel
from bokeh.layouts import column, row, gridplot, widgetbox
from bokeh.models.widgets import Tabs


def argparser():
  parser = argparse.ArgumentParser()

  req_args = parser.add_argument_group('required arguments')
  req_args.add_argument('-f', dest='data_files', required=True, nargs='*', help='The path and filename for the data(s) to plot.')
  opt_args = parser.add_argument_group('optional arguments')
  opt_args.add_argument('-o', dest='output_dir', required=False, help='The path of the desired output location.')
  args = parser.parse_args()
  return args

def support_vector_regression(x, y):
  """
  helper function to execute an SVR model against x/y data from the solar dataset
  """

  ## List of Kernel's we'll return regressors for
  kernels = ['linear', 'poly', 'rbf', 'sigmoid']
  regressors = []

  ## Fitting the SVR Model the the dataset
  #regressor = SVR(kernel='rbf') #kernel type can be linear,poly, or gaussian. RBF is a type of guassian
  #regressor.fit(x,y)
  for k in kernels:
    regressor = SVR(kernel=k)
    regressor.fit(x,y)
    regressors.append(regressor)

  return kernels, regressors

def bokeh_lineplot(source, title='', x_name='x', y_name='y', is_datetime=False):
  """
  Function to create a lineplot in bokeh, with options for the datetime x axis or not
  """
  if is_datetime:
    p = figure(title=title, plot_width=1500, plot_height=400, x_axis_type='datetime')
  else:
    p = figure(title=title, plot_width=1500, plot_height=400)
  p.line(x=x_name, y=y_name, source=source)

  return p

def bokeh_scatterplot(source, title='', x_name='x', y_name='y', is_datetime=False):
  """
  Function to create a scatterplot in bokeh, with options for the datetime x axis or not
  """
  if is_datetime:
    p = figure(title=title, plot_width=1500, plot_height=400, x_axis_type='datetime')
  else:
    p = figure(title=title, plot_width=1500, plot_height=400)
  p.circle(x=x_name, y=y_name, source=source)

  return p

def bokeh_prediction_error_plot(source, title='', x_name='x', prediction_name='prediction', actual_name = 'actual'):
  """
  Just want this to be a plot where the x axis is a notional index, and were plotting the error on the y axis
    where each index has the real value (GHI) and the predictid value on the same index
  """

  p = figure(title=title, plot_width=1500, plot_height=400)
  p.circle(x=x_name, y=prediction_name, source=source, color='blue', legend_label=prediction_name)
  p.circle(x=x_name, y=actual_name, source=source, color='orange', legend_label=actual_name)

  return p

def main(data_files, output_dir):
  """
  Execution of data processing and bokeh html generation
  """

  ## Load the data into a pandas dataframe
  dataframes = []
  for data_file in data_files:
    if data_file[-4:] == '.csv':
      df = pd.read_csv(data_file, header=2)
      dataframes.append(df)
    else:
      print ('Err: non csv data files not supported at this time')
      return
  df = pd.concat(dataframes)

  ## set dataframe column types as needed
  #print (df.dtypes)

  ## Create a datetime field
  df['date_datetime'] = df.apply(lambda x: datetime(int(x['Year']), int(x['Month']), int(x['Day']), int(x['Hour']), int(x['Minute']), int(0)), axis=1) #Minute is always 5 for some reason

  ## Remove all records where it is "nighttime"
  #print (df.shape)
  df = df[df['Solar Zenith Angle'] < 89]
  #print (df.shape)
  
  ## Pull in scraped darsky data and match them up according to date_datetime
  darksky_csv = 'C:/Users/572784/Documents/Github/sbu-data-share/green-energy/darksky-weather/*.csv'#hourly_weather_attr.csv'
  dfs = []
  for csv in glob.glob(darksky_csv):
    df_darksky = pd.read_csv(csv)
    dfs.append(df_darksky)
  df_darksky = pd.concat(dfs)
  df_darksky['hour_date_stamp'] = pd.to_datetime(df_darksky['hour_date_stamp'])
  df_darksky['date_datetime'] = df_darksky['hour_date_stamp'].apply(lambda x: datetime(x.year, x.month, x.day, x.hour, 5, 0))
  
  ## Merge the two dataframes on their timestamps
  print (df.shape, df_darksky.shape)
  df = df.merge(df_darksky, how='inner', on=['date_datetime'])
  #print (df.head(20))
  #print (df.shape)

  ## Remove all records from merged df where cloudCover is NA
  print (df.shape)
  df = df.dropna(subset=['cloudCover'])
  print (df.shape, 'Dropped cloudCover nas')
  df = df.dropna(subset=['visibility'])
  print (df.shape, 'Dropped visibility nas')

  # Create lineplot for all data fields on interest
  data_fields = ['Air Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'DHI', 'DNI', 'GHI', 'Solar Zenith Angle', 'Albedo', 'Pressure', 'Precipitable Water', 'Wind Speed', 
    'cloudCover', 'visibility']
  lineplots = []
  for field in data_fields:
    df_field = df[['date_datetime', field]]
    source = ColumnDataSource(df_field)
    p = bokeh_lineplot(source, field, 'date_datetime', field, True)
    lineplots.append(p)

  # Create relationship scatter plots
  relationship_pairs = [('Precipitable Water','GHI'),('Air Temperature','GHI'),('Solar Zenith Angle','GHI'),('Wind Speed','GHI'),('Pressure','GHI'),
    ('cloudCover','GHI'),('visibility','GHI'),('Precipitable Water','precipProbability'),('Precipitable Water','precipIntensity')]
  relationship_plots = []
  for pair in relationship_pairs:
    df_relationship = df[[pair[0], pair[1]]]
    source = ColumnDataSource(df_relationship)
    p = bokeh_scatterplot(source, '{} vs {}'.format(pair[1],pair[0]), pair[0], pair[1])
    relationship_plots.append(p)

  """
  ## SVR Test: GHI vs Precipitable Wager
  precipitable_water = np.array([[x] for x in df['Precipitable Water']])
  ghi = np.array(df['GHI'])
  kernels, svr_regressors = support_vector_regression(precipitable_water, ghi) #svr_regressor = support_vector_regression(precipitable_water, ghi)
  """

  ## Plot SVR results
  """
  df_svr = df[['Precipitable Water','GHI']]
  #df_svr['GHI Prediction'] = df['Precipitable Water'].apply(lambda x: svr_regressor.predict(x))
  df_svr['GHI Prediction'] = svr_regressor.predict([[x] for x in df['Precipitable Water']])
  source_svr = ColumnDataSource(df_svr)
  p_svr = bokeh_scatterplot(source_svr, 'GHI vs Precipitable Water', 'Precipitable Water', 'GHI')
  """

  ## Create Features (X) and result value (y) from df
  
  #X = np.array(df[['Air Temperature','Solar Zenith Angle','Albedo','Pressure','Precipitable Water','Wind Speed',
  #  'cloudCover','visibility']])
  
  X = np.array(df[['Air Temperature','Solar Zenith Angle','Pressure','Precipitable Water',
    'cloudCover','visibility','precipIntensity','precipProbability']])

  #X = np.array(df[['Solar Zenith Angle']])

  y = np.array(df[['GHI']]).ravel()

  X = preprocessing.scale(X) # Set X to be [-1,1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  kernels, svr_regressors = support_vector_regression(X_train, y_train)
  prediction_plots = []
  for i, svr in enumerate(svr_regressors):
    k_type = kernels[i]
    svr_conf = svr.score(X_test, y_test)
    print (k_type, svr_conf)
  
    y_test_predicted = [svr.predict(val.reshape(1,-1))[0] for val in X_test]
    df_test_predicted = pd.DataFrame({
      'prediction': y_test_predicted,
      'actual': y_test
    })
    df_test_predicted = df_test_predicted.sort_values(by=['actual'])
    df_test_predicted['row_index'] = range(len(X_test))
    source_test_predicted = ColumnDataSource(df_test_predicted)
    p_test_predicted = bokeh_prediction_error_plot(source_test_predicted, title=k_type, x_name='row_index', prediction_name='prediction', actual_name='actual')
    prediction_plots.append(p_test_predicted)


  ## Output the plots as html
  if output_dir != None:
    output_file(output_dir + '/plot_solar_timeseries.html')
  else:
    output_file('plot_solar_timeseries.html')
  
  tab_timeseries_lineplots = Panel(child=column(lineplots), title = 'Timeseries Lineplots')
  tab_relationships = Panel(child=column(relationship_plots), title = 'Relationships')
  tab_prediction_vs_actual = Panel(child=column(prediction_plots), title = 'Predictions vs Actuals')

  tabs = Tabs(tabs=[tab_timeseries_lineplots, tab_relationships, tab_prediction_vs_actual])

  save(tabs)

  return

if __name__ == '__main__':
  args = argparser()
  main(args.data_files, args.output_dir)
  print ('Success!!')

