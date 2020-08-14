"""
Script to facilitate webscraping darksy for historical and/or current and/or future weather data
  hatfieldm

  links:
  - https://darksky.net/dev/docs

  run commands:
  - python webscrape_weather.py -lat 18.4655 -lon "-66.1057" -year 2016
  - python webscrape_weather.py -lat 18.4655 -lon "-66.1057" -date 01/01/2016 -time 12:00:00
  - python webscrape_weather.py -j ./green-energy/darksky-weather/

  python webscrape_weather.py -lat 18.362169 -lon "-67.270845" -year 2016
"""

import argparse
import sys
import os
import glob
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from datetime import timedelta
from pytz import timezone

api_key = 'fdc3c16067c0697e2a032e2154e687e2'

def argparser():
  parser = argparse.ArgumentParser()

  req_args = parser.add_argument_group('required arguments')
  #req_args.add_argument('-f', dest='data_files', required=True, nargs='*', help='The path and filename for the data(s) to plot.')
  opt_args = parser.add_argument_group('optional arguments')
  opt_args.add_argument('-lat', dest='latitude', required=False, help='The latitude of the search location')
  opt_args.add_argument('-lon', dest='longitude', required=False, help='The longitude of the search location')
  opt_args.add_argument('-date', dest='date', required=False, help='The date to request, as formatted like MM/DD/YYYY')
  opt_args.add_argument('-time', dest='time', required=False, help='The time to request, as formatted like HH:MM:SS')
  opt_args.add_argument('-year', dest='year', required=False, help='Only for bulk api downloads. Will download the daily'+
  	' historical weather for this location for each day (1 day per file) for the full year')
  opt_args.add_argument('-j', dest='json_dir', required=False, help='The file path to already downloaded daily json files.'+
  	' If present, the code will not invoke any new api calls, but will pull the historical daily weather info into a csv file.')

  #opt_args.add_argument('-o', dest='output_dir', required=False, help='The path of the desired output location.')
  args = parser.parse_args()
  return args

def daterange(start_date, end_date):
  """
  Function to great a list with an item for each date between the start date and end date
  [start_date, end_date)
  """
  for n in range(int((end_date - start_date).days)):
  	yield start_date + timedelta(n)


def get_hourly_json_data(darksky_json, attr_key):
  """
  Given a json object which represents the weather for a day from darksky,
    extract the hourly data for the provided key
  Possible attr_keys:
    "precipIntensity":0.0138, "precipProbability":0.36,"precipType":"rain","temperature":76.18,
    "apparentTemperature":77.47,"dewPoint":71.19,"humidity":0.85,"pressure":1018.8,"windSpeed":7.59,
    "windGust":8.81,"windBearing":127,"cloudCover":0.75,"uvIndex":0,"visibility":9.997
  """

  hourly_data = darksky_json['hourly']['data']
  tz = timezone(darksky_json['timezone'])
  
  if attr_key == '':
  	attr_key = ['precipIntensity','precipProbability','precipType','temperature',
  	  'apparentTemperature','dewPoint','humidity','pressure','windSpeed','windGust',
  	  'windBearing','cloudCover','uvIndex','visibility']
  	for hour_data in hourly_data:
  		hour_data['hour_date_stamp'] = datetime.fromtimestamp(hour_data['time'], tz=tz)

  	return hourly_data, -1

  ## Each entry in hourly_data is the set of attributes for a given hour timestep
  ##   they aren't necessarily in order
  ##   the key 'time' gives the UTC seconds after 1/1/1970
  ##   I *think* I want to convert each UTC seconds time to a datetime, and zip that up with the data point of interest
  #hourly_attr = [(datetime.fromtimestamp(x['time'], tz=tz), x[attr_key]) for x in hourly_data]
    # ^Adjusting this from listcomp to for loop because it seems not every hour has each attribute
  hourly_attr = []
  num_valnotfound = 0
  for x in hourly_data:
  	hour_date_stamp = datetime.fromtimestamp(x['time'], tz=tz)
  	if attr_key in x: #Because not every hour has each value apparently
  		attr_value = x[attr_key]
  		hourly_attr.append((hour_date_stamp, attr_value))
  	else:
  		num_valnotfound = num_valnotfound + 1
  		hourly_attr.append((hour_date_stamp, -1))

  return hourly_attr, num_valnotfound


def create_api_request(latitude, longitude, requested_datetime, format):
  """
  A function to help create the api request for pulling weather information
    as of 3/23/2020, this only supports darksky
  example: https://api.darksky.net/forecast/0123456789abcdef9876543210fedcba/42.3601,-71.0589,255657600?exclude=currently,flags
  """
  
  url = 'https://api.darksky.net/forecast/'
  
  ## Time formatting req:
  # UNIX time (that is, seconds since midnight GMT on 1 Jan 1970)
  #      OR
  # a string formatted as follows: [YYYY]-[MM]-[DD]T[HH]:[MM]:[SS][timezone]
  #    timezone can be left out to refer to local time of lat/long requested
  dt_str_format = '%Y-%m-%dT%H:%M:%S'
  dt_string = requested_datetime.strftime(dt_str_format)

  url = url + api_key + '/' + latitude + ',' + longitude + ',' + dt_string
  print (url)

  response = requests.get(url)
  #print (response.status_code)
  #print (response.status_code=='200')
  #print (response.status_code==200)
  #response_json = response.json()
  #print (response_json)
  if (response.status_code==200):
    return response.json()
  else:
    print ('ERROR:', response.status_code)
    return 
  return


def main():

  return

if __name__ == '__main__':
  args = argparser()
  save_dir = 'C:/Users/572784/Documents/Github/sbu-data-share/green-energy/darksky-weather/'

  # if given a json path, then we don't want to do any of the api calls, just load json files and create a csv
  if args.json_dir != None:
    json_files = glob.glob(args.json_dir + '*')
    #print (json_files, len(json_files))
    all_attr_data = [] #all_cloud_data = []
    total_valnotfound = 0
    for json_file in json_files:
    	with open(json_file, 'r') as f:
    		data = json.load(f)
    	hourly_attr_data, _ = get_hourly_json_data(data, '') #hourly_cloud_data, num_valnotfound = get_hourly_json_data(data, 'cloudCover')
    	#total_valnotfound = total_valnotfound + num_valnotfound
    	all_attr_data.extend(hourly_attr_data) #all_cloud_data.extend(hourly_cloud_data)
    
    #print (all_attr_data)
    df = pd.DataFrame.from_records(all_attr_data) #pd.DataFrame.from_records(all_cloud_data)
    df.to_csv(args.json_dir + '/hourly_weather_attr.csv')

    print ('Successfully Loaded JSON Files Into CSV!!', args.json_dir + '/hourly_weather_attr.csv')
    sys.exit() #Exit without continuing onto any of the other code

  # If given just a year, the loop through all days in the year and download the daily darsky weather for this date
  if args.year != None:
    year = int(args.year)
    start_date = datetime(year, 1, 1)
    end_date = datetime(year+1, 1, 1)
    list_of_dates = list(daterange(start_date, end_date))
    
    for d in list_of_dates:
      d_str = d.strftime('%m/%d/%Y')
      requested_datetime = datetime.strptime(' '.join([d_str, '12:00:00']),'%m/%d/%Y %H:%M:%S')
      data = create_api_request(args.latitude, args.longitude, requested_datetime, '')
      with open(save_dir + 'dailyweather_' + requested_datetime.strftime('%m-%d-%Y-%H-%M-%S') + '.json','w') as f:
        json.dump(data, f)
  # If no year, then assume we were given a specific date/time, and download that day only
  else:
    requested_datetime = datetime.strptime(' '.join([args.date, args.time]),'%m/%d/%Y %H:%M:%S')
    data = create_api_request(args.latitude, args.longitude, requested_datetime, '')
    with open(save_dir + 'dailyweather_' + requested_datetime.strftime('%m-%d-%Y-%H-%M-%S') + '.json','w') as f:
      json.dump(data, f)


  #with open('C:/Users/572784/Documents/Github/sbu-data-share/green-energy/example_darksky_historical_daily.json', 'r') as f:
  #  darksky_example = json.load(f)
  #hourly_cloud_data, _ = get_hourly_json_data(darksky_example, 'cloudCover')
  #print (hourly_cloud_data)

  ###
  # Next
  #   separate argument/entry point/function for when we don't want to download, but want to parse out the existing data. Basically just want to loop through
  #   all json files between dates for a given lat/lon and extract the attribute of interest, and then save that into a more manageable file format

  print ('Success!!')