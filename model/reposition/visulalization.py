#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:37:21 2020

@author: apple
"""

import pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
#from train_value_table import FNN,decode_state,loc_table
from shapely.geometry import Polygon
import contextily as ctx
import geopandas as gpd
from train_pure_value_table import *

#visualization    

table = loc_table(load_file = 'table_table.pkl')
#table.load_NN_table('/Users/apple/Documents/EECS598/final project/nn_table.pkl')
#value_network = torch.load("/Users/apple/Documents/EECS598/final project/value_network.pkl")
grids = pd.read_csv('hexagon_grid_table.csv', names=['grid_id', 'lng1', 'lat1', 'lng2', 'lat2', 'lng3', 'lat3', 'lng4', 'lat4', 'lng5', 'lat5', 'lng6', 'lat6'])
grids = grids.drop(4183)
lng = (grids['lng1']+grids['lng2']+grids['lng3']+grids['lng4']+grids['lng5']+grids['lng6'])/6
lat = (grids['lat1']+grids['lat2']+grids['lat3']+grids['lat4']+grids['lat5']+grids['lat6'])/6
grids['value'] = None
#t = np.ones(len(lng))*1477958400 #2016/11/1/8/0/0
#t = np.ones(len(lng))*1479686400 #2016/11/1/16/0/0
t = np.ones(len(lng))*1478060000 #2016/11/3/16/0/0
b = np.array([t,np.array(lng),np.array(lat)])
b = np.transpose(b)
b = decode_state(b)
tmp = []
#b = torch.tensor(decode_state(b), dtype = torch.float)
print('start')
for i in range(len(lng)):
    #x,y = table.look_up(b[i,-2], b[i,-1])
    #grid_values = table.NN_table[x][y]['NN'].forward(torch.tensor(b[i,0:2], dtype = torch.float))
    grid_values = table.look_up_value(b[i,-2],b[i,-1],1478060000)
    tmp.append(grid_values)
print('finish')
grids['value'] = tmp
#grids['value'] = grid_values.detach().numpy()
grids.sort_values(by = 'lng1')


plt.figure(1)
plt.hist(grids['lat1'], bins = 50)
plt.yscale('log')

plt.figure(2)
plt.hist(grids['lng1'], bins = 50)
plt.yscale('log')

grids_vis = grids.loc[(grids['lng1'] > 103.78) &
                      (grids['lng1'] < 104.26) &
                      (grids['lat1'] > 30.45) &
                      (grids['lat1'] < 30.88)]

coords = grids_vis.iloc[:,1:-1].to_numpy().reshape(-1,6,2)
values = grids_vis['value'].to_numpy()
print('load_picture')
hexogons = gpd.GeoSeries([Polygon(coords[i]) for i in range(len(coords))])

d = dict(geometry=hexogons, value=values)
gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
gdf.to_crs(epsg=3857)

ax = gdf.plot(figsize=(10,10), column='value', cmap='coolwarm', alpha=0.4)
ctx.add_basemap(ax, crs=gdf.crs.to_string(), zoom=12,
                source=ctx.providers.Stamen.TonerLite)