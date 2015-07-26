import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

m = Basemap(projection = "mill", llcrnrlat = 20, urcrnlat = 50, llcrnrlon=-130, urcrnlon=-60, resolution = 'c')
m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.fillcontinents(color = '#04BAE3', lake_color="#FFFFFF")
m.drawmapboundary(fill_color = "#FFFFFF")

training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")