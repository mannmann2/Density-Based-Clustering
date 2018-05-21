from mpl_toolkits.basemap import Basemap
from matplotlib import animation as a
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

file = ("./GlobalLandTemperatures/GlobalLandTemperaturesByMajorCity.csv")
f = pd.read_csv(file, parse_dates=['dt'])

#f = f[f.dt.dt.year >= 1900]
f.City = f.City.str.cat(f.Country, sep=', ')
cmeans = f.groupby([f.City, f.dt.dt.year])['AverageTemperature'].mean().unstack()
f = f[['City', 'Country', 'Latitude', 'Longitude']].drop_duplicates()
cities_info = f.groupby(['City']).first()
countries_info = f.groupby(['Country']).first()
cities = f['City']
countries = f['Country'].drop_duplicates()

lats, lons = [], []
for city in cities:
	coords = cities_info.loc[city][['Latitude', 'Longitude']].values
	lats.append(float(coords[0][:-1]) * (-1 if coords[0][-1] == 'S' else 1))
	lons.append(float(coords[1][:-1]) * (-1 if coords[1][-1] == 'W' else 1))

cities_info['Latitude'] = lats
cities_info['Longitude'] = lons

fig = plt.figure(figsize=(13, 7), edgecolor='k')
m = Basemap(projection='cyl', resolution='c')
''', llcrnrlat=-45, urcrnrlat=25, llcrnrlon=-70, urcrnrlon=140)'''
m.fillcontinents(color='w',lake_color='lightblue', zorder=1)
m.drawmapboundary(fill_color='lightblue')
m.drawcoastlines(linewidth=0.1)
m.drawcountries(linewidth=0.1)
cmap = plt.get_cmap('coolwarm')

START_YEAR = 1900 #min 1743
LAST_YEAR = 2013 #max 2013

year_text = plt.text(-170, 80, str(START_YEAR), fontsize=16)
mean_text = plt.text(-174, 92, 'Average World Teperature: -1', fontsize=9)

points = np.zeros(len(cities), dtype=[('lon', float, 1), ('lat', float, 1),
										('size', float, 1), ('color', float, 1)])

def get_points(year):

	mean = 0
	for i, city in enumerate(cities):
		ctemps = cmeans.loc[city]
		_MIN, _MAX, _MEDIAN = ctemps.min(), ctemps.max(), ctemps.median()
		temp = ctemps.loc[year]
		mean += temp
		points['lat'][i] = lats[i]
		points['lon'][i] = lons[i]
		points['size'][i] = 100 * abs(temp - _MEDIAN)
		points['color'][i] = (temp - _MIN) / (_MAX - _MIN)
		mean_text.set_text('Average World Teperature: {0:.4f}'.format(round(float(mean)/len(cities), 4)))
	return points

markers = get_points(START_YEAR)
x, y = m(markers['lon'], markers['lat'])
scat = m.scatter(x, y, s=markers['size'], c=markers['color'], cmap = cmap,
					marker='o', alpha=0.7, zorder=2)

def update(frame_number):

	current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
	markers = get_points(current_year)
	x, y = m(markers['lon'], markers['lat'])

	scat.set_offsets(np.dstack((x, y)))
	scat.set_color(cmap(markers['color']))
	scat.set_sizes(markers['size'])

	year_text.set_text(str(current_year))

ani = a.FuncAnimation(fig, update, interval=10, frames=LAST_YEAR-START_YEAR+1)

cbar = m.colorbar(scat, location='bottom')
cbar.set_label('\n0.0 - min temperature for city                                                                                  '
				+ '1.0 - max temperature for city')
plt.title('Mean temperatures for cities since {}'.format(START_YEAR))

x, y = m(lons, lats)
plt.plot(x, y, '.k', markersize=0.5)

for i, city in enumerate(cities):
	city = city.split(',')[0]
	plt.text(x[i]+0.6, y[i]+0.35, city, size=7, alpha=0.85)

for country in countries:
	coords = countries_info.loc[country][['Latitude', 'Longitude']].values
	clat = float(coords[0][:-1]) * (-1 if coords[0][-1] == 'S' else 1)
	clon = float(coords[1][:-1]) * (-1 if coords[1][-1] == 'W' else 1)

	cx, cy = m(clon, clat)
	#plt.text(cx-10, cy-7, country, size=9, alpha=1)

plt.show()
