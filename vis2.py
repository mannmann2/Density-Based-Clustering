from matplotlib import animation as a
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore')

file = ("./GlobalLandTemperatures/GlobalLandTemperaturesByMajorCity.csv")
f = pd.read_csv(file, parse_dates=['dt'])
#f = f[f.dt.dt.year >= 1900]
f.City = f.Country.str.cat(f.City, sep=', ')
cmeans = f.groupby([f.City, f.dt.dt.year])['AverageTemperature'].mean().unstack()
f = f[['City', 'Country']].drop_duplicates()
cities = f['City'].drop_duplicates()
print (cities)
cities = cities.iloc[:15]
cities.sort()
countries = f['Country'].drop_duplicates()
countries.sort()

fig = plt.figure(figsize=(14, 7), edgecolor='k')
cmap = plt.get_cmap('RdBu_r')

START_YEAR = 1915 #min 1743
LAST_YEAR = 2013 #max 2013

plt.ylim([0, 32])
plt.xlim([START_YEAR, LAST_YEAR+1])

year_text = plt.text(START_YEAR+(LAST_YEAR-START_YEAR)/2-20, 1, str(START_YEAR), fontsize=100, alpha=0.25)
mean_text = plt.text(START_YEAR-5, -5, 'Average World Temperature: -1', fontsize=9)

points = np.zeros(len(cities), dtype=[('x', float, 1), ('y', float, 1),
										('size', float, 1), ('color', float, 1), ('mean', float, 1), ('med', float, 1)])

def get_points(year):
	
	count = mean = 0
	for i, city in enumerate(cities):
		ctemps = cmeans.loc[city]
		_MEAN, _MIN, _MAX, _MEDIAN = ctemps.mean(), ctemps.min(), ctemps.max(), ctemps.median()
		temp = ctemps.loc[year]
		if isinstance(temp, float):
			mean += temp
			count += 1
		r = ((temp - _MIN) / (_MAX - _MIN)) 
		points['x'][i] =  year
		points['y'][i] = float(temp) # float(r)
		points['size'][i] = 100 * abs(temp - _MEDIAN)
		points['color'][i] = float(r)
		points['med'][i] = _MEDIAN
		points['mean'][i] = _MEAN
		mean_text.set_text('Average World Temperature: {0:.4f}'.format(round(float(mean)/count, 4)))
	return points

points = get_points(START_YEAR)
x = points['x']
y = points['y']
s = plt.scatter(x, y, s=points['size'], c=points['color'], edgecolor='k', cmap=cmap, alpha=1, zorder=2)

city_text = []
for city, mean, med in zip(cities, points['mean'], points['med']):
	t = plt.text(LAST_YEAR+2, mean+1, city.split(', ')[1], fontsize=8, alpha=0.8)
	city_text.append(t)

# x = 0.2
# country_text = []
# for i, country in enumerate(countries):
# 	cmean = count = 0
# 	for city in cities:
# 		if city.split(', ')[0] == country:
# 			count+=1
# 			cmean += cmeans.loc[city].mean()
# 	if count > 0: 
# 		cmean = cmean/count
# 	x+=count
# 	plt.plot([x, x], [0, 42], alpha=0.4)
# 	c = plt.text(x-count+1, cmean+10+random.uniform(-2, 2), country+': -1', fontsize=8, alpha=0.9)
# 	country_text.append(c)
# country_text[i].set_text('{0}: {1:.2f}'.format(country, cmean))

def update(frame_number):

	current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
	year_text.set_text(str(current_year))
	points = get_points(current_year)
	x = points['x']
	y = points['y']
	# s.set_offsets(np.dstack((x, y)))
	plt.scatter(x, y, s=points['size'], c=points['color'], cmap=cmap, edgecolor='k', alpha=1, zorder=2)
	s.set_color(cmap(points['color']))
	s.set_sizes(points['size'])
	for i, city in enumerate(cities):
		city_text[i].set_text(str(city.split(', ')[1]))
	
ani = a.FuncAnimation(fig, update, interval=50, frames=LAST_YEAR-START_YEAR+1)

cbar = plt.colorbar(s, orientation='horizontal', aspect=100)
# cbar.set_label('\n0.0 - min temperature for city                                                                                                         '
#  			 + '1.0 - max temperature for city')
plt.xlabel('Avg. Temperature\n')
plt.ylabel('Temperature\n\n')
plt.title('Temperature change in cities since {}'.format(START_YEAR))
plt.grid()
plt.show()
