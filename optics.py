import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings('ignore')

file_name = ("./GlobalLandTemperatures/GlobalLandTemperaturesByCity.csv")
data = pd.read_csv(file_name, parse_dates=['dt'])

data = data[data.dt.dt.year >= 1900]
data.City = data.City.str.cat(data.Country, sep=', ')
cmeans = data.groupby([data.City, data.dt.dt.year])['AverageTemperature'].mean().unstack()
data = data[['City', 'Country', 'Latitude', 'Longitude']].drop_duplicates()
cities = data['City']

plt.figure(figsize=(13, 7), edgecolor='k')

class objs:

	def __init__(self, city, temps, _mean, _max, _min):
		self.city = city
		self.ids = -1
		self.rDist = -1
		self.temps = temps
		self._mean = _mean
		self._max = _max
		self._min = _min

	def dist(self, pt):
		return abs(self._mean - pt._mean)

	def display(self, x, col):
		print('{x} \t {city} \n Mean: {mean} \n rDist: {rDist} \n'.format(x=x, city=self.city, mean=self._mean, rDist=self.rDist))
		plt.subplot(211)
		plt.plot(x, self._mean, 'o', markerfacecolor=col, markeredgecolor='k', markersize=8, alpha=0.75)
		plt.text(x+1, self._mean-1, self.city.split(',')[0], size=6, alpha=1)
		plt.subplot(212)
		plt.plot(x, self.rDist, 'o', markerfacecolor=col, markeredgecolor=None, markersize=2, alpha=1)
		plt.text(x+0.5, self.rDist-0.5, self.city.split(',')[0], size=6, alpha=1)

def getNeighbors(p, points, eps):

	N = []
	for point in points:
		if p.dist(point) <= eps:
			N.append(point)
	return N

def coreDist(p, N, eps, MinPts):

	if len(N) < MinPts:
		return -1
	else:
		min1 = p.dist(N[0])
		for neighbor in N[1:]:
			D = p.dist(neighbor)
			if D < min1:
				min1 = D
		return min1

def update(p, N, seeds, eps, MinPts):

	coredist = coreDist(p, N, eps, MinPts)
	for n in N:
		if (n.ids == -1): # UNPROCESSED
			if coredist == -1:
				new_rDist = -1
			else:
				new_rDist = max(coredist, p.dist(n)) #############
			if (n.rDist == -1): # -1:UNDEFINED, n is not in Seeds
				n.rDist = new_rDist
				seeds.append(n)
			else: # n in Seeds, check for improvement
				if (new_rDist < n.rDist):
					n.rDist = new_rDist


def OPTICS(points, eps, MinPts):

	i=0
	ol = []
	noise = []
	for point in points:
		if point.ids == -1: # UNPROCESSED
			N1 = getNeighbors(point, points, eps)
			point.ids = 1 # mark point as processed
			if coreDist(point, N1, eps, MinPts) == -1: # -1:UNDEFINED
				noise.append(point)
			else:
				ol.append([])
				ol[i].append(point) # output point to the ordered list
				seeds = [] # empty priority queue
				update(point, N1, seeds, eps, MinPts) ##
				for seed in seeds:
					N2 = getNeighbors(seed, points, eps)
					seed.ids = 1 # mark q as processed
					ol[i].append(seed) # output q to the ordered list
					if coreDist(seed, N2, eps, MinPts) != -1: # -1:UNDEFINED
						update(seed, N2, seeds, eps, MinPts)
				i+=1
	return ol, noise


X = []

for i, city in enumerate(cities):
	ctemps = cmeans.loc[city]
	temps = []
	for temp in ctemps:
	 	temp = float(temp)
	 	temps.append(temp)
	
	_MIN, _MAX, _MEAN = ctemps.min(), ctemps.max(), ctemps.mean()
	obj = objs(city, temps, _MEAN, _MAX, _MIN)
	X.append(obj)

ordered, noise = OPTICS(X, eps=0.4, MinPts=25)

x=0
colors = plt.cm.gist_ncar(np.linspace(0, 1, len(ordered)))
for i, (cluster, color) in enumerate(zip(ordered, colors)):
	print ("\nCluster: %d\n" % int(i+1))
	for point in cluster:
		point.display(x, color)
		x+=1

print ("\nOutliers\n")
for point in noise:
	point.display(x, 'k')
	x+=1

plt.xlabel('x')
plt.ylabel('Reachability')
plt.ylim([-1.5, 5])
plt.subplot(211)
plt.ylabel('Mean Temp')
plt.title('OPTICS\nNumber of clusters: %d' % int(i+1))
plt.show()
