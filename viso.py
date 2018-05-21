from matplotlib import animation as a
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import warnings
warnings.filterwarnings('ignore')

file_name = ("./GlobalLandTemperatures/GlobalLandTemperaturesByMajorCity.csv")
data = pd.read_csv(file_name, parse_dates=['dt'])

data = data[data.dt.dt.year >= 1900]
data.City = data.City.str.cat(data.Country, sep=', ')
cmeans = data.groupby([data.City, data.dt.dt.year])['AverageTemperature'].mean().unstack()
data = data[['City', 'Country', 'Latitude', 'Longitude']].drop_duplicates()
cities = data['City']

fig = plt.figure(figsize=(13, 7), edgecolor='k')

START_YEAR = 2000 #min 1743
LAST_YEAR = 2013 #max 2013

class objs:

	def __init__(self, city, temps):
		self.city = city
		self.ids = -1
		self.rDist = -1
		self.temps = temps

	def dist(self, pt, year):
		return abs(self.temps[year] - pt.temps[year])

	def display(self, x, year):
		print('{x} \t {city} \n rDist: {rDist} \n'.format(x=x, city=self.city, rDist=self.rDist))
		plt.subplot(211)
		plt.text(x+1, self.temps[year]-1, self.city.split(',')[0], size=7, alpha=0.9, zorder=3)
		
		plt.subplot(212)
		plt.plot(x, self.rDist, 'o', markeredgecolor=None, markersize=4, alpha=1)
		plt.text(x+0.5, self.rDist-0.5, self.city.split(',')[0], size=7, alpha=1)
		return self.temps[year]

def getNeighbors(p, points, year, eps):

	N = []
	for point in points:
		if p.dist(point, year) <= eps:
			N.append(point)
	return N

def coreDist(p, N, year, eps, MinPts):

	if len(N) < MinPts:
		return -1
	else:
		min1 = p.dist(N[0], year)
		for neighbor in N[1:]:
			D = p.dist(neighbor, year)
			if D < min1:
				min1 = D
		return min1

def up(p, N, year, seeds, eps, MinPts):

	coredist = coreDist(p, N, year, eps, MinPts)
	for n in N:
		if (n.ids == -1): # UNPROCESSED
			if coredist == -1:
				new_rDist = -1
			else:
				new_rDist = max(coredist, p.dist(n, year)) #############
			if (n.rDist == -1): # -1:UNDEFINED, n is not in Seeds
				n.rDist = new_rDist
				seeds.append(n)
			else: # n in Seeds, check for improvement
				if (new_rDist < n.rDist):
					n.rDist = new_rDist


def OPTICS(points, year, eps, MinPts):

	i=0
	ol = []
	noise = []
	for point in points:
		if point.ids == -1: # UNPROCESSED
			N1 = getNeighbors(point, points, year, eps)
			point.ids = 1 # mark point as processed
			if coreDist(point, N1, year, eps, MinPts) == -1: # -1:UNDEFINED
				noise.append(point)
			else:
				ol.append([])
				ol[i].append(point) # output point to the ordered list
				seeds = [] # empty priority queue
				up(point, N1, year, seeds, eps, MinPts) ##
				for seed in seeds:
					N2 = getNeighbors(seed, points, year, eps)
					seed.ids = 1 # mark q as processed
					ol[i].append(seed) # output q to the ordered list
					if coreDist(seed, N2, year, eps, MinPts) != -1: # -1:UNDEFINED
						up(seed, N2, year, seeds, eps, MinPts)
				i+=1
	return ol, noise


X = []

for i, city in enumerate(cities):
	temps = cmeans.loc[city]		
	obj = objs(city, temps)
	X.append(obj)

def update(frame_number):

	plt.cla()
	plt.ylabel('Mean Temperature')
	current_year = START_YEAR + (frame_number % (LAST_YEAR - START_YEAR + 1))
	plt.text(25, 1, str(current_year), fontsize=100, alpha=0.25)
	plt.subplot(212)
	plt.cla()
	plt.xlabel('x')
	plt.ylabel('Reachability')

	for point in X:
		point.ids = -1
		point.rDist= -1

	ordered, noise = OPTICS(X, current_year, eps=1, MinPts=6)

	i=0
	colors = plt.cm.gist_ncar(np.linspace(0, 1, len(ordered)))

	x=[]
	y=[]
	col=[]
	for j, (cluster, color) in enumerate(zip(ordered, colors)):
		print ("\nCluster: %d\n" % int(j+1))
		for point in cluster:
			q = point.display(i, current_year)
			y.append(q)
			x.append(i)
			col.append(color)
			i+=1

	print ("\nOutliers\n")
	for point in noise:
		q=point.display(i, current_year)
		y.append(q)
		x.append(i)
		col.append('k')
		i+=1

	plt.ylim([-1.3, 3])
	plt.subplot(211)
	plt.ylim([0, 34])
	plt.scatter(x, y, c=col, edgecolor='k', alpha=0.75, zorder=1)

	plt.title('OPTICS\nNumber of clusters: %d' % len(ordered))

ani = a.FuncAnimation(fig, update, interval=30, frames=LAST_YEAR-START_YEAR+1)

plt.show()
