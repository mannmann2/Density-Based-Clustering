import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
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

plt.figure(figsize=(13, 7), edgecolor='k')

UNCLASSIFIED = -2
NOISE = -1

class Point:
    def __init__(self, city, temps, _mean, _max, _min):
        self.city = city
        self.temps = temps
        self._mean = _mean
        self._max = _max
        self._min = _min
        self.cluster_id = UNCLASSIFIED

    def change_cluster_id(self, cluster_id):
    	self.cluster_id = cluster_id

    def display(self, i, col):
        print ('{}, Mean: {}'.format(self.city, self._mean))
        plt.plot(i, self._mean, 'o', markerfacecolor=col, markeredgecolor='k', markersize=10, alpha=0.8)
        plt.text(i-0.5, self._mean-0.9, self.city.split(',')[0], size=6, alpha=1)


def dist(p1, p2):
    # return abs(p1._mean - p2._mean)
    tx = 0
    for temp1, temp2 in zip(p1.temps, p2.temps):
        tx = tx + pow(temp1 - temp2, 2)
    return math.sqrt(tx)

def getNeighbors(points, p, eps):
	N = []
	for point in points:
		if dist(point, p) <= eps:
			N.append(point)
	return N

def expand_cluster(points, point, cluster_id, eps, MinPts):

	seeds = getNeighbors(points, point, eps)
	if len(seeds) < MinPts:
		point.change_cluster_id(NOISE)
		return False
	else:
		for seed in seeds:
			seed.change_cluster_id(cluster_id)

	seeds1 = (p for p in seeds if p != point) # seeds.remove(point)
	seeds = list(seeds1)

	while len(seeds) > 0:
		current_point = seeds[0]
		results = getNeighbors(points, current_point, eps)
		if len(results) >= MinPts:
			for result in results:
				if result.cluster_id in [UNCLASSIFIED, NOISE]:
					if result.cluster_id == UNCLASSIFIED:
						seeds.append(result)
					result.change_cluster_id(cluster_id)

		seeds1 = (p for p in seeds if p != current_point)
		seeds=list(seeds1)        
	return True


def DBSCAN(points, eps, MinPts):
    
    points = copy.deepcopy(points)
    cluster_id = 0
    clusters = {}
    for point in points:
        if point.cluster_id == UNCLASSIFIED:
            if expand_cluster(points, point, cluster_id, eps, MinPts):
                cluster_id = cluster_id + 1

    for point in points:
        key = point.cluster_id
        if key in clusters:
            clusters[key].append(point)
        else:
            clusters[key] = [point]

    return clusters


X = []

for i, city in enumerate(cities):
    ctemps = cmeans.loc[city]
    temps = [] 
    for temp in ctemps:
        temp = float(temp)
        temps.append(temp)
    _MIN, _MAX, _MEAN = ctemps.min(), ctemps.max(), ctemps.mean()

    point = Point(city, temps, _MEAN, _MAX, _MIN)
    X.append(point)

clusters = DBSCAN(X, 12, 3)

i=0
colors = plt.cm.gist_ncar(np.linspace(0, 1, len(clusters)))

for key, color in zip(clusters, colors):
    if key == -1:
        print ('Outliers:')
    else:
        print ('Cluster: %d' % int(key+1))
    for p in clusters[key]:
        p.display(i, color)
        i+=1
    print ("\n")

plt.xlabel('x')
plt.ylabel('Mean Temperature\n')
plt.title('DBSCAN\nNumber of clusters: %d' % int(len(clusters) - 1))
plt.show()
