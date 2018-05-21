from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.fixes import astype
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
import matplotlib.pyplot as plt
from scipy import sparse
import pandas as pd
import numpy as np

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


def dbscan(X, eps=0.5, minpts=5, metric='minkowski',
           algorithm='auto', leaf_size=30, p=2, sample_weight=None, n_jobs=1):

	X = check_array(X, accept_sparse='csr')
	if sample_weight is not None:
		sample_weight = np.asarray(sample_weight)
		check_consistent_length(X, sample_weight)

	if metric == 'precomputed' and sparse.issparse(X):
		neighborhoods = np.empty(X.shape[0], dtype=object)
		X.sum_duplicates()  # XXX: modifies X's internals in-place
		X_mask = X.data <= eps
		masked_indices = astype(X.indices, np.intp, copy=False)[X_mask]
		masked_indptr = np.cumsum(X_mask)[X.indptr[1:] - 1]
		# insert the diagonal: a point is its own neighbor, but 0 distance
		# means absence from sparse matrix data
		masked_indices = np.insert(masked_indices, masked_indptr,
		                           np.arange(X.shape[0]))
		masked_indptr = masked_indptr[:-1] + np.arange(1, X.shape[0])
		# split into rows
		neighborhoods[:] = np.split(masked_indices, masked_indptr)
	else:
		neighbors_model = NearestNeighbors(radius=eps, algorithm=algorithm, leaf_size=leaf_size, 
										   metric=metric, p=p, n_jobs=n_jobs)
		neighbors_model.fit(X)
		# This has worst case O(n^2) memory complexity
		neighborhoods = neighbors_model.radius_neighbors(X, eps, return_distance=False)
	if sample_weight is None:
		n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
	else:
		n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                for neighbors in neighborhoods])

    # Initially, all samples are noise.
	labels = -np.ones(X.shape[0], dtype=np.intp)

    # A list of all core samples found.
	core_samples = np.asarray(n_neighbors >= minpts, dtype=np.uint8)
	dbscan_inner(core_samples, neighborhoods, labels)
	return np.where(core_samples)[0], labels


class DBSCAN(BaseEstimator, ClusterMixin):

	def __init__(self, eps=0.5, minpts=7, metric='euclidean',
	     algorithm='auto', leaf_size=30, p=None, n_jobs=1):

		self.eps = eps
		self.minpts = minpts
		self.metric = metric
		self.algorithm = algorithm
		self.leaf_size = leaf_size
		self.p = p
		self.n_jobs = n_jobs

	def fit(self, X, y=None, sample_weight=None):

		X = check_array(X, accept_sparse='csr')
		clust = dbscan(X, sample_weight=sample_weight,
		           **self.get_params())
		self.core_sample_indices_, self.labels_ = clust
		if len(self.core_sample_indices_):
		# fix for scipy sparse indexing issue
			self.components_ = X[self.core_sample_indices_].copy()
		else:
		# no core samples
			self.components_ = np.empty((0, X.shape[1]))
		return self

	def fit_predict(self, X, y=None, sample_weight=None):

		self.fit(X, sample_weight=sample_weight)
		return self.labels_


X = []

for i, city in enumerate(cities):
	ctemps = cmeans.loc[city]
	temps = []
	for temp in ctemps:
		temp = float(temp)
		temps.append(temp)

	X.append(temps)

X = StandardScaler().fit_transform(X)

clustered = DBSCAN(eps=1.5, minpts=7).fit(X)

core_mask = np.zeros_like(clustered.labels_, dtype=bool)
core_mask[clustered.core_sample_indices_] = True
labels = clustered.labels_
unique_labels = set(labels)
n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0) #ignoring noise '-1'

colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

c = []
ii=0
for k, col in zip(unique_labels, colors):
	if k == -1:
		print("Noise:\n")
		col = 'k' #black = noise
		col1 = 'none'
	else:
		print ("Cluster: %d\n" % int(k+1))
		col1 = 'k'

	c.append([])
	c[k] = (city for city in cities[labels==k])

	class_mask = (labels == k)

	xy = X[class_mask]

	for i, (x, ck) in enumerate(zip(xy, c[k])):
		if core_mask[class_mask][i]:
			plt.plot(ii, x.mean(), 'o', markerfacecolor=col, markeredgecolor='k', markersize=12, alpha=0.7)
		else:
			plt.plot(ii, x.mean(), 'o', markerfacecolor=col, markeredgecolor=col1, markersize=8, alpha=0.75)
		
		ck = ck.split(',')[0]
		plt.text(ii+0.02, x.mean()-0.03, ck, size=7.5, alpha=1)
		print (ck)
		ii+=1
	print ("\n")
		

plt.title('DBSCAN\nNumber of clusters: %d' % n_clusters_)
plt.show()
