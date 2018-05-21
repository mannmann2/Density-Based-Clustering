from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = ("./GlobalLandTemperatures/GlobalLandTemperaturesByMajorCity.csv")
f = pd.read_csv(file , parse_dates=['dt'])

f.City = f.City.str.cat(f.Country, sep=', ')
f = f[['City', 'Latitude', 'Longitude']].drop_duplicates()

fig = plt.figure(figsize=(12, 7), edgecolor='k')
m = Basemap(projection='cyl', resolution='c')
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='w',lake_color='lightblue')
m.drawmapboundary(fill_color='lightblue')

cities= []
for c in f['City']:
	cities.append(c)

lats = []
for lat in f['Latitude']:
	if lat[-1] == 'S':
		lats.append(-float(lat[ :-1]))
	else:
		lats.append(float(lat[ :-1]))

lons = []
for lon in f['Longitude']:
	if lon[-1] == 'W':
		lons.append(-float(lon[ :-1]))
	else:
		lons.append(float(lon[ :-1]))

latlngdict = []
for city, lon, lat in zip(cities, lons, lats):
	x, y = m(lon, lat)
	plt.text(x+0.6, y+0.35, city, size = 7, alpha = 0.65)
	plt.plot(x, y, 'ob', markersize = 1.5)
	latlngdict.append({'lat': y, 'lng': x})

plt.title("Plotting cities with Basemap")
plt.show()

f = open('map.html', 'w')

f.write(
'''<!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
    <meta charset="utf-8">
    <title>Marker Clustering</title>
    <style>
      #map {
        height: 100%;
      }
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
      }
    </style>
  </head>
  <body>
    <div id="map"></div><script>

      function initMap() {
        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 2,
          center: {lat: 25, lng: 15}
        });

        // Create an array of alphabetical characters used to label the markers.
        //var labels = ''' + str(cities) + ''';
        var markers = locations.map(function(location, i) {
          return new google.maps.Marker({
            position: location
            //label: labels[i % labels.length]
          });
        });

        // Add a marker clusterer to manage the markers.
        var markerCluster = new MarkerClusterer(map, markers,
        {imagePath:'https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/m'});
      }
      var locations =  ''' + str(latlngdict) + '''
    </script>
    <script src="https://developers.google.com/maps/documentation/javascript/examples/markerclusterer/markerclusterer.js">
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDX0mEB3j8ILwJXUbJJmuEOwgCTy1lnUu4&callback=initMap">
    </script>
  </body>
</html>
''')

f.close()
