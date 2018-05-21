We have implemented 3 clustering algorithms: DBSCAN, GDBSCAN and OPTICS

Along with this we also have 5 visualisations in the form of slef looping animations. The first two simply help us visualise the temperature changes at locations around the world over time. This is done on a map as well as by plotting the temperature on a graph with respect to measurement year.

To run the temperature animations type the following code in terminal or command prompt:

> python vis1.py

and

> python vis2.py

To view clusters from the entire dataset spanning multiple decades we may run the following commands:

> python dbscan.py

and

> python gdbscan.py

and 

> python optics.py

Along with this we have also clustered the data year wise and demonstrate the clusters obtained for each year in the form on an animation. To run these year wise animations do the following:

> python visd.py # for dbscan

> python visg.py # for gdbscan

> python viso.py # for optics 