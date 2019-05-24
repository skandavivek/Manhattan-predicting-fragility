# Manhattan-predicting-fragility

With internet connected vehicles on the rise, new problems arise such as large-scale hacks/ software malfunction. It is essential to predict potential impacts of a large-scale hack on city traffic. There are multiple scenarios where the impacts of hacks on traffic are similar to collisions. Hence, I use collisions as a proxy for hacks in order to use real collision data to figure out potential consequences of hacks.

I use data from NYC DOT traffic speeds (https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds-NBE/i4gi-tjb9) containing over 24 Million speeds, as well as NOT DOT collision data (https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions/h9gi-nx95) which is a comprehensive record of collisions  (over 1.4 Million collisions ) across a span of 5 years. The data is obtained from the Socrata API using SoQL queries, for year 2018.

I find that speed is higly correlated with collision rate indicating that potentially just a few collisions can greatly hamper traffic speed. In particular, les than 20 collisions per hour corresponds to a speed reduction by 33%. One caution in interpreting these results is 'causation' vs 'correlation'. There could be underlying factors as to why collision rate is correlated with traffic speeds.

Finally, I developed a Random Forest Model to predict speeds based on: location of detector, time of day, day of week and collision rate, with correlation coefficient ~ 0.8.

To visualize collisions grouped by hour in Folium leaflet maps, open the visualization notebook in nbviewer:
https://nbviewer.jupyter.org/github/skandavivek/Manhattan-predicting-fragility/blob/master/Visualization.ipynb




