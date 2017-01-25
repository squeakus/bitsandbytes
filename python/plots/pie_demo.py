from pylab import *

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

labels = 'Low Fitness', 'High Fitness', 'No Preference'
fracs = [55.9, 36.84, 7.26]

explode=(0, 0.05, 0)
pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
title('Survey Results', bbox={'facecolor':'0.8', 'pad':5})

show()
