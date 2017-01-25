
import os
datasets = ['Bark','Boat','Graf','Trees','Wall','Bikes','Leuven','UBC'] 

outfile = open('runexperiment.sh','w')
detectors = ['sift', 'surf', 'orb', 'akaze', 'brisk']

outfile.write('#/bin/bash\n')
for name in datasets:
    outfile.write("echo \'date\'\":starting experiment "+name+"\"> log.txt\n")

    for det in detectors:
	outfile.write("python comparedetectors.py "+name+"/ "+ det+ "\n")
	outfile.write("echo \'date\'\":"+det+" finished\">> log.txt\n")

