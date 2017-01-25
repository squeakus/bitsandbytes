

folders = ['richviewoblique', 'bolandsoblique', 'platin']
detectors = ['sift', 'surf','orb','akaze', 'brisk']

outfile = open('runexperiments.sh','w')
outfile.write('#!/bin/bash \n')


outfile.write('echo `date`": starting experiments" > log.txt\n')


for folder in folders:
    for det in detectors:
	outfile.write('echo `date`": applying '+det+' to '+folder+'" >> log.txt\n')
	outfile.write('python comparedetectors.py '+folder+'/ '+det+'\n')

outfile.write('echo `date`": Finished!" >> log.txt\n')

outfile.close()
