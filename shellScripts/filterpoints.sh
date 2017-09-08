#!/bin/bash

mkdir filtered

find . -maxdepth 1 -name "*.laz" | xargs -n1 -P3 -I fname sh -c 'echo fname; pdal -v 4 pipeline noisefilter.json --writers.las.filename=filtered/fname --readers.las.filename=fname'
/home/jonathan/code/vola/VOLA/vola_api/las2vola.py "filtered/*.laz" 4 --crs 29902 -n 
