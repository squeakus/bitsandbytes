#!/bin/bash

mkdir filtered

find . -name "*.laz" | xargs -n1 -P3 -I fname pdal pipeline noisefilter.json --writers.las.filename=filtered/fname --readers.las.filename=fname

