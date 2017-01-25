#!/bin/bash
date
psql -d dublin -c "create table points1(XCOORD double precision, YCOORD double precision, ZCOORD double precision, 
XRGB real, YRGB real, ZRGB real, INTENSITY real);"

psql -d dublin -c "copy points1 from '/home/jonathan/Jonathan/programs/pointcloud/damecleaned.xyz' DELIMITERS ' ' CSV;"
date
