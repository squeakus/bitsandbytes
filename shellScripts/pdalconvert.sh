#!/bin/bash

CRS="PROJCS["NAD_1983_StatePlane_Maryland_FIPS_1900",GEOGCS["GCS_North_American_1983",DATUM["D_North_American_1983",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_1SP"],PARAMETER["False_Easting",400000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",-77.0],PARAMETER["Standard_Parallel_1",38.3],PARAMETER["Standard_Parallel_2",39.45],PARAMETER["Latitude_Of_Origin",37.66666666666666],UNIT["Meter",1.0]]"

find . -maxdepth 1 -name "*.las" | xargs -I fname pdal translate fname fname.laz --readers.las.spatialreference=$CRS

