set terminal postscript color

set output "front.ps"

plot "Front000.dat" with points pointsize 0.5,\
    "Front001.dat" with points pointsize 0.5,\
    "Front002.dat" with points pointsize 0.5,\
    "Front003.dat" with points pointsize 0.5,\
    "Front004.dat" with points pointsize 0.5,\
    "Front005.dat" with points pointsize 0.5,\
    "Front006.dat" with points pointsize 0.5,\
    "Front007.dat" with points pointsize 0.5,\
    "Front008.dat" with points pointsize 0.5 
