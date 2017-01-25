#!/bin/bash
#PBS -l nodes=50:ppn=24
#PBS -l walltime=03:00:00
#PBS -N planegen
#PBS -A ucd01
#PBS -o out.log
#PBS -e err.log

module load apps
module load openfoam/intel/2.2.2
source /ichec/packages/OpenFOAM/2.2.2/intel/14.0.0.080/OpenFOAM-2.2.2/etc/bashrc

cd $PBS_O_WORKDIR
cp $PBS_NODEFILE ./hostfile

echo "1200 Core: Starting at"
date
T="$(date +%s)"

for k in `seq 50` ; do
    dirname=cfd`printf %03d $k` 
    echo "changing to dir: $dirname"
    cd $dirname

    # Run on first 24 hosts in hostfile:
    head -n 24 ../hostfile > ./hostfile
    ./fionncfd &
    cd ..
    # delete first 24 lines from hostile:
    sed -i 1,24d ./hostfile
done

echo "Run completed at "
date
T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"


