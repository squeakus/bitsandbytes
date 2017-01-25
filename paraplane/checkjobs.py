


import time
from bwbfitness import run_cmd

def checkjobs():
    jobname = "planegen"
    jobsrunning = True
    starttime = time.time()

    while jobsrunning:
        time.sleep(10)
        qstat = run_cmd('qstat | grep '+jobname,qstat=True)
        qstat = qstat[0].split('\n')
        jobs = 0
        for line in qstat:
            if isinstance(line,str):
                if jobname in line:
                    jobs += 1
        print "jobs remaining:",jobs
        if jobs == 0:
            jobsrunning = False
    endtime = time.time() - starttime
    print "generation took", endtime, "seconds"


if __name__=='__main__':
   checkjobs()
