import random
from matplotlib import pylab

def main():
    workers = []
    drones = []
    workersdead = 0
    dronesdead = 0
    cycletype = ['egg', 'larva', 'capped', 'worker']
    workerplot = []
    for i in range(365):
        workers, drones, workdeadcnt, dronedeadcnt, workcount, dronecount = day(i, workers, drones)
        eggs = workcount['egg']
        larva = workcount['larva']
        capped = workcount['capped']
        worker =workcount['worker']

        workersdead += workdeadcnt
        dronesdead += dronedeadcnt

        workerplot.append([eggs, larva, capped, worker, workersdead])

        print "workersdead", workersdead, "dronesdead", dronesdead
    #print workerplot
    pylab.plot(workerplot)
    pylab.show()
def day(day, workers, drones):

    layrate = 2000
    droneprob = 0.1
    workcycle = {3, 9, 20, 62}
    dronecycle = {3, 10, 23, 65}

    cycletype = ['egg', 'larva', 'capped', 'worker']
    workcount = {'egg':0, 'larva':0, 'capped':0, 'worker':0}
    dronecount = {'egg':0, 'larva':0, 'capped':0, 'worker':0}


    workdead = []
    for idx, worker in enumerate(workers):
        workers[idx] += 1

        if workers[idx] > 62:
            workdead.append(idx)

        for idx, val in enumerate(workcycle):
            if worker < val:
                workcount[cycletype[idx]] += 1
                break

    workdeadcnt = len(workdead)
    workers = workers[workdeadcnt:]

    dronesdead = []
    for idx, drone in enumerate(drones):
        drones[idx] += 1

        if drones[idx] > 65:
            dronesdead.append(idx)

        for idx, val in enumerate(dronecycle):
            if drone < val:
                dronecount[cycletype[idx]] += 1
                break
    dronedeadcnt = len(dronesdead)
    drones = drones[dronedeadcnt:]
    if day < 30 or day > 55:
        workers, drones = layingqueen(workers, drones, layrate, droneprob)
    print "day:", day
    print "workers:", workcount
    print "drones", dronecount
    return workers, drones, workdeadcnt, dronedeadcnt, workcount, dronecount

def layingqueen(workers, drones, layrate, droneprob):
    for egg in range(layrate):
            if random.random() > droneprob:
                workers.append(0)
            else:
                drones.append(0)

    return workers, drones

if  __name__ == '__main__':
    main()
