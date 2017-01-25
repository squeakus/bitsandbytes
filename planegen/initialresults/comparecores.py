import pylab as P

core4file = open('4coreseed1.txt','r')
core8file = open('8coreseed1.txt','r')

core4list = core4file.readlines()
core8list = core8file.readlines()

times = []
core4times = []
core8times = []
for i in range(len(core4list)):
    core4 = eval(core4list[i])
    core8 = eval(core8list[i])
    times.append([core4['time'], core8['time']])
    if core4['time'] < 250:
        core4times.append(core4['time'])
    if core8['time'] < 250:
        core8times.append(core8['time'])
    
    timediff = core8['time'] - core4['time']
    liftdiff = core8['lift'] - core4['lift']
    dragdiff = core8['drag'] - core4['drag']
    print i, timediff, "lift", liftdiff, "drag", dragdiff

P.figure()
P.hist([core4times, core8times], bins=20, normed=1, histtype='bar')
P.legend(["4 core", "8 core (hyperthreaded)"])
P.xlabel("Time taken (seconds)")
#P.hist(core8times, bins=50, normed=1, histtype='bar')

#P.hist(core4times, bins=50, normed=1, histtype='bar')
#P.hist(core8times, bins=50, normed=1, histtype='bar')

P.show()
