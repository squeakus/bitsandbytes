import math, sys, time
import pp


def multi_loop(n):
    for i in range(n):
        for j in range(n):
            x = i + j 
    return x

#nodes = ("*:35000",)
nodes = ("*",)


if len(sys.argv) > 1:
    ncpus = int(sys.argv[1])
    # Creates jobserver with ncpus workers
    job_server = pp.Server(ncpus, ppservers=nodes)
else:
    # Creates jobserver with automatically detected number of workers
    job_server = pp.Server(ppservers=nodes)
print "Starting pp with", job_server.get_ncpus(), "workers"

# func, args, depfuncs, modules, callback
#job1 = job_server.submit(multi_loop, (10000,))
#result = job1()
#print "result", result

inputs = (20000, 20100, 20200, 20300, 20400, 20500, 20600)

start_time = time.time()

jobs = [(input, job_server.submit(multi_loop, (input,))) for input in inputs]

for input, job in jobs:
    print job_server.get_active_nodes()
    print "loop", input, "=", job()
#for input in inputs:
#    multi_loop(input)

print "after", job_server.get_active_nodes()


print "Time elapsed: ", time.time() - start_time, "s"
