import multiprocessing,random

def worker(num):
    for i in range(1000):
        count = random.choice([100,10,1000])
        for j in range(count):
            x=5
    print 'Worker:', num
    return num

def finished(result):
  print "Done! result=%r" % (result,)

if __name__ == '__main__':
    worker(10)
    count = multiprocessing.cpu_count()
    print "cpu count: ",count	
    pool = multiprocessing.Pool(processes=count)

    for i in xrange(1,100):
        pool.apply_async(worker,(i,))

    pool.close()
    pool.join()
