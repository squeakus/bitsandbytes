import itertools, subprocess, time, os, multiprocessing

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

def main():
    homedir = os.getcwd()
    runs = range(1, 31)
    syncsize = [1, 10, 100, 500, 1000]
    offset = [0, 10, 20, 30, 40, 50, 60]

    start = time.time()
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    configs = list(itertools.product(*[runs,syncsize,offset]))


    for config in configs:
        seed = str(config[0])
        sync = str(config[1])
        offset = str(config[2])
        resultdir = homedir+"/sine/sync"+sync+"offset"+offset+"/"
        if not os.path.exists(resultdir): os.makedirs(resultdir)
        command = "java -jar ./bin/GEVA.jar -properties_file ./param/Parameters/sin.param"

        command += " -rng_seed "+ seed
        command += " -sync "+sync
        command += " -offset "+offset
        command += " -output "+resultdir
        pool.apply_async(run_cmd,(command,))
        
    pool.close()
    pool.join()
    print "time taken:", str(time.time()-start)

if __name__ == '__main__':
    main()

