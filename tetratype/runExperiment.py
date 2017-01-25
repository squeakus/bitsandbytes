import subprocess

runs = 20
xo_rate = [0.7, 0.0]
mut_rate = [0.15, 0.015,0.0015]

for xo in xo_rate:
    for mut in mut_rate:
        name =("xo" + str(xo).replace('.','')
               + "mut" + str(mut).replace('.',''))
        for i in range(runs):
            full_name = name + "run" + str(i) + ".dat" 
            cmd = ("python tetEvolver.py -x " + str(xo) 
                   + " -m " + str(mut) + " -l " 
                   + name + '/' + full_name)
            print cmd
            process = subprocess.Popen(cmd, shell=True,
                                       stdout=subprocess.PIPE,
                                       stdin=subprocess.PIPE)
            process.communicate()
