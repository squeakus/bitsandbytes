forcefile = open('forcecoeffs1ts.dat')

for line in forcefile:
    if not line.startswith('#'):
        line = line.strip()
        info = line.split('\t')
        info = [float(strval) for strval in info]
        print "info!", info
        forces = {'time':info[0], 'pitchmoment':info[1],'drag':info[2], 
                  'lift':info[3], 'frontlift':info[4], 'rearlift':info[5]}

print forces
       
