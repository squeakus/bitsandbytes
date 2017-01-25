from glob import glob


explist = {'nadirbolands':[],'nadirglasshouse30':[],
           'nadirglasshouse50m':[],'nadirglasshouse70':[],'nadirrichview40':[],
           'nadirrichview50':[],'nadirrichview60':[], 'obliquerichview':[],
           'obliquebolands':[],'obliqueplatin':[],}

infile = open('globalpointcount.txt','r')
for line in infile:
    line = line.rstrip()
    line = line.replace('element vertex ', '')
    line = line.split(':')
    line[0] = line[0][:-len('colorized.ply')]
    if line[0].endswith('sequential'):
        line[0] = line[0][:-7]
    for exp in explist:
        if line[0].startswith(exp):
            line[0] = line[0][len(exp):]
            explist[exp].append([line[0], line[1]])

for exp in explist:
    print "\n",exp
    for item in explist[exp]:
        print item[0],'\t\t',item[1]
