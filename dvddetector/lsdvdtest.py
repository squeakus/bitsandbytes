import subprocess

cmd = "lsdvd -Oy"
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)

dvddata = process.communicate()
if dvddata[0] == '':
    print "Cannot find DVD!"
else:
    exec(dvddata[0])
    print lsdvd['title']
    for track in lsdvd['track']:
        print "id:", track['ix'],"length",track['length']
        if track['length'] < 500:
            print "very short"
