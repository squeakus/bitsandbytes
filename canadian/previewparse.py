previewlist = open("JAN16_COF.txt",'r')

for line in previewlist:
    line = line.split()
    if len(line) > 0:
       if not line[0] == "PAGE":
       	   if len(line) < 5:
    	    print line
