import datetime, subprocess


def generateinvoice(itemlist):
    # get the number
    transfile = open("transcount.dat",'r')
    transcount = int(transfile.readline())
    transstr = "%05d" % transcount
    transfile.close()
    print "processing transaction", transstr
    
    # generate a timestamp
    transtime = datetime.datetime.now().strftime("-%b-%d-%I%M%p-%Y")
    invname = "invoice/"+transstr+transtime
    invfile = open(invname+".tex", "w")

    invheader=open("invoice/invheader.dat",'r')
    for line in invheader:
        invfile.write(line)
    invheader.close()

    #write out items
    invfile.write("\opening{Customer Invoice no. "+str(transcount)+"}\n")
    invfile.write("\\begin{invoice}{Euro}{0}\n")
    invfile.write("\\ProjectTitle{Customer Receipt}\n")
    for item in itemlist:
        invfile.write("\Fee{"+item['name']
                      +"} {"+str(item['price'])
                      +"} {"+str(item['count'])+"}\n")

    invtail=open("invoice/invtail.dat",'r')
    for line in invtail:
        invfile.write(line)
    invtail.close()
    invfile.close()

    transfile = open("transcount.dat",'w')
    transfile.write(str(transcount+1))
    transfile.close()

    #commands to generate pdf
    commands = ["pdflatex "+invname]
    commands.append("del invoice\*.tex *.aux *.dvi *.log")


    for cmd in commands:
        process = subprocess.Popen(cmd, shell=True, 
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()

    
    transfile = open("transcount.txt",'w')
    transfile.write(str(transcount+1))
    transfile.close()

ITEMLIST = [{'name':"Dave",'price':10,'count':5},{'name':"Pete",'price':10,'count':5}]
generateinvoice(ITEMLIST)