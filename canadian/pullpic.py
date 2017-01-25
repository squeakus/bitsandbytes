import wget,sys

def main():
    if len(sys.argv) < 2:
        print "python pullpic <code>"
        exit()

    comicname  = sys.argv[1]
 
    print "looking for ", comicname

    itercomic = comicname[:-4]
    for i in range(1,195):
        print i
        tmpname = itercomic + "%04d" % i
        fullcomicname = "http://www.previewsworld.com/catalog/"+tmpname
        print "pulling"+fullcomicname

        wget.download(fullcomicname)

        pullpage(tmpname)

def pullpage(comicname):
    infile = open(comicname,'r')
    for line in infile:
        #print line
        if 'catalogimage' in line:
            if 'image' in line:
                line = line.split()
                for elem in line:
                    if elem.startswith('src'):
                        img = elem.lstrip("src=\"")
                        img = img.rstrip("\"")

                        img= "http://www.previewsworld.com" + img
                        print img
                        wget.download(img)

if __name__=='__main__':
    main()
