import sys, ast, math

def main():
    results = {}
    filename = sys.argv[1]
    infile = open(filename, 'r')
    line = infile.readline()

    while line:
        line = line.rstrip()
        if line.startswith('#'):
            keyname = line.lstrip('#')
            dictinfo = infile.readline().rstrip()
            results[keyname]= ast.literal_eval(dictinfo)
        line = infile.readline()

    infile.close()
    times = []
    kps = []
    for key in results:
        print key

        times.append(results[key]['time'])
        kps.append(results[key]['keypoints'])
        inliers  = results[key]['inliers']
        matches = results[key]['matches']
        inavr = int(ave(inliers))
        instd = int(std(inliers, inavr))
        maavr = int(ave(matches))
        mastd = int(std(matches, maavr))
        print "inliers",inliers
        print "matches", matches

        percentages = []
        for a,b in zip(inliers, matches):
            if b != 0:
                percentages.append(int(((float(a)/float(b))*100)))
            else:
                percentages.append(0)
        peravr = round(ave(percentages),1)
        perstd = round(std(percentages, peravr),1)

        print "average matches:", maavr, "+-", mastd
        print "average inliers:", inavr, "+-", instd
        print "percentages", peravr, "+-", perstd
        print "best percentage=", max(percentages)
    timeavr = round(ave(times),3)
    timestd = round(std(times, timeavr),3)
    
    keyavr = int(ave(kps))
    keystd = int(std(kps, keyavr))
    print  "Mean time taken", timeavr, "+-", timestd
    print  "Mean no. of keypoints", keyavr, "+-", keystd



def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2
                     for value in values)) / len(values))



if __name__=='__main__':
    if len(sys.argv) < 2:
        print ("Usage %s <results file>" % sys.argv[0])
        sys.exit(1)
    main()
