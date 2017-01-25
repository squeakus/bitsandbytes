import sys, ast, math, glob
from tabulate import tabulate

def main():
    detectors = ['sift', 'surf', 'orb', 'akaze', 'brisk']
    detector_results ={}
    datfiles =  glob.glob("*sift*.txt")
    
    for datfile in datfiles:
        basename = datfile.split('sift')[0]
        print "\nresult for ", basename
        for det in detectors:
            filename = basename+det+"result.txt"
            res = read_result(filename)
            detector_results[det] = res
            
        process_result(detector_results, detectors)
        
def read_result(filename):
    results = {}
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
    return results

def process_result(detector_results,detectors):
    headers = ['detector','time','keypoints','matches(avr)','+-','inliers','+-', 'percentage(avr)', '+-', 'best percentage']
    stats = []
    imgcnt = len(detector_results[detectors[0]])
    
    for det in detectors:
        sumstat = [0,0,0,0,0,0,0,0,0]
        totalkeypoints = 0
        for imgname in detector_results[detectors[0]]:
            stat = process_image(detector_results[det][imgname], det)
            sumstat = [x+y for x,y in zip(sumstat, stat)]
        sumstat = [x / imgcnt for x in sumstat]
        stats.append([det] + sumstat)
    print tabulate(stats, headers)

def process_image(image, det):
    time = image['time']
    keypoints  = image['keypoints']
    matches = image['matches']
    inliers  = image['inliers']
    
    inavr = int(ave(inliers))
    instd = int(std(inliers, inavr))
    maavr = int(ave(matches))
    mastd = int(std(matches, maavr))
    
    percentages = []
    for a,b in zip(inliers, matches):
        if b != 0:
            percentages.append(int(((float(a)/float(b))*100)))
        else:
            percentages.append(0)
    peravr = round(ave(percentages),1)
    perstd = round(std(percentages, peravr),1)
    
    return [time, keypoints, maavr,mastd,inavr,instd,peravr, perstd,max(percentages)]

def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2
                     for value in values)) / len(values))



if __name__=='__main__':
    if len(sys.argv) < 1:
        print ("Usage %s" % sys.argv[0])
        sys.exit(1)
    main()
