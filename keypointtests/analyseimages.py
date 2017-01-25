import sys, ast, math
from tabulate import tabulate

detectors = ['sift', 'surf', 'orb', 'akaze', 'brisk']

def main():
    detector_results ={}
    basename = sys.argv[1]

    for det in detectors:
        print det
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
    headers = ['time','keypoints','matches(avr)','+-','inliers','+-', 'percentage(avr)', '+-', 'best percentage']
    
    for key in detector_results[detectors[0]]:
        print ""
        stats = []
        for det in detectors:
            stat = process_image(detector_results[det][key], det)
            stats.append(stat)
        
        print tabulate(stats, [key]+headers)

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
    
    return [det, time, keypoints, maavr,mastd,inavr,instd,peravr, perstd,max(percentages)]

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
