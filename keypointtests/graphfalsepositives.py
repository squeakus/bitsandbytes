"""Graphs the times, inliers, matches and keypoints as boxplots."""
import glob
import sys
import os
import ast
import math
import matplotlib.pyplot as plt


def main():
    """Collate stats from each image and then graph together."""
    if not os.path.exists('graphs'):
        os.makedirs('graphs')

    detectors = ['sift', 'surf', 'orb', 'akaze', 'brisk']
    detector_results = {}

    datfiles = glob.glob("*sift*.txt")
    #datfiles = ["richviewnadir30"]
    for datfile in datfiles:
        basename = datfile.split('sift')[0]
        print "\nresult for ", basename

        for det in detectors:
            filename = basename+det+"result.txt"
            res = read_result(filename)
            detector_results[det] = res

        stats = process_result(detector_results, detectors)
        boxplot_data(stats, basename)


def read_result(filename):
    """Evaluate dict string and return dictionary."""
    results = {}
    infile = open(filename, 'r')
    line = infile.readline()

    while line:
        line = line.rstrip()
        if line.startswith('#'):
            keyname = line.lstrip('#')
            dictinfo = infile.readline().rstrip()
            results[keyname] = ast.literal_eval(dictinfo)
        line = infile.readline()

    infile.close()
    return results


def process_result(detector_results, detectors):
    """Each detector gets a dictionary of the results for every image."""
    detstats = {}
    for det in detectors:
        stats = {'times': [], 'keypoints': [], 'matches': [],
                 'inliers': [], 'percentages': []}
        alldist = []
        for imgname in detector_results[detectors[0]]:
            alldist.extend(get_median_dist(detector_results[det][imgname]))
        alldist.sort()
        medindex = int(len(alldist) * 0.9)
        dist = alldist[medindex]
        # print "min", alldist[0], "max", alldist[-1], "med", alldist[medindex]

        for imgname in detector_results[detectors[0]]:
            result = process_image(detector_results[det][imgname], det, dist)
            times, keypoints, matches, inliers, percentages = result
            stats['times'].append(times)
            stats['keypoints'].append(keypoints)
            stats['matches'].extend(matches)
            stats['inliers'].extend(inliers)
            stats['percentages'].extend(percentages)
        detstats[det] = stats
    return detstats

def get_median_dist(image):
    distances = image['distance']
    return distances

def process_image(image, det, distance):
    """There is a single time but multiple inliers and matches."""
    time = image['time']
    keypoints = image['keypoints']
    distances = image['distance']

    matches = []
    inliers = []

    for i in range(len(distances)):
        if distances[i] > distance:
            matches.append(image['matches'][i])
            inliers.append(image['inliers'][i])

    percentages = []
    for a, b in zip(inliers, matches):
        if b != 0:
            percentages.append(int(((float(a)/float(b))*100)))
        else:
            percentages.append(0)

    return (time, keypoints, matches, inliers, percentages)


def boxplot_data(stats, filename):
    """Generate boxplot containing all the detectors."""
    results = ['times', 'keypoints', 'matches', 'inliers', 'percentages']
    detectors = ['sift', 'surf', 'orb', 'akaze', 'brisk']
    fig = plt.gcf()
    fig.set_size_inches(4, 3, forward=True)

    for result in results:
        resultlist = []
        for det in detectors:
            resultlist.append(stats[det][result])

        graphname = "graphs/" + filename + result + "false.png"
        plt.clf()
        plt.figure(1)
        print result, "has", len(resultlist[0]), "items"
        if result in ['inliers', 'matches', 'percentages']:
            if result == "percentages":
                plt.boxplot(resultlist)
            else:
                plt.boxplot(resultlist, sym="")

       		if result == 'matches':
            	plt.title('False Matches')
        	if result == 'inliers':
            	plt.title('False Inliers')
        	if result == 'percentages':
            	plt.title('% False Inliers vs False Matches')

            plt.xticks([1, 2, 3, 4, 5], detectors)
            plt.savefig(graphname)


def ave(values):
    """Calc average of a list."""
    return float(sum(values)) / len(values)


def std(values, ave):
    """Calc standard deviation of a list."""
    return math.sqrt(float(sum((value - ave) ** 2
                     for value in values)) / len(values))

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print ("Usage %s" % sys.argv[0])
        sys.exit(1)
    main()
