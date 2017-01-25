"""Graphs the times, inliers, matches and keypoints as boxplots."""

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
    basename = sys.argv[1]

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

        for imgname in detector_results[detectors[0]]:
            result = process_image(detector_results[det][imgname], det)
            times, keypoints, matches, inliers, percentages = result
            stats['times'].append(times)
            stats['keypoints'].append(keypoints)
            stats['matches'].extend(matches)
            stats['inliers'].extend(inliers)
            stats['percentages'].extend(percentages)
        detstats[det] = stats
    return detstats


def process_image(image, det):
    """There is a single time but multiple inliers and matches."""
    time = image['time']
    keypoints = image['keypoints']
    matches = image['matches']
    inliers = image['inliers']

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

        print result, "has", len(resultlist[0]), "items"
        graphname = "graphs/" + filename + result + ".png"
        plt.clf()
        plt.figure(1)
        if result in ['inliers', 'matches']:
            plt.boxplot(resultlist, sym="")
        else:
            plt.boxplot(resultlist)
        if result == 'times':
            plt.title('Time Taken')
        if result == 'keypoints':
            plt.title('Keypoints Detected')
        if result == 'matches':
            plt.title('Matches')
        if result == 'inliers':
            plt.title('Inliers')
        if result == 'percentages':
            plt.title('% Inliers vs Matches')

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
    if len(sys.argv) < 2:
        print ("Usage %s <results file>" % sys.argv[0])
        sys.exit(1)
    main()
