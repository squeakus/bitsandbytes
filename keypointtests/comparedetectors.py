import cv2, sys, os, time, subprocess
import numpy as np
from math import sqrt
from keyutils import init_feature, find_images, filter_matches
from affinedetect import affine_detect
import itertools


def main():
    image_dir = sys.argv[1]
    image_dir = image_dir.rstrip('/')
    locations = get_locations(image_dir)
    detectstr = sys.argv[2]
    detector, matcher = init_feature(detectstr)
    results = processDirectory(image_dir, detector, matcher, locations)


    outfile = open(image_dir+detectstr+'result.txt','w')
    for line in results:
        outfile.write('#'+line+'\n')
        outfile.write(str(results[line])+'\n')
    outfile.close()

def affine_detection(img, detector):
    cpucount = 2
    pool = None
    print("No. of CPUs: %d" % cpucount)
    kp, desc = affine_detect(detector, img, pool=pool)
    return kp, desc

def get_locations(idir):
    locations = {}   
    cmd  = "exiftool -filename -gpslatitude -gpslongitude -n -T "\
           + idir + " > " + idir + "dist.txt"
    
    process = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)

    process.wait()
    gpsfile = open(idir+'dist.txt','r')
    
    for line in gpsfile:
        line = line.split()
        imgname = line[0].rstrip('.jpg')
        imgname = imgname.rstrip('.JPG')
        lat, lon = float(line[1]), float(line[2])
        locations[imgname] = [lat, lon]
        
    return locations

def calc_distance(aloc, bloc, locations):
    # this is what 1 degree is equivalent to at 53 degrees north (metres)
    onedeg = 65904
    alat, alon = locations[aloc][0], locations[aloc][1]
    blat, blon = locations[bloc][0], locations[bloc][1]
    dist = sqrt(((blat-alat)**2) + ((blon-alon)**2))
    metredist = onedeg * dist
    return metredist

def analyse_image(detector, image, affine=False):
    img = cv2.imread(image)

    if affine:
        print "using affine transforms!"
        kp, desc = affine_detection(img, detector)    
    else:
        kp, desc = detector.detectAndCompute(img, None)
    
    return kp, desc

def processDirectory(directory, detector, matcher, locations):
    '''
    find image files, create filenames and keypoint detector
    write sift for each image.
    '''
    results = {}
    keylist = []

    img_files = find_images(directory)
    directory = directory + '/'

    for filename, full_file in img_files:
        results[filename] = {'keypoints':[],'matches':[],
                             'inliers':[],'distance':[]}
        starttime = time.time()
        kps,desc = analyse_image(detector, full_file)
        timetaken = time.time() - starttime
        print "img %s has %d keypoints" % (filename, len(kps))
        keylist.append([kps,desc, filename])
        results[filename]['keypoints'] = len(kps)
        results[filename]['time'] = timetaken

    analyse_keypoints(keylist, matcher, results, locations)
    return results

def analyse_keypoints(keylist, matcher, results, locations):

    for a, b in itertools.combinations(keylist, 2):
        kp1, desc1, name1 = a[0], a[1], a[2]
        kp2, desc2, name2 = b[0], b[1], b[2]
        
        #work out the distance between images
        distance = calc_distance(name1, name2, locations)
        results[name1]['distance'].append(distance)
        results[name2]['distance'].append(distance)
        print('%s  %s - %d / %d' % (name1, name2, len(kp1), len(kp2)))

        starttime = time.time()
        raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2)
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        print "BFFMatching time taken:",time.time() - starttime


        if len(p1) >= 4:
            starttime = time.time()
            H, matchMask = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	    if matchMask == None:
		inliers = 0
		total = 0
		percent = 0
	    else:
                inliers = np.sum(matchMask)
	        print inliers
                total = len(matchMask)
                percent = round(inliers / total, 3)
            print('%d / %d inliers/matched= %f' % (inliers, total, percent ))
            
            results[name1]['matches'].append(total)
            results[name1]['inliers'].append(inliers)

            results[name2]['matches'].append(total)
            results[name2]['inliers'].append(inliers)
            print "RANSAC time taken:",time.time() - starttime
            
        else:
            results[name1]['matches'].append(0)
            results[name1]['inliers'].append(0)
            results[name2]['matches'].append(0)
            results[name2]['inliers'].append(0)
            H, matchMask = None, None
            print('%d matches found, not enough for homography estimation' % len(p1))
    

if __name__=='__main__':
    if len(sys.argv) < 3:
        print ("Usage %s <image_dir> <keypoint_detector>" % sys.argv[0])
        sys.exit(1)
    main()
