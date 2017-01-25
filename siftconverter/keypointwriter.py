import cv2, sys, os
import numpy as np
from multiprocessing.pool import ThreadPool
from keyutils import init_feature, find_images

def main():
    image_dir = sys.argv[1]
    detectstr = sys.argv[2]
    detector, matcher = init_feature(detectstr)
    processDirectory(image_dir, detector)

def analyse_image(detector, image, affine = False):
    img = cv2.imread(image)
    kp, desc = detector.detectAndCompute(img, None)

    #for idx,keypoint in enumerate(kp):
    #    print idx, keypoint.pt, keypoint.angle, keypoint.size
    #    print desc[idx]
    #    print ""
    return kp, desc

def normaliseandscale(desc):
    #print "before:", desc
    desc =  desc / np.linalg.norm(desc)
    #print "normed:", desc
    for idy, elem in enumerate(desc):
        if elem > 0.2:
            desc[idy] = 0.2
    #print "trunc:", desc

    desc =  desc / np.linalg.norm(desc)
    #print "normed again:", desc
    desc =  desc * 512
    for idy, elem in enumerate(desc):
        if elem > 255:
            desc[idy] = 255

    intdesc  = desc.astype(int)
    #print "OP", intdesc
    return intdesc

def write_detector(name,keypoints,desc):
    if len(desc[0]) < 128:
        print "descriptor is too short:", len(desc[0]), "padding to 128"
        newdesc = []
        for idx, ds in enumerate(desc):
            newarray = desc[idx].copy()
            newarray.resize(128)
            intarray = normaliseandscale(newarray)
            newdesc.append(intarray)

        desc = newdesc

    detectfile = open(name+".sift", 'wb')
    detectafile = open(name+".asift", 'wb')

    rows = str(len(keypoints))
    cols = str(128)

    detectfile.write(rows+" "+cols+" \n")
    detectafile.write(rows+" "+cols+" \n")
    i=0
    for keypoint in keypoints:
        kpt = "%f %f %f %f \n"%(keypoint.pt[1], keypoint.pt[0],keypoint.size, keypoint.angle)
        detectfile.write(kpt)
        detectafile.write(kpt)
        for d in desc[i]: detectfile.write("%d "%(d))
        detectfile.write("\n")

        for d in desc[i]: detectafile.write("%d "%(d))
        detectafile.write("\n")
        i+=1
    detectfile.close()
    detectafile.close()
    return

def processDirectory(directory, detector):
    '''
    find image files, create filenames and keypoint detector
    write sift for each image.
    '''
    img_files = find_images(directory)
    if not directory.endswith('/'):
        directory = directory + '/'

    for filename, full_file in img_files:
        print "processing image:", filename
        kps,desc = analyse_image(detector, full_file)
        write_detector(directory+filename,kps,desc)

if __name__=='__main__':
    if len(sys.argv) < 3:
        print ("Usage %s <image_dir> <keypoint_detector>" % sys.argv[0])
        sys.exit(1)
    main()
