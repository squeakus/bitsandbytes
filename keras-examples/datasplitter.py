import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
import argparse

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--srcfolder",
    help="path to single set of images you want to split up")
    ap.add_argument("-d", "--datasetfolder",
    help="path to dataset containing multiple folders for splitting")
    ap.add_argument("-p", "--split", type=float, default=0.25,
                    help='the train / test split')
    
    # pull out src folders and create dest folders
    args = vars(ap.parse_args())
    src = args['srcfolder']
    dataset = args['datasetfolder']
    split = args['split']

    if not (os.path.isdir('train')):
        os.makedirs('train')
    if not (os.path.isdir('test')):
        os.makedirs('test')

    if src is not None:
        splitdir(src, split)
    
    elif dataset is not None:
        subfolders = [f.path for f in os.scandir(dataset) if f.is_dir()]
        for subfolder in subfolders:
            splitdir(subfolder, split)
    else:
        print("You need to provide a source folder or dataset folder!")
        exit()

def splitdir(src, split):
    src= os.path.join(src, '') # add trailing slash if missing
    X = y= os.listdir(src)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitls, random_state=0)
    label = src.split('/')[-2]
    if not (os.path.isdir('train/'+label)):
        os.makedirs('train/'+label)
    if not (os.path.isdir('test/'+label)):
        os.makedirs('test/'+label)
    for x in X_train:
        print("copying", src+x , 'train/'+label+'/'+x)
        copyfile(src+x , 'train/'+label+'/'+x)
    for x in X_test:
        print("copying", src+x , 'test/'+label+'/'+x)
        copyfile(src+x , 'test/'+label+'/'+x)

if __name__ == '__main__':
    main()
