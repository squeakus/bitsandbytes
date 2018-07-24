import os
from sklearn.cross_validation import train_test_split
from shutil import copyfile
import argparse

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--srcfolder", required=True,
    help="path to input dataset")
    args = vars(ap.parse_args())
    src = args['srcfolder']
    if not (os.path.isdir('train')):
        os.makedirs('train')
    if not (os.path.isdir('test')):
        os.makedirs('test')


    X = y= os.listdir(src)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    print(X_train)
    print(X_test)
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