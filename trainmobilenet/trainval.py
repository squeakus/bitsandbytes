from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", help="Path to dataset", required=True, type=str)
    args = parser.parse_args()
    createTrainvalTxt(args.dataset_dir)


def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)
    
def createTrainvalTxt(baseDirDataSet):
    buffer =''
    baseDir = os.path.join(baseDirDataSet,'images')
    for filename in os.listdir(baseDir):
        filenameOnly, file_extension = os.path.splitext(filename)
        # print (file_extension)
        s = 'images/'+filenameOnly+'.jpg'+' '+'labels/'+filenameOnly+'.xml\n'
        print (repr(s))
        img_file, anno = s.strip("\n").split(" ")
        print(repr(img_file), repr(anno))
        buffer+=s
    

    outfolder = os.path.join(baseDirDataSet, 'structure')
    create_folder(outfolder)
    outfile = os.path.join(outfolder, 'trainval.txt')
    with open(outfile, 'w') as file:
        file.write(buffer)
    print('Done')

if __name__ == '__main__':
    main()