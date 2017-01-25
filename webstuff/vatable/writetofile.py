import sys


def main():
    outfile = open("damn.txt", 'w')
    outfile.write("sometext")
    outfile.write("holy shit:" + sys.argv[1])
    outfile.close()
    return "ha!"

if __name__=='__main__':
    print main()
