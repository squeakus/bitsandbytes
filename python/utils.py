import subprocess

def main():
    print('a utility class of static functions')

def run_cmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

if __name__ == "__main__":
    main()
