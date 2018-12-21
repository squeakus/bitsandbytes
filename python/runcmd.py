import subprocess

def main():

def run_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result

if __name__ == "__main__":
    main()