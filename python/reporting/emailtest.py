import os, subprocess

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result
	
#cmd = "C:\reporting\blat311\full\Blat.exe C:\reporting\reporttest.txt -to jonathanbyrn@gmail.com"
cmd = "blat C:\\reporting\\reporttest.txt -to jonathanbyrn@gmail.com -subject \"sending report\""
run_cmd(cmd)