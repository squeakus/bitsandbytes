import subprocess

max_images = 32

def execute_command(cmd, wait=True):
   process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
			      stdin=subprocess.PIPE)
   if wait:
	process.communicate()

cmd_list = []

for img in range(1,max_images):

    old_cnt = "%04d" % (max_images-img)
    new_cnt = "%04d" % (max_images+(img-1))
    
    old_name = "img"+str(old_cnt)+".jpg"
    new_name = "img"+str(new_cnt)+".jpg"

    cmd = "cp " + old_name + " " + new_name
    cmd_list.append(cmd)

for cmd in cmd_list:
    print cmd
    execute_command(cmd)
