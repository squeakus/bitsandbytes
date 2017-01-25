import gudev, subprocess, urllib2, json, os, time
import Tkinter as tk

#cant use this as it sends it to root
#HOME_FOLDER = os.getenv("HOME") + "/Videos"
HOME_FOLDER = "/home/exclusive/Videos"

def write_log(text):
    print text
    outfile = open('/tmp/dvd.log','a')
    outfile.write(text+'\n')
    outfile.close()

def run_cmd(cmd):
    write_log(cmd)
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE, 
                               stdin=subprocess.PIPE) 
    result = process.communicate()
    return result

def check_for_dvd():
    """scan udev list for dvd drives and fund the name of the dvd"""
    dvdname = None
    client = gudev.Client(['block'])
    for device in client.query_by_subsystem("block"):
        #find device
        if device.has_property("ID_CDROM"):
            loginfo = "Found CD/DVD drive at %s" % device.get_device_file()
            write_log(loginfo)

            # check for content
            if device.has_property("ID_FS_LABEL"):
                loginfo = "Found disk: %s" % device.get_property("ID_FS_LABEL")
                dvdname = device.get_property("ID_FS_LABEL")
            elif device.has_property("ID_FS_TYPE"):
                loginfo = "Found disc but it has no name!"
            else:
                loginfo = "No disc"
            write_log(loginfo)
    return dvdname

def get_dvd_info():
    """only return tracks longer than 500 seconds"""
    dvddata = run_cmd("lsdvd -Oy")
    fulltracks = []
    if dvddata[0] == '':
        write_log("Cannot find DVD!")
    else:
        exec(dvddata[0])
        write_log(lsdvd['title'])
        for track in lsdvd['track']:
            if track['length'] > 500:
                loginfo = str(track['ix'])+'='+str(track['length'])
                write_log(loginfo)
                fulltracks.append(track['ix'])
    return fulltracks

def rip_tracks(dvdname, fulltracks):    
    ripfolder = HOME_FOLDER + '/' + dvdname + '/'
    write_log(ripfolder)

    #create the folders
    if not os.path.exists(HOME_FOLDER):
        os.mkdir(HOME_FOLDER)
    if not os.path.exists(ripfolder):
        os.mkdir(ripfolder)

    for idx, track in enumerate(fulltracks):
        info = ("ripping track " + str(idx+1)
                + " of " + str(len(fulltracks)+1))

        starting = "start:"+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        write_log(starting)
        
        cmd = 'mencoder dvd://' + str(track)      
        cmd = cmd + ' -dvd-device /dev/sr0  -alang English -info srcform="DVD ripped by Exclusive Audio"  -oac mp3lame -lameopts abr:br=128  -ovc copy -o "'
        cmd = cmd + ripfolder + dvdname + '-' + str(track) +'.avi"'
        run_cmd(cmd)
        
        ending = "end:"+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        write_log(ending)

def main():
    dvdname = check_for_dvd()
    if not dvdname == None:
        fulltracks = get_dvd_info()
        rip_tracks(dvdname, fulltracks)
        write_log("ejecting")
        run_cmd("eject")

if __name__ == "__main__":
    main()
