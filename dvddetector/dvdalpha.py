import gudev, subprocess, urllib2, json, os, time
import Tkinter as tk

HOME_FOLDER = os.getenv("HOME") + "/videos"

class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        screen_width = self.winfo_screenwidth()
        self.geometry("%dx100+0+0" % (screen_width))
        print" width",screen_width
        self.label = tk.Label(self, 
                              text="",
                              height=2, width=50,
                              font=("helvetica", 30),
                              justify= tk.CENTER)
        self.label.pack()
        self.message("Searching for media")
        self.remaining = 0

    def message(self, text):
        self.label['text'] = text
        self.update()
        time.sleep(2)


def write_log(text):
    print text
    outfile = open('/tmp/dvd.log','a')
    outfile.write(text+'\n')
    outfile.close()

def run_cmd(cmd):
    write_log(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE) 
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

def rip_tracks(dvdname, fulltracks, display):
    
    ripfolder = HOME_FOLDER + '/' + dvdname + '/'
    write_log(ripfolder)
    if not os.path.exists(ripfolder):
        os.mkdir(ripfolder)

    
    for idx, track in enumerate(fulltracks):
        info = ("ripping track " + str(idx+1)
                + " of " + str(len(fulltracks)+1))
        display.message(info)
        starting = "start:"+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        write_log(starting)
        
        cmd = 'mencoder dvd://' + str(track)      
        cmd = cmd + ' -dvd-device /dev/sr0  -alang English -info srcform="DVD ripped by Exclusive Audio"  -oac mp3lame -lameopts abr:br=128  -ovc copy -o "'
        cmd = cmd + ripfolder + dvdname + '-' + str(track) +'.avi"'
        run_cmd(cmd)
        
        ending = "end:"+time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        write_log(ending)

def get_thumbnails(dvdname):
    """google search for thumbnails"""
    searchquery = "https://www.googleapis.com/customsearch/v1?key=AIzaSyBUts7RpJt7CvITxv1WP0NAlgMQJ6TjLwA&cx=004163519135800887416:2b9zuir4iuy&alt=json&q="+dvdname
    googleresult = urllib2.urlopen(searchquery)
    jsonresult = json.load(googleresult)
    
    #download thumbnail
    items = jsonresult['items']
    thumbs = []
    for item in items:
        if 'pagemap' in item:
            if 'cse_thumbnail' in item['pagemap']:
                thumbs.append(item['pagemap']['cse_thumbnail'][0]['src']);

    f = urllib2.urlopen(thumbs[0])

    # Open our local file for writing
    thumbname = HOME_FOLDER + '/' + dvdname + '/' + dvdname+".jpg"
    write_log(thumbname)
    
    with open(thumbname, "wb") as local_file:
        local_file.write(f.read())

def main():
    dvdname = check_for_dvd()
    if not dvdname == None:
        display = GUI()
        display.message("Found dvd-drive")
        fulltracks = get_dvd_info()
        display.message("Found media: "+dvdname)
        rip_tracks(dvdname, fulltracks, display)
        #display.message("Downloading thumbnail image for "+dvdname)
        #get_thumbnails(dvdname)
        display.message("ejecting")
        write_log("ejecting")
        run_cmd("eject")

if __name__ == "__main__":
    main()
