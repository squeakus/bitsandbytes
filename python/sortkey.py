#!/usr/bin/env python
import dbus
import dbus.service
if getattr(dbus, 'version', (0,0,0)) >= (0,41,0):
    import dbus.glib
import gobject, sys, os, subprocess, time


def create_file(folder_name):
    time.sleep(1)
    cmd_list = []
    print "command: "+"rm -rf "+folder_name+"/*"
    cmd_list.append("rm -rf "+folder_name+"/*")
    cmd_list.append("touch " +folder_name+"/moo.txt")
    cmd_list.append("umount "+folder_name)
    for cmd in cmd_list:
        print  cmd
        process = subprocess.Popen(cmd, shell=True,
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()
        time.sleep(1)

class DeviceManager:
    def __init__(self):
        self.bus = dbus.SystemBus()
        self.bus.add_signal_receiver(self.device_added,
                                     'DeviceAdded',
                                     'org.freedesktop.Hal.Manager',
                                     'org.freedesktop.Hal')


        
    def udi_to_device(self, udi):
        return self.bus.get_object("org.freedesktop.Hal", udi)

    def device_added(self, udi):
        #print 'Added', udi
        result = str(udi)
        index = result.find("volume_uuid")
        if index > 0 :
            key_name = result[index+12:]
            key_name = key_name.replace("_","-")
            folder_name = "/media/"+key_name
            choice = raw_input('Do you want to format: '+folder_name+" [y/n]?")
            if choice == "y":
                create_file(folder_name)
            elif choice == "n":
                print "you chose NO!"

if __name__ == '__main__':
    m = DeviceManager()
    
    mainloop = gobject.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        mainloop.quit()
        print 'Exiting...'
        sys.exit(0)
