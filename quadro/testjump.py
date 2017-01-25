import logging
import time
 
import cflib.crtp
from cfclient.utils.logconfigreader import LogConfig
from cfclient.utils.logconfigreader import LogVariable
from cflib.crazyflie import Crazyflie

def main():
    #show available links
    cflib.crtp.init_drivers()
    available = cflib.crtp.scan_interfaces()
    for i in available:
        print "Interface with URI [%s] found and name/comment [%s]" % (i[0], i[1])

    crazyflie = Crazyflie()
    crazyflie.open_link("radio://0/10/250K")
    roll    = 0.0
    pitch   = 0.0
    yawrate = 0
    thrust  = 35001
    crazyflie.commander.send_setpoint(roll, pitch, yawrate, thrust)

    time.sleep(5)
    crazyflie.close_link()




if __name__=='__main__':
    main()
