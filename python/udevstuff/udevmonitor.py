import pyudev

context = pyudev.Context()
monitor = pyudev.Monitor.from_netlink(context)
monitor.filter_by(subsystem='usb')
for action, device in monitor:
    if action == 'add':
        print "*****************"
        print('{0!r} added'.format(device))
        print "sname:", device.sys_name
        print "spath:", device.sys_path
        print "dpath:", device.device_path
        for attrib in device.attributes:
            print attrib, device.attributes[attrib]
