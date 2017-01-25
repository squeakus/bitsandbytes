from pyudev import Context
context = Context()
#for device in context.list_devices(subsystem='input', ID_INPUT_MOUSE=True):
for device in context.list_devices():
    if device.sys_name.startswith('event'):
        print device.parent['NAME']
