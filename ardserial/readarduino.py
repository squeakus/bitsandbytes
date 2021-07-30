import serial
import serial.tools.list_ports
from time import sleep

ports = list(serial.tools.list_ports.comports())
for p in ports:
    print(p)
    sleep(.1)
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

while True:
    arduino = ser.readline().decode("utf-8").strip()
    print(arduino)
    