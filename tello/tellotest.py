#
# Tello Python3 Control Demo
#
# http://www.ryzerobotics.com/
#
# 1/1/2018

import threading
import socket
import sys
import time


host = ''
port = 9000
locaddr = (host,port)


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello_address = ('192.168.10.1', 8889)

sock.bind(locaddr)

def recv():
    try:
        data, server = sock.recvfrom(1518)
        response = data.decode(encoding="utf-8")
        if response == "ERROR":
            raise Exception("Low light or low battery")
        return response
    except Exception as error:
        print("Error:", repr(error))
        sock.sendto('land'.encode(encoding="utf-8"), tello_address)
        print('Landing . . .')

def command(command, delay=5):
    try:
        sent = sock.sendto(command.encode(encoding="utf-8"), tello_address)
        response = recv()
        print(command, response)
        time.sleep(delay)
        return response
    except KeyboardInterrupt:
        sock.sendto('land'.encode(encoding="utf-8"), tello_address)
        print ('\n . . .\n')
        sock.close()



print ('\r\n\r\nTello Flip Demo.\r\n')
instructions = ['command', 'takeoff', 'flip l', 'flip r', 'flip f', 'flip b',
                'up', 'down', 'left', 'right', 'forward', 'back', 'cw', 'ccw',
                'land', 'battery?', 'speed?', 'time?']


command('command', 1)
batlevel = command('battery?', 1)
print("batstring", type(batlevel))
# if batlevel < 45:
#     print("battery level too low")
#     exit()

command('takeoff', 5)
command('flip r', 2)
command('flip f', 2)
command('flip b', 2)
command('flip l', 2)
command('land', 5)
print('Closing connection...')
sock.close()
