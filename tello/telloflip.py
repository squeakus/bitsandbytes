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
        print(response)
        return response
    except Exception:
        print ('\nExit . . .\n')


print ('\r\n\r\nTello Flip Demo.\r\n')
instructions = ['command', 'takeoff', 'flip l', 'flip r',
                'flip f', 'flip b', 'land']

for instruction in instructions:
    try:
        # Send data
        msg = instruction.encode(encoding="utf-8")
        sent = sock.sendto(msg, tello_address)
        waiting  = True
        while waiting:
            response = recv()
            if response == "OK":
                print('command received')
                waiting = False
            else:
                print("No idea:", response)
        time.sleep(10)

    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()
        break

print ('Closing connection...')
sock.close()
