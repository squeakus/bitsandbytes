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

HOST = ''
PORT = 9000
LOCADDR = (HOST, PORT)
SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Create a UDP socket
TELLO_ADDRESS = ('192.168.10.1', 8889)
SOCK.bind(LOCADDR)

def main():
    command('command', 1)
    batlevel = int(command('battery?', 1))
    if batlevel < 45:
        print("battery level too low")
        exit()

    command('takeoff', 5)
    command('forward 100', 3)
    command('back 200', 4)
    command('forward 100', 3)
    command('left 100', 3)
    command('right 200', 4)
    command('left 100', 3)
    command('flip r', 2)
    command('flip f', 2)
    command('flip b', 2)
    command('flip l', 2)
    command('cw 360', 5)
    command('land', 5)
    print('Closing connection...')
    SOCK.close()



def recv():
    try:
        data, server = SOCK.recvfrom(1518)
        response = data.decode(encoding="utf-8")
        return response
    except Exception as error:
        print("Error:", repr(error))
        sock.sendto('land'.encode(encoding="utf-8"), TELLO_ADDRESS)
        print('Landing . . .')

def command(command, delay=5):
    try:
        sent = SOCK.sendto(command.encode(encoding="utf-8"), TELLO_ADDRESS)
        response = recv()
        print(command, response)
        time.sleep(delay)

        if response == "ERROR":
            SOCK.sendto('land'.encode(encoding="utf-8"), TELLO_ADDRESS)
        if response == "OUT OF RANGE":
            print("Input value too low or too high")

        return response
    except KeyboardInterrupt:
        print ('Keyboard Interrupt. . .\n')
        SOCK.close()



print ('\r\n\r\nTello Flip Demo.\r\n')
instructions = ['command', 'takeoff', 'flip l', 'flip r', 'flip f', 'flip b',
                'up', 'down', 'left', 'right', 'forward', 'back', 'cw', 'ccw',
                'land', 'battery?', 'speed?', 'time?']


if __name__ == "__main__":
    main()
