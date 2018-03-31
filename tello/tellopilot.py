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
    print ('\r\n\r\nTello API Demo.\r\n')
    instructions = ['command', 'takeoff', 'flip l', 'flip r', 'flip f', 'flip b',
                    'land', 'battery?', 'speed?', 'time?']

    command('command', 1)
    batlevel = int(command('battery?', 1))
    if batlevel < 40:
        print("battery level too low")
        exit()
    command('takeoff', 4)
    command('flip r', 2)
    command('flip f', 2)
    command('flip b', 2)
    command('flip l', 2)
    command('forward 100')
    command('back 200')
    command('forward 100')
    command('left 100')
    command('right 200')
    command('left 100')
    command('cw 360')
    command('land')
    print('Closing connection...')
    SOCK.close()

def recv():
    try:
        data, server = SOCK.recvfrom(1518)
        response = data.decode(encoding="utf-8")
        return response
    except Exception as error:
        print("Error:", repr(error))
        SOCK.sendto('land'.encode(encoding="utf-8"), TELLO_ADDRESS)
        print('Landing . . .')

def command(command, delay=5):
    directions = ['up', 'down', 'left', 'right', 'forward', 'back', 'cw', 'ccw']
    wait = True
    for direction in directions:
        if command.startswith(direction):
            wait = False 
    try:
        sent = SOCK.sendto(command.encode(encoding="utf-8"), TELLO_ADDRESS)
        response = recv()
        print(command, response)
        if wait:
            time.sleep(delay)

        if response == "ERROR" or response =="error":
            print("Something has gone wrong, landing")
            SOCK.sendto('land'.encode(encoding="utf-8"), TELLO_ADDRESS)
            recv()
        if response == "OUT OF RANGE":
            print("Input value too low or too high")

        return response
    except KeyboardInterrupt:
        print ('Keyboard Interrupt. . .\n')
        SOCK.close()


if __name__ == "__main__":
    main()
