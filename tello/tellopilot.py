#
# Tello Python3 Control Demo
#
# http://www.ryzerobotics.com/
#
# 1/1/2018

"""
This code is a demo of the Tello drone API. It uses the threading, socket, sys and time libraries. The code creates a UDP socket and binds it to the local address and port 9000. It then sends commands to the Tello drone via the socket and receives responses. The main function first checks the battery level and exits if it is below 40. It then sends takeoff, flip and movement commands to the drone and finally sends a land command to land the drone. The recv function is used to receive the response from the drone after sending a command. The command function sends the command to the drone and also waits for a specified delay before sending the next command.
"""
import socket
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)

HOST = ""
PORT = 9000
LOCADDR = (HOST, PORT)
SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create a UDP socket
TELLO_ADDRESS = ("192.168.10.1", 8889)
SOCK.bind(LOCADDR)


def main():
    print("\r\n\r\nTello API Demo.\r\n")
    instructions = [
        "command",
        "takeoff",
        "flip l",
        "flip r",
        "flip f",
        "flip b",
        "land",
        "battery?",
        "speed?",
        "time?",
    ]

    command("command", 1)
    batlevel = int(command("battery?", 1))
    if batlevel < 40:
        print("battery level too low")
        exit()
    command("takeoff", 4)
    command("flip r", 3)
    command("flip f", 3)
    command("flip b", 3)
    command("flip l", 3)
    command("forward 100")
    command("back 200")
    command("forward 100")
    command("left 100")
    command("right 200")
    command("left 100")
    command("cw 360")
    command("land")
    print("Closing connection...")
    SOCK.close()


def recv():
    try:
        data, server = SOCK.recvfrom(1518)
        response = data.decode(encoding="utf-8")
        return response
    except Exception as error:
        print("Error:", repr(error))
        SOCK.sendto("land".encode(encoding="utf-8"), TELLO_ADDRESS)
        print("Landing . . .")


def command(command, delay=5):
    directions = ["up", "down", "left", "right", "forward", "back", "cw", "ccw"]
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

        if response == "ERROR" or response == "error":
            print("Something has gone wrong, landing")
            SOCK.sendto("land".encode(encoding="utf-8"), TELLO_ADDRESS)
            recv()
        if response == "OUT OF RANGE":
            print("Input value too low or too high")

        return response
    except KeyboardInterrupt:
        print("Keyboard Interrupt. . .\n")
        SOCK.close()


if __name__ == "__main__":
    main()
