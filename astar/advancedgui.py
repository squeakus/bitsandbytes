#!/usr/bin/env python

from os import system
import curses

def get_param(prompt_string):
	screen.clear()
	screen.border(0)
	screen.addstr(2, 2, prompt_string)
	screen.refresh()
	input = screen.getstr(10, 10, 60)
	return input

def execute_cmd(cmd_string):
	system("clear")
	a = system(cmd_string)
	print ""
	if a == 0:
		print "Command executed correctly"
	else:
		print "Command terminated with error"
	raw_input("Press enter")
	print ""

x = 0
gridmap = [[0 for x in xrange(5)] for x in xrange(5)]

while x != ord('4'):
    screen = curses.initscr()
    screen.clear()
    screen.border(0)
    screen.addstr(2, 2, "Please enter a number...")
    screen.addstr(4, 4, "1 - Add a user")
    screen.addstr(5, 4, "2 - Restart Apache")
    screen.addstr(6, 4, "3 - Show disk space")
    screen.addstr(7, 4, "4 - Exit")
    
    
    for idx,row in enumerate(gridmap):
        screen.addstr(idx+8, 4, str(row))
    screen.refresh()

    x = screen.getch()

    if x == ord('1'):
        username = get_param("Enter the username")
        homedir = get_param("Enter the home directory, eg /home/nate")
        groups = get_param("Enter comma-separated groups, eg adm,dialout,cdrom")
        shell = get_param("Enter the shell, eg /bin/bash:")
        curses.endwin()
        execute_cmd("useradd -d " + homedir + " -g 1000 -G " + groups + " -m -s " + shell + " " + username)
    if x == ord('2'):
        curses.endwin()
        execute_cmd("df")
    if x == ord('3'):
        curses.endwin()
        execute_cmd("df -h")
    if x == ord('5'):
        gridmap[4][4] = 1

curses.endwin()
