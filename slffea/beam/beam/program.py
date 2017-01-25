# This program will make a list of nodes and beam co-ordinates to be used by
# the structural analysis program SLFFEA. The intention is to create two separate
# lists: one which will consist of a numbered list of individual nodes, and the
# other of which will consist of a list of bar connections from numbered (indexed)
# node to numbered node. The program will then save the list as a separate text
# file for use by the program SLFFEA.



# define our list of nodes and beams, along with other global variables
global nodeslist
global beamslist
n = []
m = []
global fixedpoints
global udl
global loadelement
global stresslist
global strainlist
global momentslist
global materials


# Here is where we define the maximum permitted stresses, strains and moments
# for our material.

global maxtension
maxtension = (1.8*10**4)		#tension
global maxcompression
maxcompression = (2.3*10**4)	#compression
global maxmoment
maxmoment = (3*10**4)

# We define a clear function to eusure we are starting from scratch; no values
# are retained from any previous runs of the program, it is cleared every time.

def clear():
	global beamslist
	beamslist = []
	global nodeslist
	nodeslist = []
	global fixedpoints
	fixedpoints = []
	global udl
	udl = []
	global loadelement
	loadelement = []
	global stresslist	
	stresslist = []
	global strainlist
	strainlist = []
	global momentslist
	momentslist = []


clear()			# We clear the memory

from math import sqrt
import os
import subprocess
import re

# The program begins...

print "\n\n\n\n\nThis program operates in kilonewton meters, please use correct units in your calculations.\n\n"
print "Type: beam(x1, y1, z1, x2, y2, z2) to get started...\n\n"

def runanalysis():

# This will run the analysis program for us (behind the scenes), and will create a file
# with an extension (.obm) which will contain all the data recorded from the analysis. 
    
    a = subprocess.Popen('./bm < slfinput.txt > slfoutput.txt', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    x = file("slfoutput.txt").read()
    print x

def showanalysis():
	a = subprocess.Popen('./bmpost < trussimput.txt > trussoutput.txt', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE) 

def U():
	
	global materials
	
	print "Please select a material from the following list:\n\nTimber (Class D30 Hardwood)\n\nSteel\n\n"
	material = raw_input ("Type selection in here: ")
#	if material == "Timber":
#		print "\nPlease specify a section size (in millimeters):\n"#
#		width = raw_input("Width: ")
#		height = raw_input("Height: ")
#		area = (float(width))*(float(height))*10.0**(-6)
#		print "area = ", area
#		iz = (((float(width))*(float(height)**3))/12)*10**(-12)
#		print "iz = ", iz
#		iy = (((float(height))*(float(width)**3))/12)*10**(-12)
#		print "iy = ", iy
#		materials = '0   10000000.0000    0.0000   530    0.02    1.6666666666666664e-05    6.6666666666666656e-05\n'
	if material == "Steel":
		print "\nPlease specify a section size (in format ___x___x__, eg. 254x146x43):\n"
		size = raw_input("Section Size: ")
		sections = file("SteelTables.txt", "r")
		line = sections.readline()
		while line:
			if line.startswith(size): 				# searches for the specified section size
				print "found the section size"
				everything = line.split()
				area = float(everything[8]) * 0.0001
				iy = float(everything[10]) * 10**(-8)
				iz = float(everything[9]) * 10**(-8)
				print str(line)
				materials = '0   10000000.0000    0.0000   7850    ', area, '    ', iy, '    ', iz, '\n'
				print materials
				break
#			elif line.startswith(size) not in sections:				# This should give the alternative, but seems to over-ride the original
#				print "could not find specified section size"		# option every time...
			line = sections.readline()	
	else:
		print "That is not a valid material"
	
	

def I():

# This will search the analysis results for the relevant data: stresses, strains and moments.

	global stresslist
	global strainlist
	global momentslist
	global maxtension
	global maxcompression
	global maxmoment
	
	x = file('xxx.obm', 'r')		# opens the results file	
	line = x.readline()
	while line:
		if line.startswith("element no. and nodal pt. no. with local stress xx,xy,zx and moment xx,yy,zz"): # searches for the stresses
			print "found the stresses"
			line = x.readline()
			while line != ('-10'):			
				if line.startswith("   "):		
					stresses = line.split()
					a = stresses[0:2]
					b = stresses[2:5]
					d = a + b
					stresslist.append(d)		# Adds the stresses to the stress list
					c = stresses[5:]
					e = a + c
					momentslist.append(e)		# Adds the moments to the moments list
					line = x.readline()			# Moves on to read the next line
				else:
					break

		if line.startswith("element no. and nodal pt. no. with local strain xx,xy,zx and curvature xx,yy,zz"):	# searches for the strains
			print "found the strains"
			line = x.readline()
			while line != ('-10'):			
				if line.startswith("   "):		
					strains = line.split()
					b = strains[:5]
					strainlist.append(b)		# Adds the strains to the strains list			
					line = x.readline()
				else:
					break

		line = x.readline()			

	print "\n\nStresslist:\n"
	for i, n in enumerate(stresslist):
		print n
		for x in range(2, 5):
			if float(stresslist[i][x]) > 0:		
				if abs(float(stresslist[i][x])) > maxtension:				# checks the xx stresses against the max tension			
					print "Element ", stresslist[i][0], "fails in tension."
			if float(stresslist[i][x]) < 0:		
				if abs(float(stresslist[i][x])) > maxcompression:				# checks the xx stresses against the max compression			
					print "Element ", stresslist[i][0], "fails in compression."	

#	print "\n\nStrainlist:\n"
#	for i, n in enumerate(strainlist):
#		print n
#		if abs(float(strainlist[i][2])) > maxstrain:				# checks the xx strains against the max strain
#			print "Element ", stresslist[i][0], "fails in strain."	
#		if abs(float(strainlist[i][3])) > maxstrain:				# checks the xy strains against the max strain
#			print "Element ", stresslist[i][0], "fails in strain."	
#		if abs(float(strainlist[i][4])) > maxstrain:				# checks the zx strains against the max strain
#			print "Element ", stresslist[i][0], "fails in strain."
#
#	print "\n\nMomentslist:\n"	
#	for i, n in enumerate(momentslist):
#		print n
#		if abs(float(momentslist[i][2])) > maxmoment:				# checks the xx moments against the max moment
#			print "Element ", stresslist[i][0], "fails in bending."	
#		if abs(float(momentslist[i][3])) > maxmoment:				# checks the yy moments against the max moment
#			print "Element ", stresslist[i][0], "fails in bending."	
#		if abs(float(momentslist[i][4])) > maxmoment:				# checks the zz moments against the max moment
#			print "Element ", stresslist[i][0], "fails in bending."

def test():

# This is just a test program that creates a small truss, it is here purely to save
# time when entering the same data into python

	beam(0, 0, 0, 2, 0, 0)    
	beam(2, 0, 0, 4, 0, 0)
	beam(0, 0, 0, 2, 2, 0)
	beam(2, 2, 0, 4, 0, 0)
	beam(2, 2, 0, 2, 0, 0)


def writetofile():

# This will create the input file for use in the SLFFEA analysis program

    global beamslist
    global nodeslist
    global fixedpoints
    global udl
    global loadelement
    global materials

    n = len(nodeslist)
    m = len(beamslist)
    f = len(fixedpoints)
    l = len(loadelement)

# This module:
#    xxx = raw_input('Please enter a file name: ')
# will allow the user to specify the file name for their work. Just delete the hash symbol
# and delete the inverted commas around every appearance of xxx.

    testbeams = file('xxx', 'w')
    testbeams.write('   numel numnp nmat nmode  (This is for a beam bridge)\n')
    testbeams.write('     ')
    testbeams.write(str(m))
    testbeams.write('   ')
    testbeams.write(str(n))
    testbeams.write('  1   0\n')
    testbeams.write('matl no., E mod, Poiss. Ratio, density, Area, Iy, Iz\n')
    testbeams.write('0   10000000.0000    0.0000   530    0.02    1.6666666666666664e-05    6.6666666666666656e-05\n')
    testbeams.write('el no.,connectivity, matl no, element type\n')
    for i, beam in enumerate(beamslist):
        testbeams.write(str(i) + '        ' + str(beam[0]) + '    ' + str(beam[1]) + '    ' + '0' + '     ' + '2' + '\n')
    testbeams.write('\n')
    testbeams.write('node no., coordinates\n')
    for ii, node in enumerate(nodeslist):

# There is a problem here in that the string "i" has now been defined as the last number of elements
# from the previous list, so if there are three elements (listed zero to two), every node will be
# numbered as "2" rather than zero to two.

        testbeams.write(str(ii)+ '     ' + str(node[0]) + '   ' + str(node[1]) + '   ' + str(node[2]) + '\n')
    testbeams.write('\n')
    testbeams.write('element with specified local z axis: x, y, z component\n')
    testbeams.write('-10')

# Here we have to add in the fixing points. It's probably easiest to make them the four corners
# of the bridge...

    testbeams.write('\nprescribed displacement x: node  disp value\n')
    for w in range(f):
        testbeams.write('    ' + str(fixedpoints[w]) + ' 0.0\n')
    testbeams.write('-10')
    testbeams.write('\nprescribed displacement y: node  disp value\n')
    for w in range(f):
        testbeams.write('    ' + str(fixedpoints[w]) + ' 0.0\n')
    testbeams.write('-10')
    testbeams.write('\nprescribed displacement z: node  disp value\n')
    for w in range(f):
        testbeams.write('    ' + str(fixedpoints[w]) + ' 0.0\n')
    testbeams.write('-10')
    testbeams.write('\nprescribed angle phi x: node angle value\n')
    for w in range(f):
        testbeams.write('    ' + str(fixedpoints[w]) + ' 0.0\n')
    testbeams.write('-10')
    testbeams.write('\nprescribed angle phi y: node angle value\n')
    for w in range(f):
        testbeams.write('    ' + str(fixedpoints[w]) + ' 0.0\n')
    testbeams.write('-10')
    testbeams.write('\nprescribed angle phi z: node angle value\n')
    for w in range(f):
        testbeams.write('    ' + str(fixedpoints[w]) + ' 0.0\n')

# Here we have to add in our loads. Again, it's probably best if we only add them on the deck
# of the bridge...

    testbeams.write('-10')
    testbeams.write('\nnode with point load x, y, z and 3 moments phi x, phi y, phi z\n')
#    testbeams.write('     3  0.0   0.00   0.0    0.0  0.0   0.0\n')
    testbeams.write('-10')
    testbeams.write('\nelement with distributed load in local beam y and z coordinates\n')
    for w in range(l):
        testbeams.write('     ' + str(loadelement[w]) + '   -' + str(udl) + '   0.0\n')
    testbeams.write('-10')
    testbeams.write('\nelement no. and gauss pt. no. with local stress vector xx and moment xx,yy,zz\n')
    testbeams.write('-10')
    testbeams.close()


    print "\nA file has been created for you in the directory:", os.getcwd(), '\n'




def beam(x0, y0, z0, x1, y1, z1):

    global beamslist
    global nodeslist
    global fixedpoints
    global udl
    global loadelement

# This should limit the length of any individual beam to 4.9m to avoid slenderness...

    beamlength = float(sqrt(((abs(x1 - x0))**2) + ((abs(y1 - y0))**2) + ((abs(z1 - z0))**2)))
    print "\nThe length of your beam is", beamlength, "meters\n"

    if beamlength > 4.900:
        print "\nYOU CANNOT HAVE A BEAM LONGER THAN 4.9m AS IT IS TOO SLENDER!!!\n"

    elif beamlength < 4.900:

        a = (x0, y0, z0)
        b = (x1, y1, z1)

# ensure we aren't repeating nodes, and then add nodes to our list
        if a not in nodeslist:
           nodeslist.append(a)

        if b not in nodeslist:
           nodeslist.append(b)

# ensure beams have 1 dimension
        if a != b:
           print "\nnice beam"
           print "\nyour beam goes from", a, "to", b

        if a == b:
           print "\nyou can't have a beam that starts and ends in the same place"
           print "\ntry another one"

# the number of nodes so far in the structure
        n = len(nodeslist)

# here we add the beam connectivity to the beams list
        if a != b:
           beamstuple = (nodeslist.index(a), nodeslist.index(b))

           if beamstuple not in beamslist:
              beamslist.append(beamstuple)

# the number of beams so far in the structure
        m = len(beamslist)

        print "\nnode no., coordinates"  
        for i, v in enumerate(nodeslist):
            print i, v

# Here i add the new nodes to the list of nodes. I have to find a way to do this
# without just tacking them on at the end, but over-writing the whole list every time...

        if n == 1:
            print "\nthere is", n, "node so far"

        else:
            print "\nthere are", n, "nodes so far\n"
            print "\nel no., connectivity"
            for q, r in enumerate(beamslist):
                print q, r

        if m == 1:
            print "\nthere is", m, "beam so far\n"

        else:
            print "\nthere are", m, "beams so far\n"

# The next step is to ask the user to specify fixities, i.e. nodes that will act as fixing points
# in given directions and rotations. This will be an automated step in GEVA.

    answer = raw_input('Do you want to create a new beam? (answer yes or no): ')
    if answer == 'no':

       fixedpoints = input("Please select your fixing points (enter node numbers): ")
       f = len(fixedpoints)
       for w in range(f):
           print "Fixed point ", w, " = ", fixedpoints[w], '\n'

       udl = input("Please select a uniform distributed load (in kN/m): ")
       print "UDL is ", udl, "kN/m\n"
       
       loadelement = input("Please select the elements upon which to impose this load: ")
       if len(loadelement) > 1:
           print "Elements ", loadelement, " will have a load of ", udl, " imposed on them."
       else:
           print "Element number ", loadelement, " will have a load of ", udl, " imposed on it."


       U()					# Runs the "Material Select" program
       writetofile()		# Writes the output to a file
       runanalysis()		# Runs the SLFFEA analysis
       I()					# Reads relevant data from analysis results
       showanalysis()		# Displays the results of the analysis





