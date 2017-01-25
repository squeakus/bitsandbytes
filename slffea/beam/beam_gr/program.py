# This program will make a list of nodes and beam co-ordinates to be used by
# the structural analysis program SLFFEA. The intention is to create two separate
# lists: one which will consist of a numbered list of individual nodes, and the
# other of which will consist of a list of bar connections from numbered (indexed)
# node to numbered node. The program will then save the list as a separate text
# file (extension .txt) for use by the program SLFFEA.

# What needs to be done:
#      Create a numbered list of nodes in the format: "n  x  y  z"
#      Create if loop: add nodes to the list only if those nodes do not appear
#             already on the list - DONE.
#      Create a list of beam connections which will go from one node number to
#             another. If two connectivity numbers on a single beam match up
#             (e.g. n1 = n2), don't add them to the list, or same if there already
#             exists an identical pair of nodes - DONE.
#             Format of connections list: "(element no.)  n1  n2"

# Need to find how to set up a list within the definition, as can only define things
# once at a time, and cannot recall defined variables outside of the definition... -> scope?!


# define our list of nodes and beams
global nodeslist
global beamslist
n = []
m = []
global fixedpoints
global udl
global loadelement

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

clear()

from math import sqrt
import os
import subprocess

print "\n\n\n\n\nThis program operates in kilonewton meters, please use correct units in your calculations.\n\n"
print "Type: beam(x1, y1, z1, x2, y2, z2) to get started...\n\n"

def runanalysis():
    a = subprocess.Popen('./bmpost < slfinput.txt > slfoutput.txt', shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    x = file("xxx.obm", 'r')
    print x
 	
# Now, our structure has been created and analyzed. Next, we need to retrieve and interperate these
# results.
    
	#x.split('\n')
	


def test():
    beam(0, 0, 0, 2, 0, 0)
    #beam(2, 0, 0, 4, 0, 0)
    #beam(0, 0, 0, 2, 2, 0)
    #beam(2, 2, 0, 4, 0, 0)
    #beam(2, 2, 0, 2, 0, 0)

def writetofile():

    global beamslist
    global nodeslist
    global fixedpoints
    global udl
    global loadelement

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
    testbeams.write('0   10000000.0000    0.0000 5.3    0.02000    0.000016666666    0.00006666666\n')
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
    testbeams.write('     3  0.0   10.00   0.0    0.0  0.0   0.0\n')
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


       
       writetofile()
       runanalysis()
       



