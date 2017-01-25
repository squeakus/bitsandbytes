import elementtree.ElementTree as ET
tree = ET.parse('plane.vsp')
root = tree.getroot()
sections = []
airfoils = []
variables = ['Span', 'TC', 'RC', 'Sweep']

for elem in root.getiterator():
    if elem.tag == 'Section':
        print "Found section"
        sections.append(elem)
    if elem.tag == 'Airfoil':
        print "Found airfoil"
        airfoils.append(elem)
        
sect0 = sections[0]
sect1 = sections[1]

for foil in airfoils:
    print "\n"
    for elem in foil.getchildren():
        print elem.tag, elem.text
