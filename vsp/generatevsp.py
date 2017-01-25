import elementtree.ElementTree as ET
import random, subprocess

def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result


def mutate(section, section_ranges, tree):
    variables = ['Span', 'TC', 'RC', 'Sweep']

    for idx, var in enumerate(variables):
        for elem in section.getchildren():
            if elem.tag == var:
                minval, maxval = section_ranges[idx]
                print elem.tag, elem.text, minval, maxval
                newval = random.uniform(minval, maxval)
                elem.text = str(newval)
                print "new", elem.tag, elem.text


def generate_craft(sections, tree):
    sect0_ranges = [(3.0, 5.0), (4.0, 15.0), (11.0, 25.0), (45.0, 60.0)]
    sect1_ranges = [(3.0, 10.0), (0.0, 6.0), (7.0, 14.0), (20.0, 45.0) ]

    sect0 = sections[0]
    sect1 = sections[1]

    mutate(sect0, sect0_ranges, tree)
    mutate(sect1, sect1_ranges, tree)

    tree.write("plane.vsp")
    run_cmd("vsp -script exportstl.script")
    run_cmd("openscad image.scad --imgsize=500,500 -o plane.png")

def main():
    tree = ET.parse('plane.vsp')
    root = tree.getroot()
    sections = []

    for elem in root.getiterator():
        if elem.tag == "Section":
           sections.append(elem)
    generate_craft(sections, tree)

if __name__=='__main__':
    main()
    
