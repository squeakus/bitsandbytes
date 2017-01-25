from BeautifulSoup import BeautifulSoup

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def scale(path_string):
    word_list = path_string.split(" ")
    new_string = ""
    for word in word_list:
        if is_number(word):
            scaled = (int(word)-500) / 15 
            new_string = new_string + str(scaled) + " "
        else:
            new_string = new_string + word + " "
    return new_string
            
            
svgfile = open('ireland.svg','r')
doc = ""
for line in svgfile:
    doc = doc + line

soup = BeautifulSoup(doc)
#print soup.prettify()
paths = soup.findAll('path')

out = open('paths.js', 'w')
out.write('var paths = {\n')


for path in paths:
    name, draw = path.get('id'), path.get('d')
    abbrev = name.replace(' ', '')
    abbrev = abbrev.replace('-', '')
    draw = draw.replace('\r', '')
    draw = draw.replace('\n', ' ')
    draw = draw.replace('    ', ' ')
    draw = draw.lstrip()

    scaled_draw = scale(draw)
    print abbrev, len(scaled_draw)
    out.write(abbrev+": {\n")
    out.write("name:'"+name+"',\n")
    out.write("path:'"+scaled_draw+"'")
    out.write('},\n')
out.write('}\n')
out.close()
