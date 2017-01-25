myfile = open('frames.txt','w')

for i in range (1,200):
    txt = 'render_'+str(i)+'.jpg'+'\n'
    myfile.write(txt)
myfile.close()
