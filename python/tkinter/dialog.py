# ======== Select a directory:

# import Tkinter, tkFileDialog

# root = Tkinter.Tk()
# dirname = tkFileDialog.askdirectory(parent=root,initialdir="/home/jonathan/Jonathan/programs/architype",title='Please select a directory')
# if len(dirname ) > 0:
#     print "You chose %s" % dirname 


# ======== Select a file for opening:
import Tkinter,tkFileDialog

results = []
root = Tkinter.Tk()
dat_file = tkFileDialog.askopenfile(parent=root,filetypes=[("dat file","*.dat")],initialdir="/home/jonathan/Jonathan/programs/architype", mode='r',title='Choose a file')
if file != None:
    for line in dat_file:
        if not line.startswith('#'):
            line = line.rstrip()
            array = line.split(';')
            result = {'uid': array[0], 'fitness': array[1], 'genome': array[2]}
            results.append(result)
    for result in results:
        print "uid:",result['uid'], "fitness:", result['fitness'], "genome:", str(len(result['genome']))
    dat_file.close()


# # ======== "Save as" dialog:
# import Tkinter,tkFileDialog

# myFormats = [
#     ('Windows Bitmap','*.bmp'),
#     ('Portable Network Graphics','*.png'),
#     ('JPEG / JFIF','*.jpg'),
#     ('CompuServer GIF','*.gif'),
#     ]

# root = Tkinter.Tk()
# fileName = tkFileDialog.asksaveasfilename(parent=root,filetypes=myFormats ,title="Save the image as...")
# if len(fileName ) > 0:
#     print "Now saving under %s" % nomFichier
