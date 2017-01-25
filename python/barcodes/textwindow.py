import Tkinter as TK
import sys, re

class GUI:
    def __init__(self, root, stock):
        self.root = root
        self.stock = stock

        self.barcode = []
        self.codes = []
        self.items = []
        
        item_frame = TK.Frame(root)
        item_frame.pack()
        root.bind("<Key>", lambda event: self.keypress(event))
        
        self.text=TK.Label(item_frame,height=40, width=50, 
                           justify= TK.LEFT, background='white')
        self.text.pack(side=TK.LEFT)
        item_frame.pack(side=TK.TOP)
        
    def keypress(self, event):
        # only append alphanumeric chars to the barcode
        if re.match('^[\w-]+$', event.char):
            self.barcode.append(event.char)

        # return means a new barcode
        if event.char == "\r": 
            barcodestr = ''.join(self.barcode)
            self.barcode = []
            self.codes.append(barcodestr+'\n')
            
            for prod in self.stock:
                if barcodestr == prod['code']:
                    result = prod['name'].ljust(40) + " " +str(prod['price'])
                    self.items.append(result+"\n")
            self.text['text'] = ''.join(self.items)

        # delete last item on the list
        if event.char == '\b':
            self.items = self.items[:-1]
            self.text['text'] = ''.join(self.items)
        #quit if esc is pressed
        if event.char == '\x1b': exit()

def main():
    #read the stocklist
    stockfile = open("stock.txt", 'r') 
    stock = []    
    for line in stockfile:
        item = eval(line)
        stock.append(item)
    stockfile.close()

    #create the gui
    ROOT = TK.Tk()
    shopgui = GUI(ROOT, stock)
    ROOT.geometry('+0+0')
    ROOT.title('Kilmoon Cross Nurseries')
    ROOT.mainloop()

if __name__ == '__main__':
    main()
