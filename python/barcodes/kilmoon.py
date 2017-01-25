import Tkinter as TK
import sys, re
from transaction import record


class GUI:
    def __init__(self, root, stock):
        self.root = root
        self.stock = stock
        self.barcode = []
        self.items = []
        self.amtstr = ""
        
        main_frame = TK.Frame(root)
        main_frame.pack(side=TK.LEFT)
        root.bind("<Key>", lambda event: self.keypress(event))

        #add frame for item list
        self.itemframe=TK.Label(main_frame,height=40, width=50, 
                           justify= TK.LEFT, background='white')
        self.itemframe.pack()

        # total at bottom
        self.totalframe=TK.Label(main_frame,height=2, width=20,
                                 font=("helvetica", 30),
                                 justify= TK.LEFT)
        self.totalframe.pack()
        self.totalframe['text'] = "Total = 0"

        # add the keypad_frame
        self.keypad_frame = TK.Frame(root)
        self.keypad_frame.pack(side=TK.RIGHT)
        self.keylist = []
        for i in range(10):
            self.keylist.append(TK.Button(self.keypad_frame, 
                                          font=("helvetica", 30),
                                          padx=28))
            self.keylist[i]["text"] = i
            self.keylist[i]["background"] = "green"
            self.keylist[i]["command"]= lambda x=i: self.keypad(x)
            rowidx = int(i/3)
            colidx = i % 3
            self.keylist[i].grid(row=rowidx, column=colidx)

        # This label shows the scan multiplier
        self.amtlabel = TK.Label(self.keypad_frame,
                                 font=("helvetica", 30),
                                 justify= TK.LEFT, background='white')
        self.amtlabel['text'] = "Amount: 1"
        self.amtlabel.grid(row=4, column=0,
                           sticky=TK.W+TK.E, columnspan=4)

        #set the amount back to 1
        self.amtbutton = TK.Button(self.keypad_frame, 
                                   font=("helvetica", 30),
                                   padx=19)
        self.amtbutton['text'] = "amt = 1"
        self.amtbutton['background'] = "green"
        self.amtbutton["command"]= lambda: self.resetamt()
        self.amtbutton.grid(row=3, column=1, columnspan=2)
        
        #undo the last item
        self.undobutton = TK.Button(self.keypad_frame,
                                    font=("helvetica", 30))
        self.undobutton['text'] = "Remove Item"
        self.undobutton['background'] = "red"
        self.undobutton["command"] = lambda: self.remove_item()
        self.undobutton.grid(row=5, column=0, pady=10,
                             sticky=TK.W+TK.E, columnspan=4)

        #end the transaction
        self.finishbutton = TK.Button(self.keypad_frame,
                                      font=("helvetica", 30))
        self.finishbutton['text'] = "Finish"
        self.finishbutton["command"] = lambda: self.complete()
        self.finishbutton.grid(row=6, column=0,
                               sticky=TK.W+TK.E, columnspan=4)
        
    def keypad(self,name):
        self.amtstr = self.amtstr + str(name)
        self.amtlabel['text'] = "Amount: "+self.amtstr

    def resetamt(self):
        self.amtstr = ""
        self.amtlabel['text'] = "Amount: 1"

    def remove_item(self):
        self.items[-1]['count'] -= 1
        if self.items[-1]['count'] == 0:
            self.items = self.items[:-1]
        self.update_itemframe()

    def add_item(self, barcodestr):
        #remove check digit:
        barcodestr = barcodestr[:-1]
        #search for the product
        current_item = ""
        for prod in self.stock:
            if barcodestr == prod['code']:
                current_item = prod
        if current_item == "":
            self.totalframe['text'] = "ERR: Item not Found!"
            return
        else:
            #get the multiplier
            count = 1
            if not self.amtstr == "":
                count = int(self.amtstr)
                self.amtstr = ""
                self.amtlabel['text'] = "Amount: 1"

            #check if already on the list
            onlist = False
            for item in self.items:
                if current_item['name'] == item['name']:
                    item['count'] += count
                    onlist = True

            if not onlist:
                newitem = {'name':current_item['name'],
                           'price':current_item['price'],
                           'count': count}
                self.items.append(newitem)
        self.update_itemframe()

    def complete(self):
        if len(self.items) < 1:
            print "nothing to process!"
        else:
            record(self.items)
            self.items = []
            self.update_itemframe()
    
    def update_itemframe(self):
        #write items to the menu frame
        itemstr = ""
        total = 0
        for item in self.items:
            total += item['price'] * item['count']
            name = item['name'].ljust(35)
            price = str(item['price']) + " euros"
            price = price.ljust(10)
            count = " X "+str(item['count'])
            count = count.ljust(5)
            itemstr += name + price + count + "\n"
        
        self.itemframe['text'] = itemstr
        self.totalframe['text'] = "Total = "+str(total)+ " Euros"
            
        
    def keypress(self, event):
        """Keypress listener, reads barcode reader and keyboard"""
        if re.match('^[\w-]+$', event.char):
            self.barcode.append(event.char)

        # return means a new barcode
        elif event.char == "\r": 
            barcodestr = ''.join(self.barcode)
            self.barcode = []
            self.add_item(barcodestr)

        # delete last item on the list
        elif event.char == '\b':
            self.remove_item()
        #quit if esc is pressed
        elif event.char == '\x1b': exit()

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
