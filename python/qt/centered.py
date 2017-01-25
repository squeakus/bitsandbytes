import sys
from PyQt4 import QtGui


class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()
        
    def initUI(self):               
        self.resize(250, 150)
        self.center()
        self.setWindowTitle('Center')    
        self.show()
        
    def center(self):  
        #frame info
        qr = self.frameGeometry()
        #monitor info
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        #frame is always move by top left coord
        self.move(qr.topLeft())
            
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

