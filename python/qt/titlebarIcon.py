import sys
from PyQt4 import QtGui
#add thingy to title bar

class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()
        
    def initUI(self):
        self.setGeometry(300, 300, 250, 150)
        self.setWindowTitle('Hello!')
        self.setWindowIcon(QtGui.QIcon('web.png'))        
        self.show()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
