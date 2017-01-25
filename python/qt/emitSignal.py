import sys
from PyQt4 import QtGui, QtCore


class Communicate(QtCore.QObject):
    # new signal called close app
    closeApp = QtCore.pyqtSignal() 
    
class Example(QtGui.QMainWindow):
    
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()
        
    def initUI(self):
        self.c = Communicate()
        self.c.closeApp.connect(self.close)       
        self.setGeometry(300, 300, 290, 150)
        self.setWindowTitle('Emit signal')
        self.show()
        
    #when we click on it the close app signal is emitted
    def mousePressEvent(self, event):
        self.c.closeApp.emit()
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
