import sys
from PyQt4 import QtGui

class Example(QtGui.QMainWindow):
    
    def __init__(self):
        super(Example, self).__init__()
        self.initUI()
        
    def initUI(self):               
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)      
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Menubar')    
        self.show()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

