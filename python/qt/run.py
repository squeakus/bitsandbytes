#!/usr/bin/python -d
 
import sys
from PyQt4 import QtCore, QtGui
from gui import Ui_Form
 
class MyForm(QtGui.QMainWindow):
  def __init__(self, parent=None):
    QtGui.QWidget.__init__(self, parent)
    self.ui = Ui_Form()
    self.ui.setupUi(self)
    QtCore.QObject.connect(self.ui.pushButton, QtCore.SIGNAL("clicked()"), self.ui.textEdit.clear )
    QtCore.QObject.connect(self.ui.lineEdit, QtCore.SIGNAL("returnPressed()"), self.add_entry)
 
  def add_entry(self):
    self.ui.lineEdit.selectAll()
    self.ui.lineEdit.cut()
    self.ui.textEdit.append("")
    self.ui.textEdit.paste()
 
if __name__ == "__main__":
  app = QtGui.QApplication(sys.argv)
  myapp = MyForm()
  myapp.show()
  sys.exit(app.exec_())
