#!/usr/bin/python
"""
In this example, we create a simple
window in PyQt4.
"""

import sys
from PyQt4 import QtGui

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    w = QtGui.QWidget()
    w.resize(250, 200)
    w.move(300, 300)
    w.setWindowTitle('Simple')
    w.show()
    sys.exit(app.exec_())
