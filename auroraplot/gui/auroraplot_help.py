#!/usr/bin/env python3

import sys
import platform
import os
workingdir = os.path.dirname(__file__)
help_file = ''.join([workingdir,'/../docs/AuroraplotGUI.html'])

import PySide
from PySide.QtGui import *
from PySide.QtCore import *

import resources_rc

class HelpWindow(QDialog):
    def __init__(self, parent=None):
        super(HelpWindow, self).__init__(parent)
        self.centralLayout = QHBoxLayout()
        self.setLayout(self.centralLayout)
        self.setModal(False)
        self.setWindowTitle("Help")
        self.setGeometry(0,0,500,500)
        icon = QPixmap()
        icon.load(":/images/icons/auroraplot_32.png")
        self.setWindowIcon(icon)
        
        self.textBrowser = QTextBrowser()
        self.centralLayout.addWidget(self.textBrowser)
        self.textBrowser.setOpenExternalLinks(False)
        self.textBrowser.setOpenLinks(True)
        self.textBrowser.setReadOnly(True)
        self.textBrowser.setSource(help_file)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = HelpWindow()
    frame.show()
    app.exec_()
