#!/usr/bin/env python

import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rcParams['backend.qt4']='PySide'

import sys
import platform
import os
workingdir = os.path.dirname(__file__)

import PySide
from PySide.QtGui import *
from PySide.QtCore import *

import auroraplot_dataviewer
import auroraplot_help

import resources_rc

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.centralPanel = QWidget()
        self.centralLayout = QGridLayout()
        self.centralPanel.setLayout(self.centralLayout)
        self.setCentralWidget(self.centralPanel)
        self.statusBar().setVisible(False)
        self.statusBar().setVisible(False)
        self.setWindowTitle("AuroraPlot")
        self.setGeometry(0,0,20,20)
        icon = QPixmap()
        icon.load(":/images/icons/auroraplot_32.png")
        self.setWindowIcon(icon)

        self.viewerIcon = ClickLabel(":/images/icons/dataviewer_128.png")
        self.centralLayout.addWidget(self.viewerIcon,0,0,Qt.AlignLeft)
        self.creatorIcon = ClickLabel(":/images/icons/qdccreator_128.png")
        self.centralLayout.addWidget(self.creatorIcon,0,1,Qt.AlignLeft)
        self.helpButton = QPushButton("Help")
        self.centralLayout.addWidget(self.helpButton,1,0,1,2,Qt.AlignCenter)
        
        # Define actions
        self.helpButton.clicked.connect(self.showHelp)
        self.viewerIcon.clicked.connect(self.viewerClicked)
        self.creatorIcon.clicked.connect(self.creatorClicked)




    def viewerClicked(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.viewerframe = auroraplot_dataviewer.MainWindow(self)
            self.viewerframe.show()
        except Exception as e:
            raise e
            print("Error {}".format(e.args[0]))
        finally:
            QApplication.restoreOverrideCursor()
            
    def creatorClicked(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            pass
            #self.creatorframe = auroraplot_qdccreator.MainWindow(self)
            #self.creatorframe.show()
        except Exception as e:
            raise e
            print("Error {}".format(e.args[0]))
        finally:
            QApplication.restoreOverrideCursor()

    def showHelp(self):
        helpBox = auroraplot_help.HelpWindow(self)
        helpBox.show()
        
    def closeEvent(self, event):
        msgBox = QMessageBox()
        msgBox.setText("Are you sure?")
        msgBox.setWindowTitle(" ")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        ret = msgBox.exec_()
        if ret == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class ClickLabel(QLabel):
    clicked = Signal(str)
    def __init__(self, image):
        super(ClickLabel, self).__init__()
        pixmap = QPixmap()
        pixmap.load(image)
        self.setPixmap(pixmap)
    def mousePressEvent(self, event):
        self.clicked.emit(self.objectName())
            
app = QApplication(sys.argv)
frame = MainWindow()
frame.show()
screenRect = QApplication.desktop().screenGeometry()
QApplication.flush() # Need to process .show() before size is known
frame.setGeometry(screenRect.right()-frame.width(),
                  screenRect.bottom()-frame.height(),
                  frame.width(),frame.height())
app.exec_()

