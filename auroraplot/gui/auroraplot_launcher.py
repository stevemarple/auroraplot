#!/usr/bin/env python3

import sys
import platform

import PySide
from PySide.QtGui import *
from PySide.QtCore import *

from ui_auroraplot_launcher import Ui_MainWindow
from ui_about import Ui_AboutWindow

import auroraplot_dataviewer

__version__ = '2016.0.1'


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.statusBar().hide()
        self.viewerIcon = ClickLabel(":/images/icons/dataviewer_128.png")
        self.centralLayout.addWidget(self.viewerIcon)
        self.editorIcon = ClickLabel(":/images/icons/qdceditor_128.png")
        self.centralLayout.addWidget(self.editorIcon)
        # Define actions
        self.actionAbout.triggered.connect(self.showAbout)
        self.viewerIcon.clicked.connect(self.viewerClicked)
        self.editorIcon.clicked.connect(self.editorClicked)
                
    def showAbout(self):
        aboutbox = AboutWindow()
        aboutbox.exec()

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
            
    def editorClicked(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.editorframe = auroraplot_dataviewer.MainWindow(self)
            self.editorframe.show()
        except Exception as e:
            raise e
            print("Error {}".format(e.args[0]))
        finally:
            QApplication.restoreOverrideCursor()

    def closeEvent(self, event):
        msgBox = QMessageBox()
        msgBox.setText("Really quit?")
        msgBox.setWindowTitle(" ")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        ret = msgBox.exec()
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
            
class AboutWindow(QDialog,Ui_AboutWindow):
    def __init__(self, parent=None):
        super(AboutWindow, self).__init__(parent)
        self.setupUi(self)
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    app.exec_()

