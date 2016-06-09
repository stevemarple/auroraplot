# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'about.ui'
#
# Created: Mon Jun  6 15:37:11 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_AboutWindow(object):
    def setupUi(self, AboutWindow):
        AboutWindow.setObjectName("AboutWindow")
        AboutWindow.resize(400, 262)
        self.pushButton = QtGui.QPushButton(AboutWindow)
        self.pushButton.setGeometry(QtCore.QRect(160, 230, 85, 27))
        self.pushButton.setObjectName("pushButton")
        self.textBrowser = QtGui.QTextBrowser(AboutWindow)
        self.textBrowser.setGeometry(QtCore.QRect(10, 10, 381, 211))
        self.textBrowser.setAutoFormatting(QtGui.QTextEdit.AutoNone)
        self.textBrowser.setAcceptRichText(False)
        self.textBrowser.setSource(QtCore.QUrl("qrc:/file/About"))
        self.textBrowser.setSearchPaths([])
        self.textBrowser.setOpenLinks(True)
        self.textBrowser.setObjectName("textBrowser")

        self.retranslateUi(AboutWindow)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL("clicked()"), AboutWindow.close)
        QtCore.QMetaObject.connectSlotsByName(AboutWindow)

    def retranslateUi(self, AboutWindow):
        AboutWindow.setWindowTitle(QtGui.QApplication.translate("AboutWindow", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("AboutWindow", "Ok", None, QtGui.QApplication.UnicodeUTF8))

import resources_rc
