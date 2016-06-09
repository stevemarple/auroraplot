#!/usr/bin/env python3

import sys
import platform

import os
import time
os.environ['QT_API'] = 'pyside'
os.environ['TZ'] = 'UTC'
time.tzset()
import datetime

import PySide
from PySide.QtGui import *
from PySide.QtCore import *

import numpy as np

import matplotlib as mpl
mpl.use('Qt4Agg')
mpl.rcParams['backend.qt4']='PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.dt64tools as dt64
import auroraplot.magdata
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.uit
import auroraplot.datasets.dtu

from ui_auroraplot import Ui_MainWindow
from ui_about import Ui_AboutWindow


__version__ = '2016.0.1'


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        # Set up data canvas and ancillary data canvas
        self.splitter.setSizes([3,1])
        self.datafig = plt.figure()
        self.datafig.patch.set_facecolor('w')
        self.datacanvas = FigureCanvas(self.datafig)
        self.dataLayout.addWidget(self.datacanvas)
        self.ancifig = plt.figure()
        self.ancifig.patch.set_facecolor('w')
        self.ancicanvas = FigureCanvas(self.ancifig)
        self.anciLayout.addWidget(self.ancicanvas)
        
        # Set up time selection
        self.durationUnitsBox.addItem("days")
        self.durationUnitsBox.addItem("hours")
        self.durationUnitsBox.addItem("minutes")
        self.durationUnitsBox.addItem("seconds")
        self.durationUnitsBox.setCurrentIndex(0)
        self.durationBox.setMaximum(60)
        self.durationBox.setValue(1)

        # Fill plot options list - Problem with Qt when last item is spinBox (can vanish)
        self.optionsDict = {}
        self.optionsDict['dataResolution'] = SpinOption(self.optionsListWidget,
                                                        prefix="Data resolution ",
                                                        suffix=" seconds.",
                                                        value=1,minimum=1,maximum=120)
        self.optionsDict['updateInterval'] = SpinOption(self.optionsListWidget,
                                                        prefix="Auto-update interval ",
                                                        suffix=" minutes.",
                                                        value=5,minimum=1,maximum=120)
        self.optionsDict['drawQDC'] = CheckOption(self.optionsListWidget,
                                                  optionText="Draw Quiet Day Curve",
                                                  checked = False)


        # Define actions
        self.actionAbout.triggered.connect(self.showAbout)
        self.goButton.clicked.connect(self.goClicked)
                
    def showAbout(self):
        aboutbox = AboutWindow()
        aboutbox.exec()

    def goClicked(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.statusBar().showMessage("Loading data...")
            Qdate = self.calendarWidget.selectedDate()
            Qtime = self.starttimeWidget.time()
            durationUnits = self.durationUnitsBox.currentText()
            if durationUnits == "days":
                dt_unit = 'D'
            elif durationUnits == "hours":
                dt_unit = 'h'
            elif durationUnits == "minutes":
                dt_unit = 'm'
            else:
                dt_unit = 's'   
            st = Qdate_Qtime_to_dt64(Qdate,Qtime)
            et = st + np.timedelta64(self.durationBox.value(),dt_unit)
            # Want to complete the widget drawing processes before starting locking process
            QApplication.flush()
            md = ap.load_data('AURORAWATCHNET', 'LAN1', 'MagData', st, et)
            self.statusBar().showMessage("Plotting data...")
            self.datafig.clear()
            md.plot(figure=self.datafig.number,color='k')
            self.datacanvas.draw()
            self.statusBar().showMessage("Ready.")
        except Exception as e:
            raise e
            print("Error {}".format(e.args[0]))
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.flush()
        
class AboutWindow(QDialog,Ui_AboutWindow):
    def __init__(self, parent=None):
        super(AboutWindow, self).__init__(parent)
        self.setupUi(self)

class CheckOption:
    def __init__(self,listWidget,optionText="Check option",checked=True):
        self.listItem = QListWidgetItem(optionText,listWidget)
        #self.listItem.setFlags(Qt.ItemIsUserCheckable) # locks widget
        self.setCheckState(checked)
    def setCheckState(self,checked):
        if checked:
            self.listItem.setCheckState(Qt.Checked)
        else:
            self.listItem.setCheckState(Qt.Unchecked)
    def getCheckState(self):
        return self.listItem.checkState()
    
class SpinOption:
    def __init__(self,listWidget,prefix="",suffix="",value=0,minimum=0,maximum=99):
        self.spinWidget = QSpinBox()
        self.spinWidget.setPrefix(prefix)
        self.spinWidget.setSuffix(suffix)
        self.spinWidget.setValue(value)
        self.spinWidget.setMinimum(minimum)
        self.spinWidget.setMaximum(maximum)
        self.listItem = QListWidgetItem("",listWidget)
        listWidget.setItemWidget(self.listItem,self.spinWidget)
    def setMinimum(self,minimum):
        self.spinWidget.setMinimum(minimum)
    def setMaximum(self,maximum):
        self.spinWidget.setMaximum(maximum)
    def setValue(self,value):
        self.spinWidget.setValue(value)
    def getValue(self):
        return self.spinWidget.value()

        
def Qdate_Qtime_to_dt64(Qdate,Qtime):
    d = datetime.datetime(Qdate.year(),Qdate.month(),Qdate.day(),
                          Qtime.hour(),Qtime.minute(),Qtime.second(),
                          1000*(Qtime.msec())) # datetime uses microseconds
    return np.datetime64(d)

            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    frame.statusBar().showMessage("Ready.")
    app.exec_()

