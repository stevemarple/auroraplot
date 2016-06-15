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

from ui_auroraplot_dataviewer import Ui_MainWindow
from ui_about import Ui_AboutWindow

import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.uit
import auroraplot.datasets.dtu

class Datasets:
    def __init__(self):
        self.datasetList = [auroraplot.datasets.aurorawatchnet,
                            auroraplot.datasets.samnet,
                            auroraplot.datasets.uit,
                            auroraplot.datasets.dtu]
        self.dataTypes = {'Magnetic field, nT':['MagData','MagQDC'],
                          'Temperature, Celcius':['TemperatureData'],
                          'Voltage, V':['VoltageData']}

    def listAllTypes(self):
        return list(sorted(self.dataTypes.keys()))

    def getDataMethods(self,project=None,dataType=None):
        # project is a string beginning with abbreviation
        # Used to check which methods associated
        # with each datatype are available to project
        # Returns subset of methods as from self.dataTypes
        # as a list
        if dataType is None:
            return [None]
        allMethods = self.dataTypes[dataType]
        if project is None:
            return allMethods
        for i_d in range(len(self.datasetList)):
            abbr = self.datasetList[i_d].project['abbreviation']
            if len(project)<len(abbr):
                continue
            if any([not project[n] is abbr[n] for n in range(len(abbr))]):
                continue
            # If we get here, we found match for project
            availableMethods = dir(self.datasetList[i_d])
            return [t for t in allMethods if t in availableMethods]
        # Didn't find a match for project
        return [None]
    
    def getProjects(self,dataType=None,project=None):
        # dataType is a key from the self.dataTypes dict
        # project is None, or text that begins with project abbreviation
        # returns lists of project abbreviations and names 
        projectAbbrs = []
        projectNames = []
        for i_d in range(len(self.datasetList)):
            if not dataType is None:
                allMethods = self.dataTypes[dataType]
                availableMethods = dir(self.datasetList[i_d])
                if all([not t in availableMethods for t in allMethods]):
                    continue
            if not project is None:
                abbr = self.datasetList[i_d].project['abbreviation']
                if len(project)<len(abbr):
                    continue
                if any([not project[n] is abbr[n] for n in range(len(abbr))]):
                    continue
            projectAbbrs.append(self.datasetList[i_d].project['abbreviation'])
            projectNames.append(self.datasetList[i_d].project['name'])
        return projectAbbrs,projectNames
    
    def getSites(self,project=None,dataType=None,site=None):
        # project is None, or text that begins with project abbreviation
        # dataType is a key from the self.dataTypes dict
        siteKeys = []
        siteLocs = []
        for i_d in range(len(self.datasetList)):
            if not project is None:
                abbr = self.datasetList[i_d].project['abbreviation']
                if len(project)<len(abbr):
                    continue
                if any([not project[n] is abbr[n] for n in range(len(abbr))]):
                    continue
            if not dataType is None:
                wantedTypes = self.dataTypes[dataType]
                currentTypes = dir(self.datasetList[i_d])
                if all([t not in currentTypes for t in wantedTypes]):
                    continue
            allSiteKeys = list(sorted(self.datasetList[i_d].project['sites'].keys()))
            for k in allSiteKeys:
                if site is None: 
                    siteKeys.append(k)
                    siteLocs.append(self.datasetList[i_d].project['sites'][k]['location'])
                elif all([site[n] is k[n] for n in range(len(k))]):
                    siteKeys.append(k)
                    siteLocs.append(self.datasetList[i_d].project['sites'][k]['location'])
        return siteKeys,siteLocs
            

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        # Set up data canvas and ancillary data canvas
        self.splitter.setSizes([1,0])
        self.dataFig = plt.figure()
        self.dataFig.patch.set_facecolor('w')
        self.dataCanvas = FigureCanvas(self.dataFig)
        self.dataLayout.addWidget(self.dataCanvas)
        
        # Set up data types
        self.currentDataType = None
        self.datasets = Datasets()
        self.plotsTreeWidget.setColumnCount(4)
        self.plotsTreeWidget.setHeaderLabels(["Plot type","Project","Site","Channels"])
        for u in self.datasets.listAllTypes():
            self.plotTypeBox.addItem(u)
        self.plotTypeBox.setCurrentIndex(0)
        self.addPlotButton.clicked.connect(self.addPlotIsClicked)
        self.plotsTreeWidget.itemClicked.connect(self.plotsTreeIsClicked)
        self.addDatasetButton.clicked.connect(self.addDatasetIsClicked)
        self.projectBox.currentIndexChanged.connect(self.projectIsChanged)
        self.projectBox.setCurrentIndex(0)
        self.projectIsChanged()
        self.removeDatasetButton.clicked.connect(self.removeDatasetIsClicked)

        
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

        # Set up logging
        self.log = Log(self.logTextEdit)
        self.logClearButton.clicked.connect(self.log.clear)
        self.logSaveButton.clicked.connect(self.log.save)
        
        # Define other actions
        self.actionAbout.triggered.connect(self.showAbout)
        self.goButton.clicked.connect(self.goClicked)

    def addPlotIsClicked(self):
        self.currentDataType = self.plotTypeBox.currentText()
        newItem = QTreeWidgetItem()
        newItem.setText(0,self.currentDataType)
        self.plotsTreeWidget.addTopLevelItem(newItem)
        self.plotsTreeWidget.setCurrentItem(newItem)
        self.plotsTreeWidget.expandItem(newItem)
        self.dataTypeIsChanged()

    def plotsTreeIsClicked(self):
        itemClicked = self.plotsTreeWidget.currentItem()
        # Find top level item, there are only 2 levels. Top level has no parent
        if itemClicked.parent() is None:
            dataType = itemClicked.text(0)
        elif itemClicked.parent().parent() is None:
            dataType = itemClicked.parent().text(0)
        else:
            return
        self.currentDataType = dataType
        self.dataTypeIsChanged()

    def dataTypeIsChanged(self):
        self.projectBox.clear()
        self.siteBox.clear()
        projectAbbrs, projectNames = self.datasets.getProjects(dataType=self.currentDataType)
        for n in range(len(projectNames)):
            self.projectBox.addItem("".join([projectAbbrs[n],", ",projectNames[n]]))
        self.projectIsChanged()
        
    def projectIsChanged(self):
        project = self.projectBox.currentText()
        self.methodBox.clear()
        self.siteBox.clear()
        methods = self.datasets.getDataMethods(project=project,
                                               dataType=self.currentDataType)
        siteKeys, siteLocs = self.datasets.getSites(project=project,
                                                    dataType=self.currentDataType)
        for n in range(len(siteKeys)):
            self.siteBox.addItem("".join([siteKeys[n],", ",siteLocs[n]]))
        for n in range(len(methods)):
            self.methodBox.addItem(methods[n])
            
    def addDatasetIsClicked(self):
        currentTreeItem = self.plotsTreeWidget.currentItem()
        project = self.projectBox.currentText()
        site = self.siteBox.currentText()
        dataMethod = self.methodBox.currentText()
        channelsText = self.channelsLineEdit.text()
        # Find top level item, there are only 2 levels. Top level has no parent
        if currentTreeItem is None:
            return
        if currentTreeItem.parent() is None:
            topLevelItem = currentTreeItem
        elif currentTreeItem.parent().parent() is None:
            topLevelItem = currentTreeItem.parent()
        newItem = QTreeWidgetItem()
        newItem.setText(0,dataMethod)
        newItem.setText(1,project)
        newItem.setText(2,site)
        newItem.setText(3,channelsText)
        topLevelItem.addChild(newItem)
            
    def removeDatasetIsClicked(self):
        currentTreeItem = self.plotsTreeWidget.currentItem()
        if currentTreeItem is None:
            return
        if currentTreeItem.parent() is None:
            self.plotsTreeWidget.takeTopLevelItem(
                self.plotsTreeWidget.indexOfTopLevelItem(currentTreeItem))
        elif currentTreeItem.parent().parent() is None:
            parent = currentTreeItem.parent()
            parent.removeChild(currentTreeItem)
            #parent.takeChild(parent.indexOfChild(currentTreeItem))

    def showAbout(self):
        aboutbox = AboutWindow()
        aboutbox.exec()

    def goClicked(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            numberOfPlots = self.plotsTreeWidget.topLevelItemCount()
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
            self.log.append("".join(["loading data: ",np.datetime_as_string(st)," to ",
                                     np.datetime_as_string(et)]))
            QApplication.flush()
            self.dataFig.clear()
            for sp in range(numberOfPlots):
                ax = self.dataFig.add_subplot(numberOfPlots,1,sp+1)                
                topLevelItem = self.plotsTreeWidget.topLevelItem(sp)
                numberOfData = topLevelItem.childCount()
                for p in range(numberOfData):
                    child = topLevelItem.child(p)
                    method = child.text(0)
                    project = child.text(1)
                    site = child.text(2)
                    channels = child.text(3)
                    pabbr,pnames = self.datasets.getProjects(project=project)
                    sabbr,snames = self.datasets.getSites(project=project,site=site)
                    try:
                        md = ap.load_data(pabbr[0], sabbr[0], method, st, et,
                                          cadence=np.timedelta64(40,'s'))
                        if md is not None and md.data.size:
                            md.mark_missing_data(cadence=2*md.nominal_cadence)
                            self.statusBar().showMessage("Plotting data...")
                            #self.datafig.clear()
                            md.plot(axes = ax)
                            self.dataCanvas.draw()
                            self.log.append("Plotting completed successfully.")
                            self.statusBar().showMessage("Ready.")
                        else:
                            self.log.append("No data.")
                            self.statusBar().showMessage("No data.")
                    except Exception as e:
                        self.log.append("Loading data failed.")
                        #self.log.append("Error {}".format(e.args[0]))
        except Exception as e:
            self.log.append("Plotting failed.")
            self.log.append("Error {}".format(e.args[0]))
        finally:
            QApplication.restoreOverrideCursor()
            QApplication.flush()
        
class AboutWindow(QDialog,Ui_AboutWindow):
    def __init__(self, parent=None):
        super(AboutWindow, self).__init__(parent)
        self.setupUi(self)

class Log:
    def __init__(self,textEditWidget):
        self.saveDir = QDir.home() # folder for save dialog on first run
        self.textEditWidget = textEditWidget
        self.textEditWidget.appendPlainText("".join(["Log started at  ",
            "{:%X %Z %a %d %B %Y}".format(datetime.datetime.now())]))
    def append(self,logText):
        self.textEditWidget.appendPlainText("".join([
            "{:%X %Z} ".format(datetime.datetime.now()),logText]))
    def clear(self):
        msgBox = QMessageBox()
        msgBox.setText("Really clear the log?")
        msgBox.setWindowTitle(" ")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        ret = msgBox.exec()
        if ret == QMessageBox.Yes:
            self.textEditWidget.clear()
            self.textEditWidget.appendPlainText("".join(["Log started at  ",
                    "{:%X %Z %a %d %B %Y}".format(datetime.datetime.now())]))
    def save(self):
        fileBox = QFileDialog()
        fileBox.setAcceptMode(QFileDialog.AcceptSave)
        fileBox.setDirectory(self.saveDir)
        ret = fileBox.exec()
        self.saveDir = fileBox.directory()
        if ret == QFileDialog.Accepted:
            try:
                fileName = fileBox.selectedFiles()[0]
                with open(fileName,"wt") as file:
                    file.write(self.textEditWidget.document().toPlainText())
                self.append("".join(["Log written to ",fileName]))
            except Exception as e:
                self.append("Failed to write log to file")
                self.append("Error {}".format(e.args[0]))
        
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

