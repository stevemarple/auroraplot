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
                          'Magnetometer Stack, nT':['MagData'],
                          'Riometer power, dB':['RioPower','RioQDC'],
                          'Riometer raw power':['RioRawPower'],
                          'Riometer absorption, dB':['RioAbs'],
                          'Riometer keogram, latitude/dB':['RioKeo'],
                          'Temperature, Celcius':['TemperatureData'],
                          'Voltage, V':['VoltageData']}

    def listAllPlotTypes(self):
        return list(sorted(self.dataTypes.keys()))

    def getDataTypes(self,project=None,plotType=None):
        # project is a string beginning with abbreviation
        # Used to check which dataType associated
        # with each plotType are available to project
        # Returns subset of dataTypes from self.dataTypes
        # as a list
        if plotType is None:
            return [None]
        allDataTypes = self.dataTypes[plotType]
        if project is None:
            return allDataTypes
        for i_d in range(len(self.datasetList)):
            abbr = self.datasetList[i_d].project['abbreviation']
            if len(project)<len(abbr):
                continue
            if any([not project[n] is abbr[n] for n in range(len(abbr))]):
                continue
            # If we get here, we found match for project
            availableDataTypes = dir(self.datasetList[i_d])
            return [t for t in allDataTypes if t in availableDataTypes]
        # Didn't find a match for project
        return [None]
    
    def getProjects(self,plotType=None,project=None):
        # plotType is a key from the self.dataTypes dict
        # project is None, or text that begins with project abbreviation
        # returns lists of project abbreviations and names 
        projectAbbrs = []
        projectNames = []
        for i_d in range(len(self.datasetList)):
            if not plotType is None:
                allDataTypes = self.dataTypes[plotType]
                availableDataTypes = dir(self.datasetList[i_d])
                if all([not t in availableDataTypes for t in allDataTypes]):
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
    
    def getSites(self,project=None,plotType=None,site=None):
        # project is None, or text that begins with project abbreviation
        # site is None, or text that begins with site key
        # plotType is a key from the self.dataTypes dict
        siteKeys = []
        siteLocs = []
        for i_d in range(len(self.datasetList)):
            if not project is None:
                abbr = self.datasetList[i_d].project['abbreviation']
                if len(project)<len(abbr):
                    continue
                if any([not project[n] is abbr[n] for n in range(len(abbr))]):
                    continue
            if not plotType is None:
                wantedTypes = self.dataTypes[plotType]
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

    def getAvailableArchives(self,project,site,dataType):
        # Returns sorted list of available archives for particular project, site, dataType.
        # Returns None if it fails to find any.
        # project is text that begins with project abbreviation
        # site is text that begins with site key
        projectInd = None
        siteKey = None
        for i_d in range(len(self.datasetList)):
            projectAbbr = self.datasetList[i_d].project['abbreviation']
            if len(project)<len(projectAbbr):
                continue
            if any([not project[n] is projectAbbr[n] for n in range(len(projectAbbr))]):
                continue
            # if it gets here, found match for project, now look for a match for site
            projectInd = i_d
            allSiteKeys = list(sorted(self.datasetList[i_d].project['sites'].keys()))
            for k in allSiteKeys:
                if len(site) < len(k): 
                    continue
                elif all([site[n] is k[n] for n in range(len(k))]):
                    if dataType in list(self.datasetList[i_d].project['sites'][k]['data_types'].keys()):
                        siteKey = k
            if not siteKey is None:
                break
        if (not projectInd is None) and (not siteKey is None):
            arch = sorted(list(
                self.datasetList[projectInd].project['sites'][siteKey]['data_types'][dataType].keys()))
            if len(arch):
                return arch
            else:
                return None
        else:
            return None

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
        self.currentPlotType = None
        self.datasets = Datasets()
        self.plotsTreeWidget.setColumnCount(5)
        self.plotsTreeWidget.setHeaderLabels(["Plot type","Site",
                                              "Channels","Project","Archive"])
        self.plotsTreeWidget.header().resizeSection(0,130)
        self.plotsTreeWidget.header().resizeSection(1,60)
        self.plotsTreeWidget.header().resizeSection(2,90)
        self.plotsTreeWidget.header().resizeSection(3,60)
        self.plotsTreeWidget.header().resizeSection(4,60)
        for u in self.datasets.listAllPlotTypes():
            self.plotTypeBox.addItem(u)
        self.plotTypeBox.setCurrentIndex(0)
        self.addPlotButton.clicked.connect(self.addPlotIsClicked)
        self.plotsTreeWidget.itemClicked.connect(self.plotTypeIsChanged)
        self.addDatasetButton.clicked.connect(self.addDatasetIsClicked)
        self.projectBox.currentIndexChanged.connect(self.projectIsChanged)
        self.projectBox.setCurrentIndex(0)
        self.projectIsChanged()
        self.siteBox.currentIndexChanged.connect(self.updateArchiveBox)
        self.dataTypeBox.currentIndexChanged.connect(self.updateArchiveBox)
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
        self.currentPlotType = self.plotTypeBox.currentText()
        newItem = QTreeWidgetItem()
        pixmap = QPixmap()
        pixmap.load(':/images/icons/plot_22.png')
        icon = QIcon(pixmap)
        newItem.setIcon(0,icon)
        newItem.setText(0,self.currentPlotType)
        self.plotsTreeWidget.addTopLevelItem(newItem)
        self.plotsTreeWidget.setCurrentItem(newItem)
        self.plotsTreeWidget.expandItem(newItem)
        self.plotTypeIsChanged()

    def plotTypeIsChanged(self):
        itemClicked = self.plotsTreeWidget.currentItem()
        # Find top level item, there are only 2 levels. Top level has no parent
        if itemClicked is None:
            plotType = None
        elif itemClicked.parent() is None:
            plotType = itemClicked.text(0)
        elif itemClicked.parent().parent() is None:
            plotType = itemClicked.parent().text(0)
        self.currentPlotType = plotType
        self.projectBox.clear()
        self.siteBox.clear()
        self.dataTypeBox.clear()
        if self.currentPlotType is None:
            return
        projectAbbrs, projectNames = self.datasets.getProjects(
                                              plotType=self.currentPlotType)
        for n in range(len(projectNames)):
            self.projectBox.addItem("".join([projectAbbrs[n],", ",
                                             projectNames[n]]))
        self.projectIsChanged()
        
    def projectIsChanged(self):
        project = self.projectBox.currentText()
        self.dataTypeBox.clear()
        self.siteBox.clear()
        dataTypes = self.datasets.getDataTypes(project=project,
                                               plotType=self.currentPlotType)
        siteKeys, siteLocs = self.datasets.getSites(project=project,
                                                    plotType=self.currentPlotType)
        for n in range(len(siteKeys)):
            self.siteBox.addItem("".join([siteKeys[n],", ",siteLocs[n]]))
        for n in range(len(dataTypes)):
            self.dataTypeBox.addItem(dataTypes[n])

    def updateArchiveBox(self):
        project = self.projectBox.currentText()
        site = self.siteBox.currentText()
        dataType = self.dataTypeBox.currentText()
        self.archiveBox.clear()
        if (not project is None) and (not site is None) and (not dataType is None):
            archives = self.datasets.getAvailableArchives(project,site,dataType)
        if not archives is None:
            for n in range(len(archives)):
                self.archiveBox.addItem(archives[n])
        defaultInd = self.archiveBox.findText("default")
        if defaultInd > -1:
            self.archiveBox.setCurrentIndex(defaultInd)
            
                
    def addDatasetIsClicked(self):
        currentTreeItem = self.plotsTreeWidget.currentItem()
        project = self.projectBox.currentText()
        site = self.siteBox.currentText()
        dataType = self.dataTypeBox.currentText()
        channelsText = self.channelsLineEdit.text()
        archive = self.archiveBox.currentText()
        # Find top level item, there are only 2 levels. Top level has no parent
        if currentTreeItem is None:
            return
        if currentTreeItem.parent() is None:
            topLevelItem = currentTreeItem
        elif currentTreeItem.parent().parent() is None:
            topLevelItem = currentTreeItem.parent()
        newItem = QTreeWidgetItem()
        newItem.setText(0,dataType)
        newItem.setText(1,site)
        newItem.setText(2,channelsText)
        newItem.setText(3,project)
        newItem.setText(4,archive)
        topLevelItem.addChild(newItem)
            
    def removeDatasetIsClicked(self):
        currentTreeItem = self.plotsTreeWidget.currentItem()
        if currentTreeItem is None:
            return
        if currentTreeItem.parent() is None:
            self.plotsTreeWidget.takeTopLevelItem(
                self.plotsTreeWidget.indexOfTopLevelItem(currentTreeItem))
            self.plotTypeIsChanged() # Need to update currentPlotType
        elif currentTreeItem.parent().parent() is None:
            parent = currentTreeItem.parent()
            parent.takeChild(parent.indexOfChild(currentTreeItem))
        
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
                    dataType = child.text(0)
                    site = child.text(1)
                    channels = child.text(2)
                    project = child.text(3)
                    archive = child.text(4)
                    pabbr,pnames = self.datasets.getProjects(project=project)
                    sabbr,snames = self.datasets.getSites(project=project,site=site)
                    try:
                        md = ap.load_data(pabbr[0], sabbr[0], dataType, st, et,
                                          archive=archive,
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
                        self.log.append("Error {}".format(e.args[0]))
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

