#!/usr/bin/env python

import sys
import platform

import os
workingdir = os.path.dirname(__file__)

import logging
import traceback
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as Navigationtoolbar
from ui_auroraplot_dataviewer import Ui_MainWindow
import auroraplot_help

import matplotlib.pyplot as plt

import auroraplot as ap
import auroraplot.dt64tools as dt64
from auroraplot.data import Data

import auroraplot.magdata
import auroraplot.riodata

import auroraplot.datasets.riometernet
import auroraplot.datasets.aurorawatchnet
import auroraplot.datasets.samnet
import auroraplot.datasets.uit
import auroraplot.datasets.dtu
import auroraplot.datasets.bgs_schools

def get_units_prefix(data_type):
    if data_type == "MagData":
        units_prefix = 'n'
    else:
        units_prefix = ''
    return units_prefix

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.statusBar().setVisible(False)
        self.menuBar().setVisible(False)
        # Set up data canvas and ancillary data canvas
        self.splitter.setSizes([1,0])
        self.dataFig = plt.figure()
        self.dataFig.patch.set_facecolor('w')
        self.dataCanvas = FigureCanvas(self.dataFig)
        self.dataLayout.addWidget(self.dataCanvas)
        self.mpltoolbar = Navigationtoolbar(self.dataCanvas, self.toolbarFrame)
        self.toolbarLayout.setAlignment(Qt.AlignCenter)
        self.toolbarLayout.addWidget(self.mpltoolbar)
        self.progressBar.setFormat("")
        self.progressBar.setValue(0)

        # Fill plot options page
        self.optionsLayout.setAlignment(Qt.AlignTop)
        self.updateIntervalLabel = QLabel("Real-time update interval: ")
        self.updateIntervalUnitsBox = QComboBox()
        self.updateIntervalUnitsBox.addItem("days")
        self.updateIntervalUnitsBox.addItem("hours")
        self.updateIntervalUnitsBox.addItem("minutes")
        self.updateIntervalUnitsBox.addItem("seconds")
        self.updateIntervalUnitsBox.setCurrentIndex(3)
        self.updateIntervalBox = QSpinBox()
        self.updateIntervalBox.setRange(1,60)
        self.updateIntervalBox.setValue(30)
        self.optionsLayout.addWidget(self.updateIntervalLabel,0,0,
                                            Qt.AlignRight)
        self.optionsLayout.addWidget(self.updateIntervalBox,0,1,
                                            Qt.AlignRight)
        self.optionsLayout.addWidget(self.updateIntervalUnitsBox,0,2,
                                            Qt.AlignLeft)
        self.integrationLabel = QLabel("Integration: ")
        self.integrationUnitsBox = QComboBox()
        self.integrationUnitsBox.addItem("days")
        self.integrationUnitsBox.addItem("hours")
        self.integrationUnitsBox.addItem("minutes")
        self.integrationUnitsBox.addItem("seconds")
        self.integrationUnitsBox.setCurrentIndex(3)
        self.integrationBox = QSpinBox()
        self.integrationBox.setRange(1,60)
        self.integrationBox.setValue(1)
        self.optionsLayout.addWidget(self.integrationLabel,1,0,Qt.AlignRight)
        self.optionsLayout.addWidget(self.integrationBox,1,1,Qt.AlignRight)
        self.optionsLayout.addWidget(self.integrationUnitsBox,1,2,Qt.AlignLeft)
        
        
        
        # Set up data types
        self.current_data_type = None
        # Want to select data_type first, so need to make a dict with
        # order of: channels=datadict[data_type][project][site][archive]
        self.datadict = {}
        for p in ap.get_projects():
            for s in list(ap.get_sites(p)):
                for d in ap.get_data_types(p,s):
                    for a in ap.get_archives(p,s,d)[0]:
                        chans = ap.get_archive_info(p,s,d,a)[1]['channels']
                        if d not in self.datadict:
                            self.datadict[d] = {}
                        if p not in self.datadict[d]:
                            self.datadict[d][p] = {}
                        if s not in self.datadict[d][p]:
                            self.datadict[d][p][s] = {}
                        self.datadict[d][p][s][a] = chans  
        self.plotsTreeWidget.setColumnCount(4)
        self.plotsTreeWidget.setHeaderLabels(["Plot/Archive","Project","Site",
                                              "Channels"])
        self.plotsTreeWidget.header().resizeSection(0,120)
        self.plotsTreeWidget.header().resizeSection(1,55)
        self.plotsTreeWidget.header().resizeSection(2,45)
        self.plotsTreeWidget.header().resizeSection(3,130)
        for d in sorted(list(self.datadict.keys())):
            self.dataTypeBox.addItem(d)
        self.dataTypeBox.setCurrentIndex(0)
        self.addPlotButton.clicked.connect(self.add_plot_clicked)
        self.plotsTreeWidget.itemClicked.connect(self.data_type_changed)
        self.addDatasetButton.clicked.connect(self.add_dataset_clicked)
        self.projectBox.currentIndexChanged.connect(self.project_changed)
        self.siteBox.currentIndexChanged.connect(self.site_changed)
        self.archiveBox.currentIndexChanged.connect(self.archive_changed)
        self.removeDatasetButton.clicked.connect(self.remove_dataset_clicked)

        # Set up timer for auto-update
        self.last_auto_update = np.datetime64('now')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_timer)
        self.timer.start(1000)
        
        # Set up time selection
        self.durationUnitsBox.addItem("days")
        self.durationUnitsBox.addItem("hours")
        self.durationUnitsBox.addItem("minutes")
        self.durationUnitsBox.addItem("seconds")
        self.durationUnitsBox.setCurrentIndex(0)
        self.durationBox.setRange(1,60)
        self.durationBox.setValue(1)
        
        # Set up logging
        self.log = Log(self.logTextEdit)
        self.log_handler = CustomLogHandler(self.log)
        logger.addHandler(self.log_handler)
        self.logClearButton.clicked.connect(self.log.clear)
        self.logSaveButton.clicked.connect(self.log.save)
        
        # Define other actions
        self.helpButton.clicked.connect(self.showHelp)
        self.goButton.clicked.connect(self.draw_plots)

    def check_timer(self):
        if not self.autoUpdateCheckBox.checkState():
            self.progressBar.setTextVisible(False)
            self.progressBar.setValue(0)
            return
        time_now = np.datetime64('now')
        intervalUnits = self.updateIntervalUnitsBox.currentText()
        if intervalUnits == "days":
            dt_unit = 'D'
        elif intervalUnits == "hours":
            dt_unit = 'h'
        elif intervalUnits == "minutes":
            dt_unit = 'm'
        else:
            dt_unit = 's'
        interval = np.timedelta64(self.updateIntervalBox.value(),dt_unit)
        seconds_left = (self.last_auto_update+interval
                        -time_now).astype('m8[s]').astype('int64')
        if seconds_left <= 0:
            self.last_auto_update = time_now
            durationUnits = self.durationUnitsBox.currentText()
            if durationUnits == "days":
                dt_unit = 'D'
            elif durationUnits == "hours":
                dt_unit = 'h'
            elif durationUnits == "minutes":
                dt_unit = 'm'
            else:
                dt_unit = 's'
            # Get next dt_unit boundary
            et = (time_now.astype(''.join(['M8[',dt_unit,']']))
                  +np.timedelta64(1,dt_unit)).astype('M8[s]')
            st = et - np.timedelta64(self.durationBox.value(),dt_unit)
            Qdate, Qtime = dt64_to_Qdate_Qtime(st)
            self.calendarWidget.setSelectedDate(Qdate)
            self.starttimeWidget.setTime(Qtime)
            self.draw_plots()
        else:
            self.progressBar.setFormat("Updating in %v s")
            self.progressBar.setTextVisible(True)
            self.progressBar.setMaximum(interval/np.timedelta64(1,'s'))
            self.progressBar.setValue(seconds_left)
            QApplication.flush()
        
    def add_plot_clicked(self):
        data_type = self.dataTypeBox.currentText()
        newItem = QTreeWidgetItem()
        pixmap = QPixmap()
        pixmap.load(':/images/icons/plot_22.png')
        icon = QIcon(pixmap)
        newItem.setIcon(0,icon)
        newItem.setText(0,data_type)
        self.plotsTreeWidget.addTopLevelItem(newItem)
        self.plotsTreeWidget.setCurrentItem(newItem)
        self.plotsTreeWidget.expandItem(newItem)
        self.data_type_changed()

    def data_type_changed(self):
        currentItem = self.plotsTreeWidget.currentItem()
        # Find top level item, there are only 2 levels.
        # According to Qt Docs, top level has no parent
        if currentItem is None:
            data_type = None
        elif currentItem.parent() is None:
            data_type = currentItem.text(0)
        elif currentItem.parent().parent() is None:
            data_type = currentItem.parent().text(0)
        if self.current_data_type == data_type:
            return
        self.current_data_type = data_type
        self.projectBox.clear()
        self.siteBox.clear()
        self.archiveBox.clear()
        self.channelsLineEdit.clear()
        if self.current_data_type not in self.datadict.keys():
            return
        projects = sorted(list(self.datadict[self.current_data_type].keys()))
        for p in projects:
            pname = ap.get_project_info(p)['name']
            self.projectBox.addItem("".join([p,", ",pname]))
        self.projectBox.setCurrentIndex(0)
        self.project_changed()
        
    def project_changed(self):
        project = self.projectBox.currentText().split(',')[0]
        if self.current_data_type not in self.datadict.keys():
            return
        if project not in self.datadict[self.current_data_type].keys():
            return
        self.siteBox.clear()
        self.archiveBox.clear()
        sites = sorted(list(self.datadict[self.current_data_type][project].keys()))
        for s in sites:
            sloc = ap.get_site_info(project,s)['location']
            self.siteBox.addItem("".join([s,", ",sloc]))
        self.siteBox.setCurrentIndex(0)
        self.site_changed()

    def site_changed(self):
        project = self.projectBox.currentText().split(',')[0]
        site = self.siteBox.currentText().split(',')[0]
        if not self.current_data_type in self.datadict.keys():
            return
        if not project in self.datadict[self.current_data_type].keys():
            return
        if not site in self.datadict[self.current_data_type][project].keys():
            return
        self.archiveBox.clear()
        archives = sorted(list(self.datadict[self.current_data_type]\
                               [project][site].keys()))
        default_archive = ap.get_archives(project,site,self.current_data_type)[1]
        for a in archives:
            ainfo = ap.get_archive_info(project,site,self.current_data_type,a)[1]
            if 'nominal_cadence' in ainfo:
                nc = "".join([str(ainfo['nominal_cadence'].astype('m8[s]').astype('int64')),
                              ' s cadence'])
                self.archiveBox.addItem("".join([a,", ",nc]))
            else:
                self.archiveBox.addItem(a)
        if len(default_archive) and default_archive in archives:
            self.archiveBox.setCurrentIndex(
                np.where([default_archive == d for d in archives])[0])
        else:
            self.archiveBox.setCurrentIndex(0)
        self.archive_changed()

    def archive_changed(self):
        project = self.projectBox.currentText().split(',')[0]
        site = self.siteBox.currentText().split(',')[0]
        archive = self.archiveBox.currentText().split(',')[0]
        if not self.current_data_type in self.datadict.keys():
            return
        if not project in self.datadict[self.current_data_type].keys():
            return
        if not site in self.datadict[self.current_data_type][project].keys():
            return
        if not archive in self.datadict[self.current_data_type][project][site].keys():
            return
        ainfo = ap.get_archive_info(project,site,self.current_data_type,archive)[1]
        all_channels = ainfo['channels']
        if not hasattr(all_channels,'__iter__'):
            all_channels = list(all_channels)
        channels=''
        if all([c.isdigit() for c in all_channels]):
            all_channels = sorted(all_channels)
            min_c = (np.min(np.array([int(c) for c in all_channels])))
            max_c = (np.max(np.array([int(c) for c in all_channels])))
            if len(all_channels) < 6 and all([c in all_channels for c in range(min_c,max_c+1)]):
                channels = "-".join([min_c,max_c])
            else:
                c = str(int(np.percentile(np.array([int(c) for c in all_channels]),50)))
                if c in all_channels:
                    channels = c
        else:
            if len(all_channels) < 6:
                channels = ", ".join([c for c in all_channels])
            else:
                channels = channels[0]
        self.channelsLineEdit.setText(channels)
        
    def add_dataset_clicked(self):
        currentTreeItem = self.plotsTreeWidget.currentItem()
        project = self.projectBox.currentText().split(',')[0]
        site = self.siteBox.currentText().split(',')[0]
        archive = self.archiveBox.currentText().split(',')[0]
        channelsText = self.channelsLineEdit.text()
        # Find top level item, there are only 2 levels. Top level has no parent
        if currentTreeItem is None:
            return
        if currentTreeItem.parent() is None:
            topLevelItem = currentTreeItem
        elif currentTreeItem.parent().parent() is None:
            topLevelItem = currentTreeItem.parent()
        newItem = QTreeWidgetItem()
        newItem.setText(0,archive)
        newItem.setText(1,project)
        newItem.setText(2,site)
        newItem.setText(3,channelsText)
        topLevelItem.addChild(newItem)
            
    def remove_dataset_clicked(self):
        currentTreeItem = self.plotsTreeWidget.currentItem()
        if currentTreeItem is None:
            return
        if currentTreeItem.parent() is None:
            self.plotsTreeWidget.takeTopLevelItem(
                self.plotsTreeWidget.indexOfTopLevelItem(currentTreeItem))
            self.data_type_changed() # Need to update project
        elif currentTreeItem.parent().parent() is None:
            parent = currentTreeItem.parent()
            parent.takeChild(parent.indexOfChild(currentTreeItem))

    def parseChannels(self,ctext):
        if ctext is None:
            return [None]
        channels = []
        try:
            if len(ctext)>1:
                channels = []
                spl = ctext.split(',')
                for s in spl:
                    if len(s):
                        # remove spaces that follow commas
                        while len(s) and s[0] == ' ':
                            s = s[1:]
                        # remove trailing spaces
                        while len(s) and s[-1] == ' ':
                            s = s[:-1]
                        if len(s)>1:
                            s2 = s.split('-')
                            if len(s2)>2:
                                # more than one '-'
                                continue
                            if len(s2)==2:
                                if s2[0].isnumeric() and s2[1].isnumeric():
                                    if int(s2[0]) >= int(s2[1]):
                                        s2 = [str(n) for n in range(int(s2[0]),
                                                                    int(s2[1])+1)]
                                    else:
                                        s2 = [str(n) for n in range(int(s2[1]),
                                                                    int(s2[0])+1)]
                                else:
                                    # Don't understand eg H-Z
                                    continue
                            channels.extend(s2)
                        else:
                            channels.extend(s)
            else:
                channels = ctext
        except Exception as e:
            logger.info('Could not parse channels')
            logger.debug(str(e))
            logger.debug(traceback.format_exc())
        return channels
    
    def showHelp(self):
        helpBox = auroraplot_help.HelpWindow(self)
        helpBox.show()

    def draw_plots(self):
        self.timer.stop()
        self.last_auto_update = np.datetime64('now')
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

            integrationUnits = self.integrationUnitsBox.currentText()
            if integrationUnits == "days":
                dt_unit = 'D'
            elif integrationUnits == "hours":
                dt_unit = 'h'
            elif integrationUnits == "minutes":
                dt_unit = 'm'
            else:
                dt_unit = 's'
            integration_interval = np.timedelta64(self.integrationBox.value(),dt_unit)
            
            logger.info("".join(["Loading data: ",np.datetime_as_string(st),
                                     " to ", np.datetime_as_string(et)]))
            QApplication.flush()
            self.dataFig.clear()
            ax = []
            current_num_data = 0.0
            total_num_data = 0.0
            for sp in range(numberOfPlots):
                topLevelItem = self.plotsTreeWidget.topLevelItem(sp)
                total_num_data += topLevelItem.childCount()
            self.progressBar.setMaximum(total_num_data)
            self.progressBar.setFormat("Loaded data: %v of %m")
            self.progressBar.setTextVisible(True)
            for sp in range(numberOfPlots):
                topLevelItem = self.plotsTreeWidget.topLevelItem(sp)
                data_type = topLevelItem.text(0)
                units_prefix = get_units_prefix(data_type)
                numberOfData = topLevelItem.childCount()
                if numberOfData < 1:
                    continue
                if len(ax):
                    current_ax = self.dataFig.add_subplot(numberOfPlots,1,sp+1,
                                                          sharex=ax[0])
                else:
                    current_ax = self.dataFig.add_subplot(numberOfPlots,1,sp+1)
                ax.append(current_ax)
                previous_units = None
                for n in range(numberOfData):
                    self.progressBar.setValue(current_num_data)
                    current_num_data += 1
                    QApplication.flush()
                    child = topLevelItem.child(n)
                    site = child.text(2).split(',')[0]
                    channels = self.parseChannels(child.text(3))
                    project = child.text(1).split(',')[0]
                    archive = child.text(0).split(',')[0]
                    ainfo = ap.get_archive_info(project,site,
                                                data_type,archive)[1]
                    all_channels = ainfo['channels']
                    channels = [c for c in channels if c in all_channels]
                    try:
                        for chan in channels:
                            md = ap.load_data(project, site, data_type, st, et,
                                              archive=archive,channels=[chan])
                            if md is not None and md.data.size:
                                if integration_interval != md.nominal_cadence:
                                    logger.info("Integrating...")
                                    md.set_cadence(integration_interval,inplace=True)
                                    logger.info("Finished integrating")
                                else:
                                    md = md.mark_missing_data(cadence=\
                                                          2*md.nominal_cadence)
                                if (not previous_units is None and
                                    md.units != previous_units):
                                    logger.warning("Data have different units.")
                                    continue
                                previous_units = md.units
                                self.statusBar().showMessage("Plotting data...")
                                md.plot(units_prefix = units_prefix,axes = current_ax,
                                        label="/".join([project,site,chan]))                                    
                                self.dataCanvas.draw()
                                self.statusBar().showMessage("Ready.")
                            else:
                                logger.info("No data.")
                                self.statusBar().showMessage("No data.")
                        if not previous_units is None:
                            current_ax.set_ylabel("".join([data_type,', ',
                                                   units_prefix,previous_units]))
                        else:
                            current_ax.set_ylabel(data_type)
                        self.dataCanvas.draw()
                    except Exception as e:
                        logger.info('Loading/plotting data failed')
                        logger.debug(str(e))
                        logger.debug(traceback.format_exc())
                current_ax.legend()
                if (sp+1)<numberOfPlots:
                    current_ax.set_xlabel(" ")
                self.dataCanvas.draw()
        except Exception as e:
            logger.info('Draw failed')
            logger.debug(str(e))
            logger.debug(traceback.format_exc())
        finally:
            self.progressBar.setValue(0)
            self.progressBar.setMaximum(1)
            self.progressBar.setTextVisible(False)
            QApplication.restoreOverrideCursor()
            QApplication.flush()
            self.timer.start(1000)

class CustomLogHandler(logging.Handler):
    def __init__(self,the_log):
        super(CustomLogHandler,self).__init__()
        self.the_log = the_log
    def emit(self,record):
        msg = self.format(record)
        self.the_log.append(msg)

class Log:
    def __init__(self,textEditWidget):
        self.saveDir = QDir.home() # folder for save dialog on first run
        self.textEditWidget = textEditWidget
        self.textEditWidget.appendPlainText("".join(["Log started at  ",
            "{:%X %Z %a %d %B %Y}".format(datetime.datetime.now())]))
    def append(self,logText):
        self.textEditWidget.appendPlainText(logText)
    def clear(self):
        msgBox = QMessageBox()
        msgBox.setText("Really clear the log?")
        msgBox.setWindowTitle(" ")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        ret = msgBox.exec_()
        if ret == QMessageBox.Yes:
            self.textEditWidget.clear()
            self.textEditWidget.appendPlainText("".join(["Log started at  ",
                    "{:%X %Z %a %d %B %Y}".format(datetime.datetime.now())]))
    def save(self):
        fileBox = QFileDialog()
        fileBox.setAcceptMode(QFileDialog.AcceptSave)
        fileBox.setDirectory(self.saveDir)
        ret = fileBox.exec_()
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

def dt64_to_Qdate_Qtime(dt64):
    d = dt64.astype(datetime.datetime)
    Qdate = QDate(d.year,d.month,d.day)
    Qtime = QTime(d.hour,d.minute,d.second,d.microsecond/1000.0)
    return Qdate,Qtime

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

