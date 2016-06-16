# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'auroraplot_dataviewer.ui'
#
# Created: Thu Jun 16 09:04:53 2016
#      by: pyside-uic 0.2.15 running on PySide 1.2.2
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(987, 714)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/icons/dataviewer_32.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setIconSize(QtCore.QSize(32, 32))
        self.centralwidget = QtGui.QWidget(MainWindow)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.splitter_2 = QtGui.QSplitter(self.centralwidget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter = QtGui.QSplitter(self.splitter_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.verticalLayoutWidget = QtGui.QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.dataLayout = QtGui.QVBoxLayout(self.verticalLayoutWidget)
        self.dataLayout.setContentsMargins(0, 0, 0, 0)
        self.dataLayout.setObjectName("dataLayout")
        self.verticalLayoutWidget_2 = QtGui.QWidget(self.splitter)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.layoutWidget = QtGui.QWidget(self.splitter_2)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.toolBox = QtGui.QToolBox(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.toolBox.sizePolicy().hasHeightForWidth())
        self.toolBox.setSizePolicy(sizePolicy)
        self.toolBox.setMinimumSize(QtCore.QSize(420, 0))
        self.toolBox.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.toolBox.setObjectName("toolBox")
        self.page_0 = QtGui.QWidget()
        self.page_0.setGeometry(QtCore.QRect(0, 0, 420, 413))
        self.page_0.setObjectName("page_0")
        self.gridLayout = QtGui.QGridLayout(self.page_0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_7 = QtGui.QLabel(self.page_0)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 8, 0, 1, 1)
        self.channelsLineEdit = QtGui.QLineEdit(self.page_0)
        self.channelsLineEdit.setObjectName("channelsLineEdit")
        self.gridLayout.addWidget(self.channelsLineEdit, 10, 1, 1, 1)
        self.label_6 = QtGui.QLabel(self.page_0)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 10, 0, 1, 1)
        self.label_8 = QtGui.QLabel(self.page_0)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 0, 1, 1)
        self.addDatasetButton = QtGui.QPushButton(self.page_0)
        self.addDatasetButton.setObjectName("addDatasetButton")
        self.gridLayout.addWidget(self.addDatasetButton, 10, 2, 1, 1)
        self.dataTypeBox = QtGui.QComboBox(self.page_0)
        self.dataTypeBox.setObjectName("dataTypeBox")
        self.gridLayout.addWidget(self.dataTypeBox, 8, 1, 1, 2)
        self.plotTypeBox = QtGui.QComboBox(self.page_0)
        self.plotTypeBox.setObjectName("plotTypeBox")
        self.gridLayout.addWidget(self.plotTypeBox, 0, 1, 1, 1)
        self.projectBox = QtGui.QComboBox(self.page_0)
        self.projectBox.setObjectName("projectBox")
        self.gridLayout.addWidget(self.projectBox, 6, 1, 1, 2)
        self.label_4 = QtGui.QLabel(self.page_0)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 6, 0, 1, 1)
        self.label_5 = QtGui.QLabel(self.page_0)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 7, 0, 1, 1)
        self.plotsTreeWidget = QtGui.QTreeWidget(self.page_0)
        self.plotsTreeWidget.setObjectName("plotsTreeWidget")
        self.plotsTreeWidget.headerItem().setText(0, "1")
        self.gridLayout.addWidget(self.plotsTreeWidget, 1, 0, 1, 3)
        self.addPlotButton = QtGui.QPushButton(self.page_0)
        self.addPlotButton.setObjectName("addPlotButton")
        self.gridLayout.addWidget(self.addPlotButton, 0, 2, 1, 1)
        self.removeDatasetButton = QtGui.QPushButton(self.page_0)
        self.removeDatasetButton.setObjectName("removeDatasetButton")
        self.gridLayout.addWidget(self.removeDatasetButton, 2, 2, 1, 1)
        self.siteBox = QtGui.QComboBox(self.page_0)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.siteBox.sizePolicy().hasHeightForWidth())
        self.siteBox.setSizePolicy(sizePolicy)
        self.siteBox.setObjectName("siteBox")
        self.gridLayout.addWidget(self.siteBox, 7, 1, 1, 2)
        self.line = QtGui.QFrame(self.page_0)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 5, 0, 1, 3)
        self.archiveBox = QtGui.QComboBox(self.page_0)
        self.archiveBox.setObjectName("archiveBox")
        self.gridLayout.addWidget(self.archiveBox, 9, 1, 1, 2)
        self.label_9 = QtGui.QLabel(self.page_0)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 9, 0, 1, 1)
        self.toolBox.addItem(self.page_0, "")
        self.page_1 = QtGui.QWidget()
        self.page_1.setGeometry(QtCore.QRect(0, 0, 420, 413))
        self.page_1.setObjectName("page_1")
        self.gridLayout_2 = QtGui.QGridLayout(self.page_1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtGui.QLabel(self.page_1)
        self.label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 2, 1, 1, 1)
        self.durationUnitsBox = QtGui.QComboBox(self.page_1)
        self.durationUnitsBox.setObjectName("durationUnitsBox")
        self.gridLayout_2.addWidget(self.durationUnitsBox, 4, 3, 1, 1)
        self.calendarWidget = QtGui.QCalendarWidget(self.page_1)
        self.calendarWidget.setMinimumSize(QtCore.QSize(280, 250))
        self.calendarWidget.setMaximumSize(QtCore.QSize(280, 280))
        self.calendarWidget.setMouseTracking(False)
        self.calendarWidget.setMinimumDate(QtCore.QDate(1752, 9, 15))
        self.calendarWidget.setFirstDayOfWeek(QtCore.Qt.Monday)
        self.calendarWidget.setGridVisible(True)
        self.calendarWidget.setHorizontalHeaderFormat(QtGui.QCalendarWidget.SingleLetterDayNames)
        self.calendarWidget.setVerticalHeaderFormat(QtGui.QCalendarWidget.NoVerticalHeader)
        self.calendarWidget.setNavigationBarVisible(True)
        self.calendarWidget.setDateEditEnabled(True)
        self.calendarWidget.setObjectName("calendarWidget")
        self.gridLayout_2.addWidget(self.calendarWidget, 1, 1, 1, 3)
        self.starttimeWidget = QtGui.QTimeEdit(self.page_1)
        self.starttimeWidget.setTimeSpec(QtCore.Qt.UTC)
        self.starttimeWidget.setObjectName("starttimeWidget")
        self.gridLayout_2.addWidget(self.starttimeWidget, 2, 2, 1, 1)
        self.label_2 = QtGui.QLabel(self.page_1)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 4, 1, 1, 1)
        self.label_3 = QtGui.QLabel(self.page_1)
        self.label_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 2, 3, 1, 1)
        self.durationBox = QtGui.QSpinBox(self.page_1)
        self.durationBox.setObjectName("durationBox")
        self.gridLayout_2.addWidget(self.durationBox, 4, 2, 1, 1)
        self.widget = QtGui.QWidget(self.page_1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName("widget")
        self.gridLayout_2.addWidget(self.widget, 6, 2, 1, 1)
        self.widget_2 = QtGui.QWidget(self.page_1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setObjectName("widget_2")
        self.gridLayout_2.addWidget(self.widget_2, 0, 1, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 4, 1, 1)
        self.toolBox.addItem(self.page_1, "")
        self.page_2 = QtGui.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 420, 413))
        self.page_2.setObjectName("page_2")
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.page_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.optionsListWidget = QtGui.QListWidget(self.page_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.optionsListWidget.sizePolicy().hasHeightForWidth())
        self.optionsListWidget.setSizePolicy(sizePolicy)
        self.optionsListWidget.setMinimumSize(QtCore.QSize(100, 100))
        self.optionsListWidget.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.optionsListWidget.setObjectName("optionsListWidget")
        self.verticalLayout_2.addWidget(self.optionsListWidget)
        self.toolBox.addItem(self.page_2, "")
        self.page_3 = QtGui.QWidget()
        self.page_3.setGeometry(QtCore.QRect(0, 0, 420, 413))
        self.page_3.setObjectName("page_3")
        self.gridLayout_3 = QtGui.QGridLayout(self.page_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.logTextEdit = QtGui.QPlainTextEdit(self.page_3)
        self.logTextEdit.setObjectName("logTextEdit")
        self.gridLayout_3.addWidget(self.logTextEdit, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.logClearButton = QtGui.QPushButton(self.page_3)
        self.logClearButton.setObjectName("logClearButton")
        self.horizontalLayout_2.addWidget(self.logClearButton)
        self.logSaveButton = QtGui.QPushButton(self.page_3)
        self.logSaveButton.setObjectName("logSaveButton")
        self.horizontalLayout_2.addWidget(self.logSaveButton)
        self.gridLayout_3.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.toolBox.addItem(self.page_3, "")
        self.verticalLayout.addWidget(self.toolBox)
        self.groupBox = QtGui.QGroupBox(self.layoutWidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMinimumSize(QtCore.QSize(420, 100))
        self.groupBox.setMaximumSize(QtCore.QSize(1000, 100))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.goButton = QtGui.QPushButton(self.groupBox)
        self.goButton.setGeometry(QtCore.QRect(30, 20, 81, 61))
        self.goButton.setObjectName("goButton")
        self.autoUpdateCheckBox = QtGui.QCheckBox(self.groupBox)
        self.autoUpdateCheckBox.setGeometry(QtCore.QRect(140, 30, 111, 41))
        self.autoUpdateCheckBox.setObjectName("autoUpdateCheckBox")
        self.verticalLayout.addWidget(self.groupBox)
        self.gridLayout_4.addWidget(self.splitter_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 987, 30))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtGui.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExport_Data_Plot = QtGui.QAction(MainWindow)
        self.actionExport_Data_Plot.setObjectName("actionExport_Data_Plot")
        self.menuFile.addAction(self.actionExport_Data_Plot)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.label.setBuddy(self.starttimeWidget)

        self.retranslateUi(MainWindow)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Aurora Plot - Data Viewer", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("MainWindow", "Data type:", None, QtGui.QApplication.UnicodeUTF8))
        self.channelsLineEdit.setPlaceholderText(QtGui.QApplication.translate("MainWindow", "eg 0-3,5,7,10-12", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("MainWindow", "Channels:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("MainWindow", "Plot type:", None, QtGui.QApplication.UnicodeUTF8))
        self.addDatasetButton.setText(QtGui.QApplication.translate("MainWindow", "Add data set", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("MainWindow", "Project:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("MainWindow", "Site:", None, QtGui.QApplication.UnicodeUTF8))
        self.addPlotButton.setText(QtGui.QApplication.translate("MainWindow", "Add new plot", None, QtGui.QApplication.UnicodeUTF8))
        self.removeDatasetButton.setText(QtGui.QApplication.translate("MainWindow", "Remove", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("MainWindow", "Archive:", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_0), QtGui.QApplication.translate("MainWindow", "Data sets", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Start Time:", None, QtGui.QApplication.UnicodeUTF8))
        self.starttimeWidget.setDisplayFormat(QtGui.QApplication.translate("MainWindow", "HH:mm:ss", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("MainWindow", "Duration:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("MainWindow", "UTC", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1), QtGui.QApplication.translate("MainWindow", "Date and time", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), QtGui.QApplication.translate("MainWindow", "Plotting options", None, QtGui.QApplication.UnicodeUTF8))
        self.logClearButton.setText(QtGui.QApplication.translate("MainWindow", "Clear log", None, QtGui.QApplication.UnicodeUTF8))
        self.logSaveButton.setText(QtGui.QApplication.translate("MainWindow", "Save log", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), QtGui.QApplication.translate("MainWindow", "Log", None, QtGui.QApplication.UnicodeUTF8))
        self.goButton.setText(QtGui.QApplication.translate("MainWindow", "Draw", None, QtGui.QApplication.UnicodeUTF8))
        self.autoUpdateCheckBox.setText(QtGui.QApplication.translate("MainWindow", "Auto-update", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "&File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuHelp.setTitle(QtGui.QApplication.translate("MainWindow", "&Help", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAbout.setText(QtGui.QApplication.translate("MainWindow", "&About", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExport_Data_Plot.setText(QtGui.QApplication.translate("MainWindow", "&Export Data Plot", None, QtGui.QApplication.UnicodeUTF8))

import resources_rc
