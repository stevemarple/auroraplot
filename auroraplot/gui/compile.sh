#!/bin/bash
pyside-uic -o ui_auroraplot_launcher.py auroraplot_launcher.ui
pyside-uic -o ui_auroraplot_dataviewer.py auroraplot_dataviewer.ui
pyside-uic -o ui_about.py about.ui
pyside-rcc -o resources_rc.py resources.qrc
