#!/bin/bash
pyside-uic -o ui_auroraplot.py auroraplot.ui
pyside-uic -o ui_about.py about.ui
pyside-rcc -o resources_rc.py resources.qrc
