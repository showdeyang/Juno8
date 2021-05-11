# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
from main import *

if __name__ == '__main__':
   
    
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app.setStyle('Fusion')
    # dark_stylesheet = qdarkstyle.load_stylesheet(qt_api='pyqt5')
    # app.setStyleSheet(dark_stylesheet)
    app.setStyleSheet(open('QSS\\texstudio.qss', encoding='utf-8').read())
    
    
    
    MainWindow = QMainWindow()
    window = JunoUI(MainWindow)
    
    MainWindow.show()
    app.exec_()
    app.quit()