# -*- coding: utf-8 -*-
import sys
import os
from PyQt5 import Qt, QtCore, QtGui, QtWidgets
#from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QMenu, QMenuBar, QStatusBar, QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QProgressBar, QFileDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget, QScrollArea, QSlider, QMessageBox
#from PyQt5.QtCore import pyqtSlot
from pathlib import Path
import platform
from functools import partial
import csv
import xlrd
from  datetime import datetime
import time
import pyqtgraph as pg
import algo
import json
import numpy as np

#import corrsWidget as cw

path = Path('./')
APP_TITLE = '运行分析AI'

if platform.system() == 'Windows':
    font = '微软雅黑'
    newline = ''
    encoding = 'gbk'
else:
    font = 'Lucida Grande'
    newline = '\n'
    encoding = 'utf-8'

if not os.path.isdir(path / 'data'):
    os.mkdir(path / 'data')

if not os.path.isdir(path / 'data' / 'database'):
    os.mkdir(path / 'data' / 'database')

if not os.path.isdir(path / 'config'):
    os.mkdir(path / 'config')
    
if not os.path.isdir(path / 'output'):
    os.mkdir(path / 'output')

StyleSheet = '''
/*设置红色进度条*/
#RedProgressBar {
    text-align: center; /*进度值居中*/
}
#RedProgressBar::chunk {
    background-color: #97c8ff;
}

QScrollArea { background: transparent; border:0px}
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollArea > QWidget > QScrollBar { background: palette(scrollbar); }



'''


class Ui_Juno(object):
    def setupUi(self, Juno):
        #Juno is the QMainWindow object
        #Juno.resize(1230, 800)
        Juno.setWindowTitle(APP_TITLE)
        Juno.setMinimumSize(1300, 600)
        self.centralWidget = QWidget(Juno)
        
        
        
        self.menubar = QMenuBar(Juno)
        #self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        
        self.menuData = QMenu(self.menubar)
        self.menuData.setTitle('数据')
        self.menuData.addAction('导入运行数据', partial(self.openFileNameDialog, Juno))
        
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setTitle('分析')
        
        self.menuCommand = QMenu(self.menubar)
        self.menuCommand.setTitle('指令')
        Juno.setMenuBar(self.menubar)
        
        self.statusbar = QStatusBar(Juno)
        Juno.setStatusBar(self.statusbar)
        
        self.menubar.addAction(self.menuData.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuCommand.menuAction())
        
        ##################################
        ##### Tab Widgets
        #mainFrame = QWidget(self.centralWidget)
        #mainhbox = QHBoxLayout(mainFrame)
        
        
        self.corrsTab = QWidget()        
        #self.corrsTab.setAutoFillBackground(True)
        #self.corrsTab.setMinimumSize(QtCore.QSize(1000,900))
        self.corrsConfigTab = QWidget()
        
        self.tabs = QTabWidget(self.centralWidget)
        Juno.setCentralWidget(self.tabs)
        
        
        self.tabs.addTab(self.corrsTab, '关系探索')
        self.tabs.addTab(self.corrsConfigTab, '关系配置')
        #mainhbox.addStretch(1)
        #mainhbox.addWidget(self.tabs)
        
        
        #self.tabs.setMinimumSize(QtCore.QSize(1800,900))
        
        ##################
        #### corrsTab Widget
        
        
        mainvlay = QVBoxLayout(self.corrsTab)
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        
        scrollAreaWidgetContents = QWidget()
        #scrollAreaWidgetContents.setStyleSheet("background-color:white;")
        scrollArea.setWidget(scrollAreaWidgetContents)
        
        
        
        #scrollArea.setAutoFillBackground(True)
        #
        #scrollArea.setStyleSheet("QScrollArea {background-color:white;}");
        #scrollArea.setStyleSheet("QComboBox {background-color:blue;}");
        #scrollArea.setStyleSheet("QScrollArea {background-color:transparent;}");
        #Juno.setStyleSheet(ss)

        mainvlay.addWidget(scrollArea)
        
        mainhlay = QHBoxLayout(scrollAreaWidgetContents)
        corrsGraphWidget = QWidget()
        corrsTableWidget = QWidget()
        mainhlay.addWidget(corrsGraphWidget)
        mainhlay.addWidget(corrsTableWidget)
        
        
        
        #scroll1 = QScrollArea()
        #scroll1.setWidgetResizable(True)
        
        #self.corrsTab.setCentralWidget(scroll1)
        
        ####################
        ### corrsTab - corrsGraphWidget
        
        leftvlay = QVBoxLayout(corrsGraphWidget)
        
        controlsWidget1 = QWidget()
        
        graphBox = QWidget()
        graphHBoxLayout = QHBoxLayout(graphBox)
        
        self.ysliderLabel = QLabel()
        self.ysliderLabel.setText('y指标偏好\n0')
        
        self.graphYslider = QSlider(QtCore.Qt.Vertical)
        self.graphYslider.setValue(0)
        self.graphYslider.setMinimum(-100)
        self.graphYslider.setMaximum(100)
        self.graphYslider.setFixedHeight(150)
        self.graphYslider.setTickPosition(QSlider.TicksRight)
        self.graphYslider.setTickInterval(25)
        self.graphYslider.setSingleStep(25)
        self.graphYslider.valueChanged.connect(partial(self.graphSliderChanged,1))
        
        self.graphWidget = pg.PlotWidget(corrsGraphWidget)
        self.graphWidget.setMinimumHeight(300)
        self.graphWidget.setMinimumWidth(500)
        
        graphHBoxLayout.addWidget(self.ysliderLabel)
        graphHBoxLayout.addWidget(self.graphYslider)
        graphHBoxLayout.addWidget(self.graphWidget)
        
        graphXsliderBox = QWidget()
        graphXsliderBoxLayout = QHBoxLayout(graphXsliderBox)
        graphXsliderBoxLayout.addStretch()
        
        
        self.xsliderLabel = QLabel()
        self.xsliderLabel.setText('x指标偏好\n0')
        
        self.graphXslider = QSlider(QtCore.Qt.Horizontal)
        self.graphXslider.setValue(0)
        self.graphXslider.setMinimum(-100)
        self.graphXslider.setMaximum(100)
        self.graphXslider.setFixedWidth(150)
        self.graphXslider.setTickPosition(QSlider.TicksBelow)
        self.graphXslider.setTickInterval(25)
        self.graphXslider.setSingleStep(25)
        self.graphXslider.valueChanged.connect(partial(self.graphSliderChanged,1))
        
        graphXsliderBoxLayout.addWidget(self.xsliderLabel)
        graphXsliderBoxLayout.addWidget(self.graphXslider)
        graphXsliderBoxLayout.addStretch()
        
        
        
        analysisWidget = QWidget()
        
        leftvlay.addWidget(controlsWidget1)
        leftvlay.addWidget(graphBox)
        leftvlay.addWidget(graphXsliderBox)
        leftvlay.addWidget(analysisWidget)
        
        self.varSelectionForm = QFormLayout(controlsWidget1)
        #self.varSelectionForm.setGeometry(QtCore.QRect(330, 25, 400, 100))
        
        if os.path.isfile(path / 'config' / 'corrs.json'):
            with open(path / 'config' / 'corrs.json', 'r') as f:
                self.corrsConfig = json.loads(f.read())
        else:
            self.corrsConfig = None
            print(self.corrsConfig)
        
        self.combo1 = QComboBox()
        self.combo2 = QComboBox()
        
        self.varSelectionForm.addRow("x指标", self.combo1)
        self.varSelectionForm.addRow("y指标", self.combo2)
        
        
        self.combo1.setFixedWidth(300)
        self.combo2.setFixedWidth(300)
        
        self.combo1.activated.connect(partial(self.selectionchange, 0,0, 0))
        self.combo2.activated.connect(partial(self.selectionchange, 0,0, 0))
        
        
        self.statusbar.showMessage('准备就绪', 2000)
        
        
        self.graphWidget.setBackground('#ffffff')
        self.graphWidget.showGrid(x=True, y=True)
        #self.graphWidget.setGeometry(QtCore.QRect(80,200, 620, 400))
        #label_style = {"color": "#999", "font-size": "8pt"}
        graphFont=QtGui.QFont()
        graphFont.setPixelSize(10)
        self.graphWidget.getAxis('bottom').setStyle(tickFont = graphFont)
        self.graphWidget.getAxis('left').setStyle(tickFont = graphFont)
        
        self.graphWidget.getAxis("left").setLabel(**{"color": "#999", "font-size": "8pt"})
        self.graphWidget.getAxis("bottom").setLabel(**{"color": "#999", "font-size": "8pt"})
        self.graphWidget.setTitle("关系图", **{'color': '#777', 'size': '9pt', 'justify':'left'})
        
        analysisvlayout = QVBoxLayout(analysisWidget)
        
        
        self.dataFilenameLabel = QLabel()
        #self.dataFilenameLabel.setGeometry(QtCore.QRect(80, 150, 600, 30))
        self.dataFilenameLabel.setText('未加载任何数据')
                
        self.r2sLabel = QLabel(analysisWidget)
        #self.r2sLabel.setGeometry(QtCore.QRect(80, 630, 500, 20))
        self.r2sLabel.setText("R2判定系数: ")
        
        self.r2sDescLabel = QLabel()
        #self.r2sDescLabel.setGeometry(QtCore.QRect(80, 670, 500, 20))
        self.r2sDescLabel.setText("关系评价: " )
        
        analysisvlayout.addWidget(self.dataFilenameLabel)
        analysisvlayout.addWidget(self.r2sLabel)
        analysisvlayout.addWidget(self.r2sDescLabel)
        
        ###########################
        ### corrsTab - corrsTableWidget
        rightvlay = QVBoxLayout(corrsTableWidget)
        
        controlsWidget2 = QWidget()
        self.progressBar = QProgressBar(corrsTableWidget, objectName='RedProgressBar')
        self.corrsTable = QTableWidget(corrsTableWidget)
        
        rightvlay.addWidget(controlsWidget2)
        rightvlay.addWidget(self.progressBar)
        rightvlay.addWidget(self.corrsTable)
        
        self.varSelectionForm2 = QFormLayout(controlsWidget2)

        self.combo3 = QComboBox()
        self.combo3.activated.connect(self.corrsSelectionChange)
        self.combo3.setFixedWidth(300)
        self.varSelectionForm2.addRow("y指标",self.combo3 )
        
        if os.path.isfile(path / 'config' / 'data.config'):
            try:
                with open(path / 'config' / 'data.config', 'r', encoding='utf-8') as f:
                    self.dataFilename = f.read()
            except UnicodeDecodeError:
                with open(path / 'config' / 'data.config', 'r', encoding='gbk') as f:
                    self.dataFilename = f.read()
            self.dataFilenameLabel.setText('加载数据：' + self.dataFilename)
            
        if os.path.isfile(path / 'data' / 'database' / 'data.json'):
            with open(path / 'data' / 'database' / 'data.json', 'r') as f:
                self.data = json.loads(f.read())
                
            self.combo1.addItems(self.data.keys())
            self.combo2.addItems(self.data.keys())
            self.selectionchange()
            self.combo3.addItems(self.data.keys())
            
        if os.path.isfile(path / 'data' / 'database' / 'pools.json'):
            with open(path / 'data' / 'database' / 'pools.json', 'r') as f:
                self.pools = json.loads(f.read())
        
        
            
        
        #self.corrsTable.setGeometry(QtCore.QRect(750,125, 450, 575))
        #self.corrsTable.setFixedWidth(600)
        self.corrsTable.setRowCount(4)  
        self.corrsTable.setColumnCount(2)   
        self.corrsTable.setHorizontalHeaderLabels(['x指标', 'R2判定系数'])
        self.corrsTable.setSortingEnabled(True)
        self.corrsTable.resizeColumnsToContents()
        self.corrsTable.resizeRowsToContents()
        self.corrsTable.clicked.connect(self.tableClick)
        
        
        ##############
        ##### corrsConfigTab Widget
        
        corrsConfigVBoxLayout = QVBoxLayout(self.corrsConfigTab)
        label = QLabel()
        label.setText('指标优化方向偏好：')
        corrsConfigVBoxLayout.addWidget(label)
        
        label = QLabel()
        label.setText('正数代表该指标越高越好，负数代表越低越好，0则代表中立。\n偏好绝对值代表偏好程度，该值越大则拟合线将越往数据边界靠拢，越小则拟合线越往数据中心靠拢。')
        corrsConfigVBoxLayout.addWidget(label)
        
        hboxlayout = QWidget()
        controlsHBoxLayout = QHBoxLayout(hboxlayout)
        corrsConfigVBoxLayout.addWidget(hboxlayout)
        
        resetAllButton = QPushButton()
        resetAllButton.setText('重置全部')
        resetAllButton.setFixedWidth(100)
        resetAllButton.clicked.connect(self.resetAll)
        controlsHBoxLayout.addWidget(resetAllButton)
        
        saveConfigButton = QPushButton()
        saveConfigButton.setText('保存配置')
        saveConfigButton.setFixedWidth(100)
        saveConfigButton.clicked.connect(self.saveConfig)
        controlsHBoxLayout.addWidget(saveConfigButton)
        
        controlsHBoxLayout.addStretch()
        
        self.corrsConfigTable = QTableWidget(self.corrsConfigTab)
        
        corrsConfigVBoxLayout.addWidget(self.corrsConfigTable)
        
        #self.corrsConfigTable.setFixedWidth(1000)
         
        self.corrsConfigTable.setColumnCount(6)   
        self.corrsConfigTable.setHorizontalHeaderLabels(['指标', '偏好', '偏好系数','', '',''])
        
        
        self.corrsConfigTable.setRowCount(len(self.data)) 
        self.corrsConfigTable.setRowHeight(30,30)
        self.sliders = []
        for i, v in enumerate(self.data):
            self.corrsConfigTable.setItem(i,0, QTableWidgetItem(v))
            
            slider = QSlider(QtCore.Qt.Horizontal)
            self.corrsConfigTable.setCellWidget(i,1, slider)
            
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setMinimum(-100)
            slider.setMaximum(100)
            if self.corrsConfig:
                slider.setValue(self.corrsConfig[v])
                item = QTableWidgetItem(str(self.corrsConfig[v]))
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.corrsConfigTable.setItem(i, 2, item)
            else:
                slider.setValue(0)
                item = QTableWidgetItem(str(0))
                item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
                self.corrsConfigTable.setItem(i, 2, item)
            slider.setTickInterval(25)
            slider.setSingleStep(25)
            slider.valueChanged.connect(partial(self.sliderValueChanged,i))
            self.sliders.append(slider)
            
            resetBtn = QPushButton()
            resetBtn.setText('重置')
            resetBtn.clicked.connect(partial(self.resetRow, i))
            self.corrsConfigTable.setCellWidget(i,3, resetBtn)
            
            minBtn = QPushButton()
            minBtn.setText('最小')
            minBtn.clicked.connect(partial(self.minRow, i))
            self.corrsConfigTable.setCellWidget(i,4, minBtn)
            
            maxBtn = QPushButton()
            maxBtn.setText('最大')
            maxBtn.clicked.connect(partial(self.maxRow, i))
            self.corrsConfigTable.setCellWidget(i,5, maxBtn)
            
        #self.corrsConfigTable.setSortingEnabled(True)
        self.corrsConfigTable.resizeColumnsToContents()
        self.corrsConfigTable.resizeRowsToContents()
    
    def graphSliderChanged(self, trigger=0):
        
        xPolicy = self.graphXslider.value()
        yPolicy = self.graphYslider.value()
        #print(xPolicy, yPolicy)
        self.xsliderLabel.setText('x指标偏好\n' + str(xPolicy))
        self.ysliderLabel.setText('y指标偏好\n' + str(yPolicy))
        if trigger==1:
            self.selectionchange(xPolicy=xPolicy, yPolicy=yPolicy, trigger=1)
        
    
    def saveConfig(self):
        configDict = dict(zip(self.data.keys(), [slider.value() for i,slider in enumerate(self.sliders)]))
        print(configDict)
        with open(path / 'config' / 'corrs.json', 'w') as f:
            f.write(json.dumps(configDict))
        self.statusbar.showMessage('存储配置成功！', 2000) 
        with open(path / 'config' / 'corrs.json', 'r') as f:
            self.corrsConfig = json.loads(f.read())
        self.selectionchange(0,0,3)
        
        msg = QMessageBox()
        msg.setWindowTitle("通知")
        msg.setText("存储配置成功！")
        msg.setIcon(QMessageBox.Information)
        #msg.setStandardButtons(QMessageBox.Retry | QMessageBox.Ignore | QMessageBox.Cancel)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Cancel
        msg.exec_()
        ...
    
    def minRow(self, row):
        item = QTableWidgetItem(str(-100))
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.corrsConfigTable.setItem(row, 2, item)
        self.sliders[row].setValue(-100)  
    
    def maxRow(self, row):
        item = QTableWidgetItem(str(100))
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.corrsConfigTable.setItem(row, 2, item)
        self.sliders[row].setValue(100)  
    
    def sliderValueChanged(self,row):
        v = self.sliders[row].value()
        item = QTableWidgetItem(str(v))
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.corrsConfigTable.setItem(row, 2, item)
        
    def resetRow(self, row):
        item = QTableWidgetItem(str(0))
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.corrsConfigTable.setItem(row, 2, item)
        self.sliders[row].setValue(0)  
        
    def resetAll(self):
        msg = QMessageBox()
        msg.setWindowTitle("通知")
        msg.setText("确定重置所有指标偏好？")
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        #msg.setStandardButtons(QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)  # setting default button to Cancel
        buttonY = msg.button(QMessageBox.Ok)
        buttonY.setText('确定')
        buttonN = msg.button(QMessageBox.Cancel)
        buttonN.setText('取消')
        result = msg.exec_()
        
        if result ==  QMessageBox.Ok:
            for i, v in enumerate(self.data):
                self.resetRow(i)
    
        else:
            msg.done(1)
        #############################
    def openFileNameDialog(self, Juno):
        self.statusbar.showMessage("导入数据", 1000)
        self.statusbar.repaint()
        options = QFileDialog.Options()
        #options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(Juno, "选择运行数据", "./data" ,"xlsx文件 (*.xlsx);;xls文件 (*.xls);;所有文件 (*)", options=options)
        if fileName:
            print(fileName)
            self.statusbar.showMessage('正在读取表格...')
            self.statusbar.repaint()
            
        try:
            wb = xlrd.open_workbook(fileName, encoding_override="gbk")
        except UnicodeDecodeError:
            wb = xlrd.open_workbook(fileName, encoding_override="utf-8")

        except FileNotFoundError:
          return
      
        
        self.statusbar.showMessage('正在解析...')
        self.statusbar.repaint()
        
         # Xlsx2csv(filename, outputencoding="utf-8").convert(path / 'data' / 'data.csv')
        sheet = wb.sheet_by_index(0) 
        csv_file = open(path / 'data' / 'data.csv', 'w', encoding='utf-8')
        wr = csv.writer(csv_file, delimiter='\t')
        
        def is_number(n):
            try:
                float(n)   # Type-casting the string to `float`.
                           # If string is not a valid `float`, 
                           # it'll raise `ValueError` exception
            except ValueError:
                return False
            return True
        
        for rownum in range(sheet.nrows):
            #print(rownum)
            
            row = [str(v) for v in sheet.row_values(rownum)]
            #print(row)
            # try:
            #     print(int(row[0]))
            #     excel_date = int(row[0])
            #     print(excel_date)
            #     dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
            #     tt = dt.timetuple()
            #     row[0] = tt
            # except ValueError:
                
            #     ...
            
            if is_number(row[0]):
                excel_date = int(float(row[0]))
                print(excel_date)
                dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + excel_date - 2)
                
                row[0] = str(dt).split(' ')[0]
            
            
            print('row', row)
            wr.writerow(row)
        csv_file.close()   
        
        self.statusbar.showMessage('导入文件成功！', 2000)
        
        #writing into data.json so that next time the data automatically loads up.
        self.data, self.logs, self.pools = algo.readData(path / 'data' / 'data.csv')
        
        with open(path / 'data' / 'database' / 'data.json', 'w') as f:
            f.write(json.dumps(self.data))
        
        with open(path / 'data' / 'database' / 'pools.json', 'w') as f:
            f.write(json.dumps(self.pools))
            
        with open(path / 'data' / 'database' / 'logs.json', 'w') as f:
            f.write(json.dumps(self.logs))
        
        #writing config so that the data filename is shown.
        with open(path / 'config' / 'data.config', 'w', encoding='utf-8') as f:
            f.write(fileName)
        
        self.dataFilenameLabel.setText('加载数据：' + fileName)
        
        self.combo1.clear()
        self.combo2.clear()
        self.combo3.clear()

        self.combo1.addItems(self.data.keys())
        self.combo2.addItems(self.data.keys())
        self.combo3.addItems(self.data.keys())
        
    def selectionchange(self, xPolicy=0, yPolicy=0, trigger=0):
        #0 = default loading, 1 = slider, 2 = tableClick, 3 = configSave
        xvar = self.combo1.currentText()
        yvar = self.combo2.currentText()
        print('trigger', trigger)
        if self.corrsConfig and trigger!=1:
            xPolicy, yPolicy = self.corrsConfig[xvar], self.corrsConfig[yvar]
        
        
        print('policies', xPolicy, yPolicy)
        
        
        x = self.data[xvar] #
        y = self.data[yvar]
        
        T = []
        for t,v in x:
            #print(t)
            if t in [y[0] for y in y]:
                T.append(t)
        tx, ty = [x[1] for x in x if x[0] in T], [y[1] for y in y if y[0] in T]
        
        scatter = pg.ScatterPlotItem(size=5, brush=pg.mkBrush(0, 0, 0, 80)) 
        scatter.setData(pos=zip(tx,ty), alpha=0.5, name='历史数据')
        #scatter.addLabel('历史数据')
        self.graphWidget.clear()
        self.graphWidget.setTitle(xvar + " 和 " + yvar + " 之间的关系")
        self.graphWidget.setLabel('left', yvar)
        self.graphWidget.setLabel('bottom',xvar)
        self.graphWidget.addItem(scatter)
        
        p, r2s, zx, ypreds = algo.polyFit(x,y, xPolicy=xPolicy, yPolicy=yPolicy)
        
        curvePred = pg.PlotDataItem(pen={'color': "b", 'width': 2}) 
        curvePred.setData(x=zx, y=ypreds, name='机器拟合')
        #curvePred.addLabel('拟合规律')
        self.graphWidget.addItem(curvePred)
        self.graphWidget.addLegend(brush=(255, 255, 255, 120), labelTextColor='555', pen={'color': "ccc", 'width': 1})#pen={'color': "#000000", 'width': 1},
        
        
        
        
        desc = ['存在强关系。','有一定关系，存在其它干扰因素。', '关系弱。', '不存在关系。', '数据存在异常。']
        
        r2sType = '关系未知。'
        if r2s >= 0.5:
            r2sType = desc[0]
        elif 0.3 <= r2s < 0.5:
            r2sType = desc[1]
        elif 0.1 <= r2s < 0.3:
            r2sType = desc[2]
        elif 0 < r2s < 0.1:
            r2sType = desc[3]
        else:
            r2sType = desc[4]
        
        self.r2sLabel.setText("R2判定系数，越高越好(0~1): " + str(round(r2s,4)))
        self.r2sDescLabel.setText("关系评价: " + r2sType)
        
        print('r2 score', r2s)
        
        self.graphXslider.setValue(xPolicy)
        self.graphYslider.setValue(yPolicy)
    
    def corrsSelectionChange(self):
        self.statusbar.showMessage('建模查找所有关系...')
        self.statusbar.repaint()
        yvar = self.combo3.currentText()
        # results = algo.polyfitFeatureImportances(yvar, self.data)
        
        # self.corrsTable.clearContents()
        # self.corrsTable.setRowCount(len(results))  
        # for i,v in enumerate(results):
        #     self.corrsTable.setItem(i,0, QTableWidgetItem(v)) 
        #     self.corrsTable.setItem(i,1, QTableWidgetItem(str(results[v])))
        
        i = 0
        self.progressBar.setMaximum(len(self.data)-1)
        self.corrsTable.clearContents()
        self.corrsTable.setRowCount(len(self.data)-1)
        for v in self.data:
            if v == yvar:
                continue
            else:
                xPolicy, yPolicy = self.corrsConfig[v], self.corrsConfig[yvar]
                
                p, r2s, zx, ypreds = algo.polyFit(self.data[v], self.data[yvar], xPolicy=xPolicy, yPolicy=yPolicy)
                
                self.corrsTable.setItem(i,0, QTableWidgetItem(v)) 
                self.corrsTable.setItem(i,1, QTableWidgetItem(str(r2s)))
                i += 1
                
                self.corrsTable.resizeColumnsToContents()
                self.corrsTable.resizeRowsToContents()
                self.corrsTable.sortItems(1, QtCore.Qt.DescendingOrder)
                
                self.corrsTable.repaint()
                
                self.progressBar.setValue(i)
                
                
                
        self.statusbar.showMessage('建模完成')
        self.statusbar.repaint()
        
        #print(results)
    
    def tableClick(self):
        yvar = self.combo3.currentText()
        row = self.corrsTable.currentRow()
        xvar = self.corrsTable.item(row,0).text()
        print(xvar, 'selected')
        
        ind1 = list(self.data.keys()).index(xvar)
        ind2 = list(self.data.keys()).index(yvar)

        self.combo1.setCurrentIndex(ind1) 
        self.combo2.setCurrentIndex(ind2)
        self.selectionchange()
        
        
        
    
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    app.setStyleSheet(StyleSheet)
    #app.setAttribute(Qt.AA_Use96Dpi);
    custom_font = QtGui.QFont(font, 10)
    #custom_font.setWeight(18)
    custom_font.setPixelSize(15)
    app.setFont(custom_font)
    MainWindow = QMainWindow()
    window = Ui_Juno()
    window.setupUi(MainWindow)
    MainWindow.showMaximized()
    sys.exit(app.exec_())