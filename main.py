import sys
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QTextBrowser, QDateEdit, QHeaderView
from functools import partial
import xlrd
import xlsxwriter
from datetime import date
import pyqtgraph
import pyqtgraph.exporters
import json
import numpy as np
import algo

Version = 'Juno 2.1'
if not os.path.isdir('./JunoProject'):
    os.mkdir('./JunoProject')
if not os.path.isdir('./SourceTable'):
    os.mkdir('./SourceTable')


class JunoUI(object):
    def __init__(self, juno):
        self.central_widget = QWidget(juno)
        self.Main_Window(juno)

        self.value_temp = {}
        self.filename = ''

    def Main_Window(self, juno):
        juno.setWindowTitle(Version + ' - 项目')
        new_project_btn = QPushButton('新建项目')
        new_project_btn.clicked.connect(partial(self.new_project_window, juno))

        self.project_table = QTableWidget()
        self.project_table.setColumnCount(1)
        self.project_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.project_table.verticalHeader().setVisible(False)
        self.project_table.horizontalHeader().setVisible(False)
        self.project_table.clicked.connect(partial(self.task_window_pre, juno))

        self.main_content = QWidget()
        self.main_content.setMinimumSize(400, 300)
        main_layout = QHBoxLayout(self.main_content)
        main_layout.addWidget(new_project_btn)
        main_layout.addWidget(self.project_table)

        juno.setCentralWidget(self.main_content)
        juno.resize(400, 300)

        self.load_project_table()

    def load_project_table(self):
        project_list = os.listdir('./JunoProject')
        load_project_sort = {}
        for i in project_list:
            time = os.path.getatime('./JunoProject/' + i)
            load_project_sort.update({i: time})
        load_project_sort = sorted(load_project_sort.items(), key=lambda item: item[1], reverse=True)
        count = 0
        for i in load_project_sort:
            self.project_table.setRowCount(count + 1)
            item = QTableWidgetItem(i[0])
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.project_table.setItem(count, 0, item)
            count += 1

    def new_project_window(self, juno):
        juno.setWindowTitle(Version + ' - 新建项目')
        setName_label = QLabel('项目名称:')
        self.inputName = QTextEdit()
        self.inputName.textChanged.connect(self.input_change)
        input_name_content = QWidget()
        input_name_content.setFixedHeight(42)
        input_name_layout = QHBoxLayout(input_name_content)
        input_name_layout.addWidget(setName_label)
        input_name_layout.addWidget(self.inputName)

        self.selectFile = QPushButton('导入新表')
        self.selectFile.clicked.connect(partial(self.load_new_value, juno))

        self.errorText = QTextBrowser()

        noButton = QPushButton('取消')
        noButton.clicked.connect(partial(self.new_project_no, juno))
        self.yesButton = QPushButton('确定')
        self.yesButton.clicked.connect(partial(self.new_project_yes, juno))
        self.yesButton.setDisabled(True)
        YesOrNo_content = QWidget()
        YesOrNo_layout = QHBoxLayout(YesOrNo_content)
        YesOrNo_layout.addWidget(noButton)
        YesOrNo_layout.addWidget(self.yesButton)

        new_project_content = QWidget()
        new_project_content.setMinimumSize(400, 300)
        new_project_layout = QVBoxLayout(new_project_content)
        new_project_layout.addWidget(input_name_content)
        new_project_layout.addWidget(self.selectFile)
        new_project_layout.addWidget(self.errorText)
        new_project_layout.addWidget(YesOrNo_content)

        juno.setCentralWidget(new_project_content)

    def input_change(self):
        project_list = os.listdir('./JunoProject')
        if self.inputName.document().toRawText().strip() != '' and self.value_temp != {} and self.inputName.document().toRawText() not in project_list:
            self.yesButton.setDisabled(False)
        else:
            self.yesButton.setDisabled(True)

    def new_project_yes(self, juno):
        project_name = self.inputName.document().toRawText()
        if 'win' in sys.platform:
            os.system('mkdir ' + os.getcwd() + '\JunoProject' + '\\' + project_name)
            os.system('mkdir ' + os.getcwd() + '\JunoProject' + '\\' + project_name + '\\' + 'Task')
            os.system('copy ' + self.filename.replace('/', '\\') + ' ' + os.getcwd() + '\\' + 'JunoProject' + '\\' + project_name + '\\' + self.filename.split('/')[-1])
        elif 'linux' in sys.platform:
            cmd = ''

        with open('./JunoProject/' + project_name + '/value.json', 'w') as f:
            f.write(json.dumps(self.value_temp))

        # self.task_window(project_name, juno)
        self.Main_Window(juno)

    def new_project_no(self, juno):
        self.Main_Window(juno)

    def load_new_value(self, juno):
        option = QFileDialog.Option()
        option |= QFileDialog.DontUseNativeDialog
        self.filename, _ = QFileDialog.getOpenFileName(
            juno, '选择运行数据', './SourceTable',
            '所有文件 (*);;xlsx文件 (*.xlsx);;xls文件 (*.xls)',
            options=option
        )

        try:
            wb = xlrd.open_workbook(self.filename, encoding_override='gbk')
        except UnicodeDecodeError:
            wb = xlrd.open_workbook(self.filename, encoding_override='utf-8')
        except FileNotFoundError:
            self.errorText.setText('未读取数据!')
            return

        sheet = wb.sheet_by_index(0)

        self.value_temp = {}

        # 填充第三行的大名称
        last_name_1 = ''
        name_1 = []
        for i in sheet.row_values(2):
            if i != '':
                last_name_1 = i
            else:
                i = last_name_1
            name_1.append(i)

        # 填充第四行的小名称
        last_name_2 = ''
        name_2 = []
        for i in sheet.row_values(3):
            if '\n' in i:
                i = i.replace('\n', '')
            if '（' in i:
                i = i.replace('（', '(')
            if '）' in i:
                i = i.replace('）', ')')
            if i != '':
                last_name_2 = i
            else:
                i = last_name_2 + "%"
            name_2.append(i)
        name_2[0] = ''
        name_2[-1] = ''

        # 组合成 "大名称-小名称 单位"
        if len(name_1) == len(name_2):
            for i in range(len(name_1)):
                if name_2[i] == '':
                    str_name = name_1[i]
                else:
                    unit = sheet.row_values(4)[i]
                    if unit != '' and unit != '/':
                        str_name = str(name_1[i]) + "-" + str(name_2[i]) + " " + str(unit)
                    else:
                        str_name = str(name_1[i]) + "-" + str(name_2[i])
                self.value_temp.update({str_name: []})
        else:
            self.errorText.setText('表格第3、4行长度不一致!')
            return

        self.value_temp.pop('备注')

        # 将列数转换成AZ
        def col2az(col):
            # print(col)
            if col <= 26:
                return chr(col + ord("A"))
            a = chr(col // 26 - 1 + ord("A"))
            b = chr(col % 26 + ord("A"))
            return a + b

        # 数据格式纠正
        def value_rectify(row):
            # 将每行数据中的'-'、'--'、'\'、'/'等统一转换成''
            # 将每行数据中的','、'，'等统一转换成'.'
            for block_count in range(len(row)):
                if type(row[block_count]) == str and row[block_count] != '':
                    if ' ' in row[block_count]:
                        row[block_count] = row[block_count].strip()
                    if '..' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('..', '.'))
                    elif ',.' in row[block_count]:
                        row[block_count] = float(row[block_count].replace(',.', '.'))
                    elif ',' in row[block_count]:
                        row[block_count] = float(row[block_count].replace(',', '.'))
                    elif '，' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('，', '.'))
                    elif '。' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('。', '.'))
                    elif '．' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('．', '.'))
                    elif '<' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('<', '')) / 2.0
                    elif '《' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('《', '')) / 2.0
                    elif '＜' in row[block_count]:
                        row[block_count] = float(row[block_count].replace('＜', '')) / 2.0
                if row[block_count] == '-' or row[block_count] == '—' or row[block_count] == '——' or row[block_count] == '－' or row[block_count] == '—' or row[block_count] == '——' or row[block_count] == '-' or row[block_count] == '_' or row[block_count] == '－' or row[block_count] == '＿':
                    row[block_count] = ''
                if row[block_count] == '\\' or row[block_count] == '/' or row[block_count] == '＼' or row[block_count] == '／' or row[block_count] == '|' or row[block_count] == '｜':
                    row[block_count] = ''
                if row[block_count] == '.' or row[block_count] == '．':
                    row[block_count] = ''

            return row

        # 数据类型无法纠正的，给出警告
        def value_check(row_index, row_temp):
            col = -1
            error_per_row = ''
            for a in row_temp:
                col += 1
                if a == '':
                    continue
                elif type(a) != float and type(a) != int:
                    ord_list = []
                    for i in a:
                        ord_list.append(ord(i))
                    error = str(row_index + 1) + " 行 " + col2az(col) + " 列，参数 \'" + str(a) + " " + str(ord_list) + "\' 格式错误！\n"
                    error_per_row += error
            return error_per_row

        matrix = []
        error_total = ''
        # 将表格的数据部分组合成一个二维list，并趁机对每个数据进行格式检查
        for row_num in range(sheet.nrows):
            if row_num < 6:  # 抛弃表的前6行
                continue
            if sheet.row_values(row_num)[0] == '':
                break
            row = [i for i in sheet.row_values(row_num)]
            row = row[:-1]  # 去除最后一列备注

            row = value_rectify(row)
            error_total += value_check(row_num, row)
            matrix.append(row)
        # 如果数据存在格式错误，则弹窗告错
        if error_total != '':
            self.errorText.setText(error_total)
            self.errorText.append('\n文件导入失败!')

            self.value_temp = {}  # 清除已导入的错误数据
            return

        # 二维数据表xy转置成二维list: a天每天记录b种数据 变成 b种数据每种记录a天
        matrix_trans = np.array(matrix).transpose().tolist()

        enumerate_value = list(enumerate(self.value_temp))
        matrix_trans_temp = []
        name_filter = []
        index = -1
        for a in matrix_trans:
            index += 1
            s = ('').join(a).strip()
            if not s:
                name_filter.append(enumerate_value[index][1])
                continue
            matrix_trans_temp.append(a)
        print(self.value_temp)
        for a in name_filter:
            self.value_temp.pop(a)

        # 将转置后的二维list依次填进字典的每种"大名称-小名称 单位"中
        row_count = -1
        for i in self.value_temp:
            row_count += 1
            value_per_col = matrix_trans_temp[row_count]
            self.value_temp.update({i: value_per_col})

        self.errorText.setText('文件导入成功!')
        self.selectFile.setText(self.filename)
        self.input_change()

    def task_window_pre(self, juno):
        self.task_window(self.project_table.currentItem().text(), juno)

    def task_window(self, project_name, juno):
        juno.setWindowTitle(Version + ' - 命题')
        label = QLabel('添加命题: ')
        self.task_textedit = QTextEdit()
        self.task_textedit.textChanged.connect(partial(self.task_textedit_change, project_name))
        self.submit_task_btn = QPushButton('确定添加')
        self.submit_task_btn.clicked.connect(partial(self.task_submit, project_name))
        self.submit_task_btn.setDisabled(True)
        task_input_content = QWidget()
        task_input_content.setFixedHeight(42)
        task_input_layout = QHBoxLayout(task_input_content)
        task_input_layout.addWidget(label)
        task_input_layout.addWidget(self.task_textedit)
        task_input_layout.addWidget(self.submit_task_btn)

        self.task_table = QTableWidget()
        self.task_table.setColumnCount(1)
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.task_table.clicked.connect(partial(self.config_window, project_name, juno))
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.horizontalHeader().setVisible(False)

        task_back_btn = QPushButton('返回')
        task_back_btn.clicked.connect(partial(self.task_back_btn, juno))
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(task_back_btn)

        task_content = QWidget()
        task_content.setMinimumSize(400, 300)
        task_layout = QVBoxLayout(task_content)
        task_layout.addWidget(task_input_content)
        task_layout.addWidget(self.task_table)
        task_layout.addWidget(flow_content)

        with open('./JunoProject/' + project_name + '/value.json', 'r') as f:
            self.value = json.load(f)

        juno.setCentralWidget(task_content)
        juno.resize(400, 300)
        self.load_task_table(project_name)

    def task_textedit_change(self, project_name):
        task_name = self.task_textedit.document().toRawText().strip()
        task_list = os.listdir('./JunoProject/' + project_name + '/Task/')
        if task_name == '':
            self.submit_task_btn.setDisabled(True)
            return
        if task_name + '.json' in task_list:
            self.submit_task_btn.setDisabled(True)
            return
        self.submit_task_btn.setDisabled(False)

    def task_submit(self, project_name):
        task_name = self.task_textedit.document().toRawText()
        var = {'situation': [], 'action': [], 'result': [], 'cost': {}, 'risk': {}, 'border': {}, 'begin': None, 'end': None, 'safety': "0.5"}
        with open('./JunoProject/' + project_name + '/Task/' + task_name + '.json', 'w') as f:
            f.write(json.dumps(var))
        self.task_textedit.clear()
        self.load_task_table(project_name)

    def load_task_table(self, project_name):
        task_list = os.listdir('./JunoProject/' + project_name + '/Task/')
        load_task_sort = {}
        for i in task_list:
            time = os.path.getatime('./JunoProject/' + project_name + '/Task/' + i)
            load_task_sort.update({i: time})
        load_task_sort = sorted(load_task_sort.items(), key=lambda item: item[1], reverse=True)
        count = 0
        for i in load_task_sort:
            self.task_table.setRowCount(count + 1)
            item = QTableWidgetItem(i[0].split('.')[0])
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.task_table.setItem(count, 0, item)
            count += 1

    def task_back_btn(self, juno):
        self.Main_Window(juno)

    def config_window(self, project_name, juno, *task_name_1):
        juno.setWindowTitle(Version + ' - 配置参数')

        try:
            task_name = self.task_table.currentItem().text()
        except:
            task_name = task_name_1[0]

        self.situation_combo = QComboBox()
        self.situation_combo.addItems(self.value.keys())
        self.situation_combo.activated.connect(self.situation_combo_change)
        self.situation_table = QTableWidget()
        self.situation_table.setColumnCount(2)
        self.situation_table.verticalHeader().setVisible(False)
        self.situation_table.setHorizontalHeaderLabels(['指标', ''])
        self.situation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        situation_content = QWidget()
        situation_layout = QFormLayout(situation_content)
        situation_layout.addRow("情况选择 ", self.situation_combo)
        situation_layout.addRow(self.situation_table)

        self.action_combo = QComboBox()
        self.action_combo.addItems(self.value.keys())
        self.action_combo.activated.connect(self.action_combo_change)
        self.action_table = QTableWidget()
        self.action_table.setColumnCount(2)
        self.action_table.verticalHeader().setVisible(False)
        self.action_table.setHorizontalHeaderLabels(['指标', ''])
        self.action_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        action_content = QWidget()
        action_layout = QFormLayout(action_content)
        action_layout.addRow("行为选择 ", self.action_combo)
        action_layout.addRow(self.action_table)

        self.result_combo = QComboBox()
        self.result_combo.addItems(self.value.keys())
        self.result_combo.activated.connect(self.result_combo_change)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setHorizontalHeaderLabels(['指标', ''])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_content = QWidget()
        result_layout = QFormLayout(result_content)
        result_layout.addRow("结果选择 ", self.result_combo)
        result_layout.addRow(self.result_table)

        var_content = QWidget()
        var_layout = QHBoxLayout(var_content)
        var_layout.addWidget(situation_content)
        var_layout.addWidget(action_content)
        var_layout.addWidget(result_content)

        cost_table_label = QLabel('成本占比:\n')
        self.cost_table = QTableWidget()
        self.cost_table.setColumnCount(2)
        self.cost_table.verticalHeader().setVisible(False)
        self.cost_table.setHorizontalHeaderLabels(['行为变量', '成本占比 %'])
        self.cost_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cost_table_content = QWidget()
        cost_table_layout = QVBoxLayout(cost_table_content)
        cost_table_layout.addWidget(cost_table_label)
        cost_table_layout.addWidget(self.cost_table)

        config_content_1 = QWidget()
        config_layout_1 = QHBoxLayout(config_content_1)
        config_layout_1.addWidget(var_content)
        config_layout_1.addWidget(cost_table_content)

        risk_table_label = QLabel('重要性占比:\n')
        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(2)
        self.risk_table.verticalHeader().setVisible(False)
        self.risk_table.setHorizontalHeaderLabels(['结果变量', '重要性占比 %'])
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.risk_table.clicked.connect(self.risk_to_border)
        risk_table_content = QWidget()
        risk_table_layout = QVBoxLayout(risk_table_content)
        risk_table_layout.addWidget(risk_table_label)
        risk_table_layout.addWidget(self.risk_table)

        border_table_label = QLabel('边界设置:\n')
        self.border_add_btn = QPushButton('添加边界')
        self.border_add_btn.clicked.connect(self.border_table_add_row)
        self.border_add_btn.setDisabled(True)
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.addWidget(border_table_label)
        layout.addWidget(self.border_add_btn)
        self.border_table = QTableWidget()
        self.border_table.setColumnCount(3)
        self.border_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.border_table.cellChanged.connect(self.borber_table_cell_change)
        self.border_table.verticalHeader().setVisible(False)
        self.border_table.horizontalHeader().setVisible(False)
        self.border_table_graph = pyqtgraph.GraphicsLayoutWidget()
        self.border_table_graph.setBackground('#ffffff')
        self.border_table_graph_plt = self.border_table_graph.addPlot()
        border_table_content = QWidget()
        border_table_layout = QVBoxLayout(border_table_content)
        border_table_layout.addWidget(content)
        border_table_layout.addWidget(self.border_table)
        border_table_layout.addWidget(self.border_table_graph)

        config_content_2 = QWidget()
        config_layout_2 = QHBoxLayout(config_content_2)
        config_layout_2.addWidget(risk_table_content)
        config_layout_2.addWidget(border_table_content)

        begin_label = QLabel('起始时间')
        self.date_begin = QDateEdit()
        self.date_begin.setCalendarPopup(True)
        self.date_begin.setDate(date.fromordinal(int(float(min(self.value['日期']))) + date(1900, 1, 1).toordinal() - 2))
        end_label = QLabel('结束时间')
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(date.fromordinal(int(float(max(self.value['日期']))) + date(1900, 1, 1).toordinal() - 2))
        label = QLabel('safety')
        self.safety_textedit = QTextEdit('0.5')
        content_1 = QWidget()
        content_1.setFixedHeight(42)
        layout_1 = QHBoxLayout(content_1)
        layout_1.addWidget(begin_label)
        layout_1.addWidget(self.date_begin)
        layout_1.addWidget(end_label)
        layout_1.addWidget(self.date_end)
        layout_1.addWidget(label)
        layout_1.addWidget(self.safety_textedit)

        back_btn = QPushButton('返回')
        back_btn.clicked.connect(partial(self.config_back, project_name, juno))
        clean_btn = QPushButton('清空')
        # clean_btn.clicked.connect()
        save_btn = QPushButton('保存')
        save_btn.clicked.connect(partial(self.config_save, project_name, task_name))
        finish_btn = QPushButton('计算')
        finish_btn.clicked.connect(partial(self.config_calc, juno, project_name, task_name))
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(back_btn)
        flow_layout.addWidget(clean_btn)
        flow_layout.addWidget(save_btn)
        flow_layout.addWidget(finish_btn)

        config_content = QWidget()
        config_layout = QVBoxLayout(config_content)
        config_layout.addWidget(config_content_1)
        config_layout.addWidget(config_content_2)
        config_layout.addWidget(content_1)
        config_layout.addWidget(flow_content)

        juno.setCentralWidget(config_content)
        juno.showMaximized()

        self.load_config_window(project_name, task_name)

    def load_config_window(self, project_name, task_name):
        with open('./JunoProject/' + project_name + '/Task/' + task_name + '.json', 'r') as f:
            self.config_data = json.load(f)
            self.situation_data = self.config_data['situation']         # 指向 json 指定部分内容的指针
            self.action_data = self.config_data['action']               # 指向 json 指定部分内容的指针
            self.result_data = self.config_data['result']               # 指向 json 指定部分内容的指针
            self.cost_data = self.config_data['cost']                   # 指向 json 指定部分内容的指针
            self.risk_data = self.config_data['risk']                   # 指向 json 指定部分内容的指针
            self.border_data = self.config_data['border']               # 指向 json 指定部分内容的指针
            self.begin_data = self.config_data['begin']
            self.end_data = self.config_data['end']
            self.safety_data = self.config_data['safety']

        for i in self.situation_data:
            self.situation_combo.setCurrentText(i)
            self.situation_combo_change()
        for i in self.action_data:
            self.action_combo.setCurrentText(i)
            self.action_combo_change()
        for i in self.result_data:
            self.result_combo.setCurrentText(i)
            self.result_combo_change()

        if self.begin_data is not None:
            self.date_begin.setDate(date.fromordinal(int(float(self.begin_data)) + date(1900, 1, 1).toordinal() - 2))
        if self.end_data is not None:
            self.date_end.setDate(date.fromordinal(int(float(self.end_data)) + date(1900, 1, 1).toordinal() - 2))
        if self.safety_data is not None:
            self.safety_textedit.setText(self.safety_data)

    def situation_combo_change(self):
        situation_current = self.situation_combo.currentText()
        situation = []
        for i in range(self.situation_combo.count()):
            if self.situation_combo.itemText(i) == situation_current:
                continue
            situation.append(self.situation_combo.itemText(i))
        self.situation_combo.clear()
        self.situation_combo.addItems(situation)
        self.action_combo.clear()
        self.action_combo.addItems(situation)
        self.result_combo.clear()
        self.result_combo.addItems(situation)

        item = QTableWidgetItem(situation_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.situation_table.insertRow(self.situation_table.rowCount())
        self.situation_table.setItem(self.situation_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删除')
        cancel_button.clicked.connect(self.situation_combo_cancel)
        self.situation_table.setCellWidget(self.situation_table.rowCount() - 1, 1, cancel_button)

    def situation_combo_cancel(self):
        index = self.situation_table.currentRow()
        name = self.situation_table.item(index, 0).text()
        self.situation_table.removeRow(index)

        count = 1
        for i in self.value.keys():
            if i == name:
                break
            count += 1
        situation = []
        for i in range(self.situation_combo.count()):
            count_i = 1
            for j in self.value.keys():
                if self.situation_combo.itemText(i) == j:
                    break
                count_i += 1
            if count_i < count:
                situation.append(self.situation_combo.itemText(i))
        situation.append(name)
        for i in range(self.situation_combo.count()):
            count_i = 1
            for j in self.value.keys():
                if self.situation_combo.itemText(i) == j:
                    break
                count_i += 1
            if count_i > count:
                situation.append(self.situation_combo.itemText(i))

        self.situation_combo.clear()
        self.situation_combo.addItems(situation)
        self.action_combo.clear()
        self.action_combo.addItems(situation)
        self.result_combo.clear()
        self.result_combo.addItems(situation)

    def action_combo_change(self):
        action_current = self.action_combo.currentText()
        action = []
        for i in range(self.action_combo.count()):
            if self.action_combo.itemText(i) == action_current:
                continue
            action.append(self.action_combo.itemText(i))
        self.situation_combo.clear()
        self.situation_combo.addItems(action)
        self.action_combo.clear()
        self.action_combo.addItems(action)
        self.result_combo.clear()
        self.result_combo.addItems(action)

        item = QTableWidgetItem(action_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.action_table.insertRow(self.action_table.rowCount())
        self.action_table.setItem(self.action_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删除')
        cancel_button.clicked.connect(self.action_combo_cancel)
        self.action_table.setCellWidget(self.action_table.rowCount() - 1, 1, cancel_button)

        item = QTableWidgetItem(action_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.cost_table.insertRow(self.cost_table.rowCount())
        self.cost_table.setItem(self.cost_table.rowCount() - 1, 0, item)

        item = QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.cost_table.setItem(self.cost_table.rowCount() - 1, 1, item)

        if action_current in self.cost_data:
            count = 0
            for i in self.action_data:
                item = QTableWidgetItem(self.cost_data[i])
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.cost_table.setItem(count, 1, item)
                count += 1

    def action_combo_cancel(self):
        index = self.action_table.currentRow()
        name = self.action_table.item(index, 0).text()

        self.action_table.removeRow(index)

        self.cost_table.removeRow(index)

        count = 1
        for i in self.value.keys():
            if i == name:
                break
            count += 1
        action = []
        for i in range(self.action_combo.count()):
            count_i = 1
            for j in self.value.keys():
                if self.action_combo.itemText(i) == j:
                    break
                count_i += 1
            if count_i < count:
                action.append(self.action_combo.itemText(i))
        action.append(name)
        for i in range(self.action_combo.count()):
            count_i = 1
            for j in self.value.keys():
                if self.action_combo.itemText(i) == j:
                    break
                count_i += 1
            if count_i > count:
                action.append(self.action_combo.itemText(i))

        self.situation_combo.clear()
        self.situation_combo.addItems(action)
        self.action_combo.clear()
        self.action_combo.addItems(action)
        self.result_combo.clear()
        self.result_combo.addItems(action)

    def result_combo_change(self):
        result_current = self.result_combo.currentText()
        result = []
        for i in range(self.result_combo.count()):
            if self.result_combo.itemText(i) == result_current:
                continue
            result.append(self.result_combo.itemText(i))
        self.situation_combo.clear()
        self.situation_combo.addItems(result)
        self.action_combo.clear()
        self.action_combo.addItems(result)
        self.result_combo.clear()
        self.result_combo.addItems(result)

        item = QTableWidgetItem(result_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.result_table.insertRow(self.result_table.rowCount())
        self.result_table.setItem(self.result_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删除')
        cancel_button.clicked.connect(self.result_combo_cancel)
        self.result_table.setCellWidget(self.result_table.rowCount() - 1, 1, cancel_button)

        item = QTableWidgetItem(result_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.risk_table.insertRow(self.risk_table.rowCount())
        self.risk_table.setItem(self.risk_table.rowCount() - 1, 0, item)

        item = QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.risk_table.setItem(self.risk_table.rowCount() - 1, 1, item)

        if result_current in self.risk_data:
            count = 0
            for i in self.risk_data:
                item = QTableWidgetItem(self.risk_data[i])
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.risk_table.setItem(count, 1, item)
                count += 1

    def result_combo_cancel(self):
        index = self.result_table.currentRow()
        name = self.result_table.item(index, 0).text()

        self.result_table.removeRow(index)

        self.risk_table.removeRow(index)

        count = 1
        for i in self.value.keys():
            if i == name:
                break
            count += 1
        result = []
        for i in range(self.result_combo.count()):
            count_i = 1
            for j in self.value.keys():
                if self.result_combo.itemText(i) == j:
                    break
                count_i += 1
            if count_i < count:
                result.append(self.result_combo.itemText(i))
        result.append(name)
        for i in range(self.result_combo.count()):
            count_i = 1
            for j in self.value.keys():
                if self.result_combo.itemText(i) == j:
                    break
                count_i += 1
            if count_i > count:
                result.append(self.result_combo.itemText(i))

        self.situation_combo.clear()
        self.situation_combo.addItems(result)
        self.action_combo.clear()
        self.action_combo.addItems(result)
        self.result_combo.clear()
        self.result_combo.addItems(result)

        if self.border_table.rowCount() > 0:
            if name == self.border_table.item(0, 0).text().split('--')[0]:
                self.border_table.clear()
                self.border_table.setRowCount(0)
                self.border_add_btn.setDisabled(True)

    def risk_to_border(self):
        row_count = self.risk_table.currentRow()
        border_name = self.risk_table.item(row_count, 0).text()

        self.border_table_graph_plt.clear()
        self.border_table_graph_plt.addLegend(brush=(255, 255, 255, 120), labelTextColor='555', pen={'color': "ccc", 'width': 1})
        border_data = self.value[border_name]
        border_data_1 = []
        for i in border_data:
            if i != '':
                border_data_1.append(float(i))
        border_data_1 = np.array(border_data_1)
        y, x = np.histogram(border_data_1, bins=30)
        self.border_table_graph_plt.plot(x, y, stepMode='center', fillLevel=0, fillOutLine=False, brush=(161, 164, 167, 255), name='人')

        if self.risk_table.currentColumn() == 1:
            return

        self.border_add_btn.setDisabled(False)
        if border_name not in self.border_data:
            self.border_table.clear()
            self.border_table.setRowCount(1)
            item = QTableWidgetItem(border_name + '--分布 %')
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.border_table.setItem(0, 0, item)
            item = QTableWidgetItem(border_name + '--边界 (值不要相同)')
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.border_table.setItem(0, 1, item)
            item = QTableWidgetItem('')
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.border_table.setItem(0, 2, item)
            return
        border_info = self.border_data[border_name]

        self.border_table.setRowCount(1)

        item = QTableWidgetItem(border_name + '--分布 %')
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(0, 0, item)
        item = QTableWidgetItem(border_name + '--边界 (值不要相同)')
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(0, 1, item)
        item = QTableWidgetItem('')
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(0, 2, item)
        count = 1
        for i in border_info.keys():
            self.border_table.setRowCount(self.border_table.rowCount() + 1)

            item = QTableWidgetItem(i)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.border_table.setItem(count, 0, item)

            item = QTableWidgetItem(border_info[i])
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.border_table.setItem(count, 1, item)

            delete_btn = QPushButton('删除')
            delete_btn.clicked.connect(self.border_table_delete_row)
            self.border_table.setCellWidget(count, 2, delete_btn)

            count += 1

    def borber_table_cell_change(self):
        current_border_name = self.border_table.item(0, 0).text().split('--')[0]
        row_count = self.border_table.rowCount()
        if row_count == 1:
            try:
                self.border_data.pop(current_border_name)
            except:
                pass
            return

        c = {}
        for i in range(1, row_count):
            if self.border_table.item(i, 0) is None:
                continue
            target = self.border_table.item(i, 0).text()
            if target == '':
                continue

            if self.border_table.item(i, 1) is None:
                border = ''
            else:
                border = self.border_table.item(i, 1).text()

            c.update({target: border})

        if c == {}:
            return
        b = {current_border_name: c}
        self.border_data.update(b)

    def border_table_add_row(self):
        self.border_table.setRowCount(self.border_table.rowCount() + 1)

        item = QTableWidgetItem('')
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(self.border_table.rowCount() - 1, 0, item)

        item = QTableWidgetItem('')
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(self.border_table.rowCount() - 1, 1, item)

        delete_btn = QPushButton('删除')
        delete_btn.clicked.connect(self.border_table_delete_row)
        self.border_table.setCellWidget(self.border_table.rowCount() - 1, 2, delete_btn)

    def border_table_delete_row(self):
        delete_row = self.border_table.currentRow()
        self.border_table.removeRow(delete_row)
        self.borber_table_cell_change()

    def config_back(self, project_name, juno):
        self.task_window(project_name, juno)

    def config_save(self, project_name, task_name):
        situation_temp = []
        for i in range(self.situation_table.rowCount()):
            situation_temp.append(self.situation_table.item(i, 0).text())
        self.config_data.update({'situation': situation_temp})

        action_temp = []
        cost_temp = {}
        for i in range(self.cost_table.rowCount()):
            cost_name = self.cost_table.item(i, 0).text()
            if self.cost_table.item(i, 1) is None:
                cost_value = ''
            else:
                cost_value = self.cost_table.item(i, 1).text()
            action_temp.append(cost_name)
            cost_temp.update({cost_name: cost_value})
        self.config_data.update({'action': action_temp})
        self.config_data.update({'cost': cost_temp})

        temp = {}
        for i in range(self.risk_table.rowCount()):
            risk_table_temp = self.risk_table.item(i, 0).text()
            temp.update({risk_table_temp: {}})
        border_data_not = self.border_data.keys() - temp.keys()
        for i in border_data_not:
            self.border_data.pop(i)

        result_temp = []
        for i in self.border_data.keys():
            result_temp.append(i)
        self.config_data.update({'result': result_temp})

        risk_temp = {}
        for i in range(self.risk_table.rowCount()):
            if self.risk_table.item(i, 0).text() in self.border_data.keys():
                risk_name = self.risk_table.item(i, 0).text()
                if self.risk_table.item(i, 1) is None:
                    risk_value = ''
                else:
                    risk_value = self.risk_table.item(i, 1).text()
                risk_temp.update({risk_name: risk_value})
        self.config_data.update({'risk': risk_temp})

        begin_tuple = self.date_begin.date().getDate()
        end_tuple = self.date_end.date().getDate()
        Begin_datestamp = date(begin_tuple[0], begin_tuple[1], begin_tuple[2]).toordinal() - date(1900, 1, 1).toordinal() + 2
        End_datestamp = date(end_tuple[0], end_tuple[1], end_tuple[2]).toordinal() - date(1900, 1, 1).toordinal() + 2
        safety = self.safety_textedit.document().toRawText()
        self.config_data.update({'begin': Begin_datestamp})
        self.config_data.update({'end': End_datestamp})
        self.config_data.update({'safety': safety})

        with open('./JunoProject/' + project_name + '/Task/' + task_name + '.json', 'w') as f:
            f.write(json.dumps(self.config_data))

        print(self.config_data)

    def config_calc(self, juno, project_name, task_name):
        self.config_save(project_name, task_name)

        value = self.value.copy()
        riqi = value['日期']
        riqi_temp = []
        for i in riqi:
            riqi_temp.append(int(float(i)))
        value.pop('日期')

        values = {}
        datas = []
        for i in range(len(riqi_temp)):
            data = []
            data.append(i)
            data.append(i-len(riqi_temp))
            datas.append(data)
        values.update({'时间（过去天数）': datas})
        for i in value.keys():
            index = 0
            datas = []
            for j in value[i]:
                data = []
                data.append(index)
                index += 1
                if str(j) == '':
                    continue
                data.append(float(j))
                datas.append(data)
            values.update({i: datas})

        border = []
        for i in self.config_data['border']:
            data = {}
            for j in self.config_data['border'][i]:
                data.update({j: float(self.config_data['border'][i][j])})
            border.append(data)

        typedefs = []
        for i in range(len(self.config_data['situation'])):
            typedefs.append(-1)
        for i in self.config_data['cost']:
            typedefs.append(float(self.config_data['cost'][i]) / 100.0)
        for i in self.config_data['risk']:
            typedefs.append(float(self.config_data['risk'][i]) / 10.0)

        Begin_datestamp = self.config_data['begin']
        End_datestamp = self.config_data['end']
        startIndex = float(Begin_datestamp) - float(max(self.value['日期'])) - 1
        endIndex = float(End_datestamp) - float(max(self.value['日期'])) - 1
        result = algo.analysis(values, self.config_data['situation'], self.config_data['action'], self.config_data['result'], border, typedefs, safety=float(self.config_data['safety']), startIndex=int(startIndex), endIndex=int(endIndex), verbose=True)

        self.result_window(juno, result, project_name, task_name)

    def result_window(self, juno, result, project_name, task_name):
        juno.setWindowTitle(Version + ' - 结果展示')

        self.table1_label = QLabel('项目: ' + project_name + '\n\n命题: ' + task_name + '\n\n各种边界的AI策略结果\n')
        self.table1 = QTableWidget()
        self.table1.verticalHeader().setVisible(False)
        result_1_content = QWidget()
        result_1_content.setFixedHeight(400)
        result_1_layout = QVBoxLayout(result_1_content)
        result_1_layout.addWidget(self.table1_label)
        result_1_layout.addWidget(self.table1)

        self.table2_label = QLabel('边界人机对比结果\n')
        self.table2 = QTableWidget()
        self.table2.verticalHeader().setVisible(False)
        self.table2.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table2_content = QWidget()
        table2_layout = QVBoxLayout(table2_content)
        table2_layout.addWidget(self.table2_label)
        table2_layout.addWidget(self.table2)

        self.graph1_label = QLabel('人机策略分布\n')
        self.graph1 = pyqtgraph.GraphicsLayoutWidget()
        self.graph1.setBackground('#ffffff')
        self.plt1 = self.graph1.addPlot()
        self.graph1_combo = QComboBox()
        self.graph1_combo.activated.connect(partial(self.graph1_combo_change, result))
        graph1_combo_y_content = QWidget()
        graph1_combo_y_layout = QFormLayout(graph1_combo_y_content)
        graph1_combo_y_layout.addRow(self.graph1_label)
        graph1_combo_y_layout.addRow('横轴: ', self.graph1_combo)
        graph1_combo_y_layout.addRow(self.graph1)

        self.graph2_label = QLabel('两色散点图\n')
        self.graph2 = pyqtgraph.PlotWidget()
        self.graph2.setBackground('#ffffff')
        self.graph2.showGrid(x=True, y=True)
        self.graph2.getAxis('bottom').setLabel(**{"color": "#999", "font-size": "8pt"})
        self.graph2.getAxis('left').setLabel(**{"color": "#999", "font-size": "8pt"})
        self.graph2.setTitle('关系图', **{'color': '#777', 'size': '9pt', 'justify': 'left'})
        self.graph2_combo_x = QComboBox()
        self.graph2_combo_x.activated.connect(partial(self.graph2_combo_change, result))
        self.graph2_combo_y = QComboBox()
        self.graph2_combo_y.activated.connect(partial(self.graph2_combo_change, result))
        graph2_combo_content = QWidget()
        graph2_combo_layout = QFormLayout(graph2_combo_content)
        graph2_combo_layout.addRow(self.graph2_label)
        graph2_combo_layout.addRow('横轴: ', self.graph2_combo_x)
        graph2_combo_layout.addRow('纵轴: ', self.graph2_combo_y)
        graph2_combo_layout.addRow(self.graph2)

        self.tab1 = QWidget()
        result_2_layout = QHBoxLayout(self.tab1)
        result_2_layout.addWidget(table2_content)
        result_2_layout.addWidget(graph1_combo_y_content)
        result_2_layout.addWidget(graph2_combo_content)

        table3_label = QLabel('历史数据人机策略\n')
        self.table3 = QTableWidget()
        self.table3.verticalHeader().setVisible(False)
        table3_content = QWidget()
        table3_layout = QVBoxLayout(table3_content)
        table3_layout.addWidget(table3_label)
        table3_layout.addWidget(self.table3)
        table4_label = QLabel('边界定义\n')
        self.table4 = QTableWidget()
        self.table4.verticalHeader().setVisible(False)
        table4_content = QWidget()
        table4_layout = QVBoxLayout(table4_content)
        table4_layout.addWidget(table4_label)
        table4_layout.addWidget(self.table4)
        table5_label = QLabel('人机策略-成本对比\n')
        self.table5 = QTableWidget()
        self.table5.verticalHeader().setVisible(False)
        self.table5.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table5_content = QWidget()
        table5_layout = QVBoxLayout(table5_content)
        table5_layout.addWidget(table5_label)
        table5_layout.addWidget(self.table5)
        table6_label = QLabel('人机策略-风险对比\n')
        self.table6 = QTableWidget()
        self.table6.verticalHeader().setVisible(False)
        self.table6.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table6_content = QWidget()
        table6_layout = QVBoxLayout(table6_content)
        table6_layout.addWidget(table6_label)
        table6_layout.addWidget(self.table6)
        table456_content = QWidget()
        table456_layout = QHBoxLayout(table456_content)
        table456_layout.addWidget(table4_content)
        table456_layout.addWidget(table5_content)
        table456_layout.addWidget(table6_content)

        self.tab2 = QWidget()
        result_3_layout = QVBoxLayout(self.tab2)
        result_3_layout.addWidget(table3_content)
        result_3_layout.addWidget(table456_content)

        self.table7 = QTableWidget()
        self.table7.horizontalHeader().setVisible(False)
        self.table7.cellChanged.connect(partial(self.table7_change, result))
        self.table7_add = QPushButton('添加')
        self.table7_add.clicked.connect(partial(self.table7_add_btn, result))
        table7_content = QWidget()
        table7_layout = QVBoxLayout(table7_content)
        table7_layout.addWidget(self.table7)
        table7_layout.addWidget(self.table7_add)

        self.table8 = QTableWidget()
        self.table8.horizontalHeader().setVisible(False)
        self.table8.cellChanged.connect(partial(self.table8_change, result))
        self.table8_add = QPushButton('添加')
        self.table8_add.clicked.connect(partial(self.table8_add_btn, result))
        table8_content = QWidget()
        table8_layout = QVBoxLayout(table8_content)
        table8_layout.addWidget(self.table8)
        table8_layout.addWidget(self.table8_add)

        # self.forecast_result = QPushButton('预测结果')
        # self.forecast_result.clicked.connect(partial(self.forecast_result_btn, result))

        self.table9 = QTableWidget()
        self.table9.horizontalHeader().setVisible(False)

        self.tab3 = QWidget()
        result_4_layout = QHBoxLayout(self.tab3)
        result_4_layout.addWidget(table7_content)
        result_4_layout.addWidget(table8_content)
        # result_4_layout.addWidget(self.forecast_result)
        result_4_layout.addWidget(self.table9)

        tabs = QTabWidget()
        tabs.addTab(self.tab1, '策略概览')
        tabs.addTab(self.tab2, '策略分析依据')
        tabs.addTab(self.tab3, '策略应用')

        back_btn = QPushButton('返回')
        back_btn.clicked.connect(partial(self.result_back_to_config, project_name, task_name, juno))
        back_btn.setFixedHeight(50)
        back_btn.setFixedWidth(133)
        export_btn = QPushButton('导出')
        export_btn.clicked.connect(partial(self.export_to_excel, project_name, task_name, result))
        export_btn.setFixedHeight(50)
        export_btn.setFixedWidth(133)
        btn_content = QWidget()
        btn_layout = QHBoxLayout(btn_content)
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(export_btn)

        result_content = QWidget()
        result_layout = QVBoxLayout(result_content)
        result_layout.addWidget(result_1_content)
        result_layout.addWidget(tabs)
        result_layout.addWidget(btn_content)

        juno.setCentralWidget(result_content)

        self.load_table1(result)
        self.load_table2(result)
        self.load_table3(result)
        self.load_table4()
        self.load_table5(result)
        self.load_table6(result)
        self.load_table7(result)
        self.load_table8()
        self.load_table9()

        self.load_graph1_combo(result)
        self.load_graph2_combo(result)

        self.graph1_combo_change(result)
        self.graph2_combo_change(result)

    def result_back_to_config(self, project_name, task_name, juno):
        self.config_window(project_name, juno, task_name)

    def load_table1(self, result):
        table_title = []
        count = 0
        for i in result.risks.keys():
            count += 1
            for j in result.risks[i]['threshold']:
                table_title.append('Z' + str(count) + ' - ' + i + '-' + j)
        count = 0
        for i in result.consumptions.keys():
            count += 1
            table_title.append('Y' + str(count) + ' - ' + i + ' 节约率(%)')
        count = 0
        for i in result.risks.keys():
            count += 1
            table_title.append('Z' + str(count) + ' - ' + i + ' 达标优势(%)')

        value = []
        for i in result.risks.keys():
            for j in result.risks[i]['threshold']:
                value.append(round(result.risks[i]['threshold'][j], 2))
        for i in result.consumptions.keys():
            value.append(round(result.consumptions[i]['saving_rate'], 2))
        for i in result.risks.keys():
            value.append(round(result.risks[i]['ai_advantage'], 2))

        self.table1.setColumnCount(len(table_title))
        self.table1.setHorizontalHeaderLabels(table_title)
        self.table1.resizeColumnsToContents()
        self.table1.setRowCount(1)
        for i in range(len(value)):
            item = QTableWidgetItem(str(value[i]))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table1.setItem(0, i, item)

    def load_table2(self, result):
        self.table2.setColumnCount(3)
        self.table2.setHorizontalHeaderLabels(['', '历史运行表现(人)', 'AI策略预计表现'])

        count = 0
        for i in result.consumptions.keys():
            count += 1
            self.table2.setRowCount(self.table2.rowCount() + 1)

            item = QTableWidgetItem('Y' + str(count) + ' - ' + i + ' - 使用量(%)')
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table2.setItem(self.table2.rowCount() - 1, 0, item)

            item = QTableWidgetItem(format(result.consumptions[i]['old_usage'], '.2f'))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table2.setItem(self.table2.rowCount() - 1, 1, item)

            item = QTableWidgetItem(format(result.consumptions[i]['new_usage'], '.2f'))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table2.setItem(self.table2.rowCount() - 1, 2, item)

        count = 0
        for i in result.risks.keys():
            count += 1
            self.table2.setRowCount(self.table2.rowCount() + 1)

            item = QTableWidgetItem('Z' + str(count) + ' - ' + i + ' - 超标率(%)')
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table2.setItem(self.table2.rowCount() - 1, 0, item)

            item = QTableWidgetItem(format(result.risks[i]['hm_failure_rate'], '.2f'))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table2.setItem(self.table2.rowCount() - 1, 1, item)

            item = QTableWidgetItem(format(result.risks[i]['ai_failure_rate'], '.2f'))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table2.setItem(self.table2.rowCount() - 1, 2, item)

        self.table2.resizeRowsToContents()

    def load_graph1_combo(self, result):
        self.graph1_combo_value = []
        for i in result.consumptions.keys():
            self.graph1_combo_value.append('行为 - ' + i)
        for i in result.risks.keys():
            self.graph1_combo_value.append('结果 - ' + i)
        self.graph1_combo.addItems(self.graph1_combo_value)

    def load_graph2_combo(self, result):
        self.graph2_combo_value = []
        for i in self.config_data['situation']:
            self.graph2_combo_value.append('情况 - ' + i)
        for i in result.consumptions.keys():
            self.graph2_combo_value.append('行为 - ' + i)
        for i in result.risks.keys():
            self.graph2_combo_value.append('结果 - ' + i)
        self.graph2_combo_x.addItems(self.graph2_combo_value)
        self.graph2_combo_y.addItems(self.graph2_combo_value)

    def graph1_combo_change(self, result):
        self.plt1.clear()
        self.plt1.addLegend(brush=(255, 255, 255, 120), labelTextColor='555', pen={'color': "ccc", 'width': 1})

        select_text = self.graph1_combo.currentText()

        if '行为' in select_text:
            select_text = select_text.split(' - ')[1]
            index = self.config_data['action'].index(select_text)

            y_hm = []
            for i in result.Y:
                y_hm.append(i[index])
            y_hm = np.array(y_hm)
            y_hm, x_hm = np.histogram(y_hm, bins=10)

            y_ai = []
            for i in result.Yopt:
                y_ai.append(i[index])
            y_ai = np.array(y_ai)
            y_ai, x_ai = np.histogram(y_ai, bins=10)

            dx = (np.percentile(x_hm, 75) - np.percentile(x_hm, 25)) / (np.percentile(x_ai, 75) - np.percentile(x_ai, 25))
            y_hm = y_hm / (dx * np.percentile(y_hm, 100))
            y_ai = y_ai / np.max(y_ai)

        elif '结果' in select_text:
            select_text = select_text.split(' - ')[1]
            index = self.config_data['result'].index(select_text)

            y_hm = []
            for i in result.Z:
                y_hm.append(i[index])
            y_hm = np.array(y_hm)
            y_hm, x_hm = np.histogram(y_hm, bins=10)

            y_ai = []
            for i in result.Zopt:
                y_ai.append(i[index])
            y_ai = np.array(y_ai)
            y_ai, x_ai = np.histogram(y_ai, bins=10)

            dx = (np.percentile(x_hm, 75) - np.percentile(x_hm, 25)) / (np.percentile(x_ai, 75) - np.percentile(x_ai, 25))
            y_hm = y_hm / (dx * np.percentile(y_hm, 100))
            y_ai = y_ai / np.max(y_ai)

        else:
            print("graph1_combo_change() 出错啦！")
            return

        self.plt1.plot(x_hm, y_hm, stepMode='center', fillLevel=0, fillOutLine=False, brush=(161, 164, 167, 150), name='人 - ' + self.graph1_combo.currentText())
        self.plt1.plot(x_ai, y_ai, stepMode='center', fillLevel=0, fillOutLine=False, brush=(15, 79, 168, 150), name='AI - ' + self.graph1_combo.currentText())

    def graph2_combo_change(self, result):

        x = self.graph2_combo_x.currentText()
        y = self.graph2_combo_y.currentText()

        self.graph2.clear()
        self.graph2.setTitle(x + " 和 " + y + " 之间的关系")
        self.graph2.setLabel('left', y)
        self.graph2.setLabel('bottom', x)
        self.graph2.addLegend(brush=(255, 255, 255, 120), labelTextColor='555', pen={'color': "ccc", 'width': 1})

        axis_x_hm = []
        axis_x_ai = []
        if '情况' in x:
            x = x.split(' - ')[1]
            index_x = self.config_data['situation'].index(x)
            for i in result.Xhm:
                axis_x_hm.append(i[index_x])
            for i in result.Xopt:
                axis_x_ai.append(i[index_x])
        elif '行为' in x:
            x = x.split(' - ')[1]
            index_x = self.config_data['action'].index(x)
            for i in result.Yhm:
                axis_x_hm.append(i[index_x])
            for i in result.Yopt:
                axis_x_ai.append(i[index_x])
        elif '结果' in x:
            x = x.split(' - ')[1]
            index_x = self.config_data['result'].index(x)
            for i in result.Zhm:
                axis_x_hm.append(i[index_x])
            for i in result.Zopt:
                axis_x_ai.append(i[index_x])

        axis_y_hm = []
        axis_y_ai = []
        if '情况' in y:
            y = y.split(' - ')[1]
            index_y = self.config_data['situation'].index(y)
            for i in result.Xhm:
                axis_y_hm.append(i[index_y])
            for i in result.Xopt:
                axis_y_ai.append(i[index_y])
        elif '行为' in y:
            y = y.split(' - ')[1]
            index_y = self.config_data['action'].index(y)
            for i in result.Yhm:
                axis_y_hm.append(i[index_y])
            for i in result.Yopt:
                axis_y_ai.append(i[index_y])
        elif '结果' in y:
            y = y.split(' - ')[1]
            index_y = self.config_data['result'].index(y)
            for i in result.Zhm:
                axis_y_hm.append(i[index_y])
            for i in result.Zopt:
                axis_y_ai.append(i[index_y])

        scatter_hm = pyqtgraph.ScatterPlotItem(size=5, brush=pyqtgraph.mkBrush(161, 164, 167, 50))
        scatter_hm.setData(pos=zip(axis_x_hm, axis_y_hm), alpha=1, name='人')
        self.graph2.addItem(scatter_hm)

        scatter_ai = pyqtgraph.ScatterPlotItem(size=7, brush=pyqtgraph.mkBrush(15, 79, 168, 150))
        scatter_ai.setData(pos=zip(axis_x_ai, axis_y_ai), alpha=1, name='AI')
        self.graph2.addItem(scatter_ai)

    def load_table3(self, result):
        title = ['时间']
        count = 1
        for i in self.config_data['situation']:
            title.append('X' + str(count) + ' - ' + i)
            count += 1
        count = 1
        for i in self.config_data['action']:
            title.append('Y' + str(count) + ' - ' + i + ' - 历史')
            title.append('Y' + str(count) + ' - ' + i + ' - AI')
            count += 1
        count = 1
        for i in self.config_data['result']:
            title.append('Z' + str(count) + ' - ' + i + ' - 历史')
            title.append('Z' + str(count) + ' - ' + i + ' - AI')
            count += 1
        self.table3.setColumnCount(len(title))
        self.table3.setHorizontalHeaderLabels(title)

        begin_tuple = self.date_begin.date().getDate()
        Begin_datestamp = date(begin_tuple[0], begin_tuple[1], begin_tuple[2]).toordinal() - date(1900, 1, 1).toordinal() + 2
        index = self.value['日期'].index(str(float(Begin_datestamp)))
        values = []
        for i in range(len(result.Yhm)):
            value = []
            value.append(date.fromordinal(int(float(self.value['日期'][i+index]) + date(1900, 1, 1).toordinal() - 2)))
            for j in result.X[i]:
                value.append(round(j, 2))
            for j in range(len(self.config_data['action'])):
                value.append(round(result.Yhm[i][j], 2))
                value.append(round(result.Yopt[i][j], 2))
            for j in range(len(self.config_data['result'])):
                value.append(round(result.Zhm[i][j], 2))
                value.append(round(result.Zopt[i][j], 2))
            values.append(value)
        values.reverse()
        for i in values:
            self.table3.setRowCount(self.table3.rowCount() + 1)
            count = 0
            for j in i:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table3.setItem(self.table3.rowCount() - 1, count, item)
                count += 1

        self.table3.resizeColumnsToContents()

    def load_table4(self):
        title = ['结果变量', '重要性占比']
        border_value = []
        for i in self.config_data['border']:
            for j in self.config_data['border'][i]:
                border_value.append(j)
        border_value = sorted(border_value, reverse=True)
        title += border_value
        title_1 = []
        for i in title:
            if i not in title_1:
                title_1.append(i)
        self.table4.setColumnCount(len(title_1))
        self.table4.setHorizontalHeaderLabels(title_1)

        count_z = 1
        for i in self.config_data['border']:
            value = []
            value.append('Z' + str(count_z) + ' - ' + i)
            value.append(self.config_data['risk'][i])
            for j in border_value:
                if j in self.config_data['border'][i].keys():
                    value.append(self.config_data['border'][i][j])
                else:
                    value.append('')
            count_z += 1
            value_1 = []
            for i in value:
                if i not in value_1:
                    value_1.append(i)

            self.table4.setRowCount(self.table4.rowCount() + 1)
            count = 0
            for j in value_1:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table4.setItem(self.table4.rowCount() - 1, count, item)
                count += 1
            self.table4.resizeColumnToContents(0)

    def load_table5(self, result):
        self.table5.setColumnCount(3)
        self.table5.setHorizontalHeaderLabels(['行为变量', '历史平均', 'AI平均'])

        count_y = 1
        for i in result.consumptions:
            value = []
            value.append('Y' + str(count_y) + ' - ' + i)
            value.append(format(result.consumptions[i]['old_usage'], '.2f'))
            value.append(format(result.consumptions[i]['new_usage'], '.2f'))
            count_y += 1

            self.table5.setRowCount(self.table5.rowCount() + 1)
            count = 0
            for j in value:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table5.setItem(self.table5.rowCount() - 1, count, item)
                count += 1

    def load_table6(self, result):
        self.table6.setColumnCount(3)
        self.table6.setHorizontalHeaderLabels(['结果变量', '历史超标率(%)', 'AI超标率(%)'])

        count_z = 1
        for i in result.risks:
            value = []
            value.append('Z' + str(count_z) + ' - ' + i)
            value.append(format(result.risks[i]['hm_failure_rate'], '.2f'))
            value.append(format(result.risks[i]['ai_failure_rate'], '.2f'))
            count_z += 1

            self.table6.setRowCount(self.table6.rowCount() + 1)
            count = 0
            for j in value:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table6.setItem(self.table6.rowCount() - 1, count, item)
                count += 1

    def load_table7(self, result):
        self.table7.setRowCount(len(self.config_data['situation']) + 2)
        x_title = ['情况变量']
        count_x = 1
        for i in self.config_data['situation']:
            x_title.append('X' + str(count_x) + ' - ' + i)
            count_x += 1
        x_title.append('')
        self.table7.setVerticalHeaderLabels(x_title)
        self.table7.resizeRowsToContents()

        self.table7_add_btn(result)

    def table7_add_btn(self, result):
        self.table7.setColumnCount(self.table7.columnCount() + 1)

        item = QTableWidgetItem('自定义输入 - ' + str(self.table7.columnCount()))
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.table7.setItem(0, self.table7.columnCount() - 1, item)

        for i in range(1, self.table7.rowCount() - 1):
            item = QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table7.setItem(i, self.table7.columnCount() - 1, item)

        table7_delete_column_btn = QPushButton('删除')
        table7_delete_column_btn.clicked.connect(partial(self.table7_delete_column_btn, result))
        self.table7.setCellWidget(self.table7.rowCount() - 1, self.table7.columnCount() - 1, table7_delete_column_btn)
        self.table7.resizeRowsToContents()
        self.table7.resizeColumnsToContents()

    def table7_delete_column_btn(self, result):
        delete_index = self.table7.currentColumn()
        self.table7.removeColumn(delete_index)
        self.table7_change(result)

    def table7_change(self, result):
        row_count = self.table7.rowCount()
        column_count = self.table7.columnCount()
        if row_count == 0 or column_count == 0:
            self.table8.setColumnCount(0)
            self.table9.setColumnCount(0)
            return
        for i in range(1, row_count - 1):
            for j in range(column_count):
                try:
                    float(self.table7.item(i, j).text())
                except:
                    return
        self.forecast_action(result)

    def forecast_action(self, result):
        rowCount = self.table7.rowCount()
        columnCount = self.table7.columnCount()
        x = []
        for i in range(columnCount):
            x1 = []
            for j in range(1, rowCount - 1):
                x1.append(float(self.table7.item(j, i).text()))
            x.append(x1)
        y_hm = result.S_hm(x)
        y_ai = result.S(x)

        self.table8.setColumnCount(len(y_hm) + len(y_ai))
        count_table8 = 0

        count_hm = 1
        for i in y_hm:
            y = ['人 - ' + str(count_hm)]
            for j in i:
                y.append(round(j, 2))
            count_hm += 1
            y.append('')
            count_y = 0
            for j in y:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table8.setItem(count_y, count_table8, item)
                count_y += 1
            count_table8 += 1

        count_ai = 1
        for i in y_ai:
            y = ['AI - ' + str(count_ai)]
            for j in i:
                y.append(round(j, 2))
            count_ai += 1
            y.append('')
            count_y = 0
            for j in y:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table8.setItem(count_y, count_table8, item)
                count_y += 1
            count_table8 += 1

        self.table8.resizeRowsToContents()

    def load_table8(self):
        self.table8.setRowCount(len(self.config_data['action']) + 2)
        y_title = ['行为变量']
        count_y = 1
        for i in self.config_data['action']:
            y_title.append('Y' + str(count_y) + ' - ' + i)
            count_y += 1
        y_title.append('')
        self.table8.setVerticalHeaderLabels(y_title)
        self.table8.resizeRowsToContents()

    def table8_add_btn(self, result):
        old_table8_columnCount = 0
        for i in range(self.table8.columnCount()):
            if '自定义输入' not in self.table8.item(0, i).text():
                old_table8_columnCount += 1
        for i in range(self.table8.columnCount() - old_table8_columnCount):
            self.table8.removeColumn(old_table8_columnCount)

        for i in range(self.table7.columnCount()):
            self.table8.setColumnCount(self.table8.columnCount() + 1)

            item = QTableWidgetItem('自定义输入 - ' + str(self.table8.columnCount() - self.table7.columnCount()*2))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table8.setItem(0, self.table8.columnCount() - 1, item)

            for j in range(1, self.table8.rowCount() - 1):
                item = QTableWidgetItem()
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table8.setItem(j, self.table8.columnCount() - 1, item)
            table8_delete_column_btn = QPushButton('删除')
            table8_delete_column_btn.clicked.connect(partial(self.table8_delete_column_btn, result))
            self.table8.setCellWidget(self.table8.rowCount() - 1, self.table8.columnCount() - 1, table8_delete_column_btn)
            self.table8.resizeRowsToContents()

    def table8_delete_column_btn(self, result):
        delete_columns_count = self.table7.columnCount()
        for i in range(delete_columns_count):
            self.table8.removeColumn(delete_columns_count*2)

        self.table8_change(result)

    def table8_change(self, result):
        row_count = self.table8.rowCount()
        column_count = self.table8.columnCount()
        for i in range(1, row_count - 1):
            for j in range(column_count):
                try:
                    float(self.table8.item(i, j).text())
                except:
                    return
        self.forecast_result(result)

    def forecast_result(self, result):
        rowCount_x = self.table7.rowCount()
        columnCount_x = self.table7.columnCount()
        x = []
        for i in range(columnCount_x):
            x1 = []
            for j in range(1, rowCount_x - 1):
                x1.append(float(self.table7.item(j, i).text()))
            x.append(x1)

        rowCount_y = self.table8.rowCount()
        columnCount_y = self.table8.columnCount()
        y = []
        for i in range(columnCount_y):
            y1 = []
            for j in range(1, rowCount_y - 1):
                y1.append(float(self.table8.item(j, i).text()))
            y.append(y1)

        T_input = []
        for i in range(len(y) // len(x)):
            for j in range(len(x)):
                a = x[j] + y[0]
                T_input.append(a)
                y.remove(y[0])

        z = result.T(T_input).tolist()

        self.table9.setColumnCount(len(z))
        count_table9 = 0

        count_hm = 1
        for i in range(self.table7.columnCount()):
            z1 = ['人 - ' + str(count_hm)]
            for j in z[0]:
                z1.append(round(j, 2))
            z.pop(0)
            count_hm += 1
            count_z = 0
            for j in z1:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table9.setItem(count_z, count_table9, item)
                count_z += 1
            count_table9 += 1

        count_ai = 1
        for i in range(self.table7.columnCount()):
            z2 = ['AI - ' + str(count_ai)]
            for j in z[0]:
                z2.append(round(j, 2))
            z.pop(0)
            count_ai += 1
            count_z = 0
            for j in z2:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table9.setItem(count_z, count_table9, item)
                count_z += 1
            count_table9 += 1

        table8_header = []
        for i in range(self.table8.columnCount()):
            table8_header.append(self.table8.item(0, i).text())
        if '自定义输入 - 1' in table8_header:
            count_label = 1
            for i in range(self.table7.columnCount()):
                z2 = ['自定义预测 - ' + str(count_label)]
                for j in z[0]:
                    z2.append(round(j, 2))
                z.pop(0)
                count_z = 0
                for j in z2:
                    item = QTableWidgetItem(str(j))
                    item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.table9.setItem(count_z, count_table9, item)
                    count_z += 1
                count_table9 += 1
                count_label += 1

        self.table9.resizeRowsToContents()

    def load_table9(self):
        self.table9.setRowCount(len(self.config_data['result']) + 1)
        z_title = ['结果变量']
        count_z = 1
        for i in self.config_data['result']:
            z_title.append('Z' + str(count_z) + ' - ' + i)
            count_z += 1
        self.table9.setVerticalHeaderLabels(z_title)
        self.table9.resizeColumnsToContents()

    def export_to_excel(self, project_name, task_name, result):
        str = "./JunoProject/" + project_name + "/Output/"
        str_image_1 = "./JunoProject/" + project_name + "/Output/Image/人机策略分布"
        str_image_2 = "./JunoProject/" + project_name + "/Output/Image/两色散点图"
        if not os.path.isdir(str):
            os.makedirs(str)
        if not os.path.isdir(str_image_1):
            os.makedirs(str_image_1)
        if not os.path.isdir(str_image_2):
            os.makedirs(str_image_2)

        current_graph1_combo = self.graph1_combo.currentText()
        current_graph2_combo_x = self.graph2_combo_x.currentText()
        current_graph2_combo_y = self.graph2_combo_y.currentText()
        for i in self.graph1_combo_value:
            self.graph1_combo.setCurrentText(i)
            self.graph1_combo_change(result)
            a = 'JunoProject/' + project_name + '/Output/Image/两色散点图/' + i.replace(' ', '').replace('/', '') +'.png'
            pyqtgraph.exporters.ImageExporter(self.plt1).export(fileName=a)
        for i in self.graph2_combo_value:
            for j in self.graph2_combo_value:
                if i != j:
                    self.graph2_combo_x.setCurrentText(i)
                    self.graph2_combo_y.setCurrentText(j)
                    self.graph2_combo_change(result)
                    a = 'JunoProject/' + project_name + '/Output/Image/人机策略分布/' + i.replace(' ', '').replace('/', '').split('(')[0] + '&' + j.replace(' ', '').replace('/', '').split('(')[0] +'.png'
                    pyqtgraph.exporters.ImageExporter(self.graph2.plotItem).export(fileName=a)
        self.graph1_combo.setCurrentText(current_graph1_combo)
        self.graph1_combo_change(result)
        self.graph2_combo_x.setCurrentText(current_graph2_combo_x)
        self.graph2_combo_y.setCurrentText(current_graph2_combo_y)
        self.graph2_combo_change(result)

        xlsx = xlsxwriter.Workbook(str + task_name + '.xlsx')
        text_format = xlsx.add_format({'text_wrap': 'True'})

        table_1 = xlsx.add_worksheet('各种边界的AI策略结果')
        for i in range(self.table1.columnCount()):
            data = self.table1.horizontalHeaderItem(i).text()
            table_1.write(0, i, data, text_format)
        for i in range(self.table1.rowCount()):
            for j in range(self.table1.columnCount()):
                data = self.table1.item(i, j).text()
                try:
                    data = float(data)
                except:
                    pass
                table_1.write(i + 1, j, data, text_format)

        table_2 = xlsx.add_worksheet('边界人机对比结果')
        for i in range(self.table2.columnCount()):
            data = self.table2.horizontalHeaderItem(i).text()
            table_2.write(0, i, data, text_format)
        for i in range(self.table2.rowCount()):
            for j in range(self.table2.columnCount()):
                data = self.table2.item(i, j).text()
                try:
                    data = float(data)
                except:
                    pass
                table_2.write(i + 1, j, data, text_format)

        table_3 = xlsx.add_worksheet('历史数据人机策略')
        for i in range(self.table3.columnCount()):
            data = self.table3.horizontalHeaderItem(i).text()
            table_3.write(0, i, data, text_format)
        for i in range(self.table3.rowCount()):
            for j in range(self.table3.columnCount()):
                data = self.table3.item(i, j).text()
                try:
                    data = float(data)
                except:
                    pass
                table_3.write(i + 1, j, data, text_format)

        table_4 = xlsx.add_worksheet('边界定义')
        for i in range(self.table4.columnCount()):
            data = self.table4.horizontalHeaderItem(i).text()
            table_4.write(0, i, data, text_format)
        for i in range(self.table4.rowCount()):
            for j in range(self.table4.columnCount()):
                data = self.table4.item(i, j).text()
                try:
                    data = float(data)
                except:
                    pass
                table_4.write(i + 1, j, data, text_format)

        table_5 = xlsx.add_worksheet('人机策略-成本对比')
        for i in range(self.table5.columnCount()):
            data = self.table5.horizontalHeaderItem(i).text()
            table_5.write(0, i, data, text_format)
        for i in range(self.table5.rowCount()):
            for j in range(self.table5.columnCount()):
                data = self.table5.item(i, j).text()
                try:
                    data = float(data)
                except:
                    pass
                table_5.write(i + 1, j, data, text_format)

        table_6 = xlsx.add_worksheet('人机策略-风险对比')
        for i in range(self.table6.columnCount()):
            data = self.table6.horizontalHeaderItem(i).text()
            table_6.write(0, i, data, text_format)
        for i in range(self.table6.rowCount()):
            for j in range(self.table6.columnCount()):
                data = self.table6.item(i, j).text()
                try:
                    data = float(data)
                except:
                    pass
                table_6.write(i + 1, j, data, text_format)

        xlsx.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    MainWindow = QMainWindow()
    window = JunoUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
