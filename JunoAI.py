import base64
import datetime
import random
import re
import sys
import os
from PyQt5 import QtCore, QtGui
from PyQt5.Qt import QStandardItemModel, QStandardItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, \
    QWidget, QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem, QFileDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QTabWidget, QTextBrowser, QDateEdit, QHeaderView, QDialog, QTreeView, QLineEdit, QMenuBar, QMenu
from PyQt5.QtWidgets import *

from functools import partial
from Crypto.Cipher import AES
import xlrd
import xlsxwriter
from datetime import date
import pyqtgraph
import pyqtgraph.exporters
import hashlib
import json
import time
import shutil
import numpy as np
import algo

Version = 'Juno 3.0'
operator_limit_task = 'AI预测_全局命题'
encrypt_decrypt_key = 'Bl6Wv4OQFqhVfFPqm8OfXNW8xpUEPw1U'
desktop_path = "C:/Users/54961/Desktop/"

if not os.path.isdir('./JunoProject'):
    os.mkdir('./JunoProject')
if not os.path.isdir('./SourceTable'):
    os.mkdir('./SourceTable')


class JunoUI(object):
    def __init__(self, juno, width, height):
        self.screen_width = width
        self.screen_height = height

        self.central_widget = QWidget(juno)
        self.login_window(juno)

        self.value_temp = {}
        self.filename = ''

    def login_window(self, juno):
        width = 0.3 * self.screen_width
        height = 0.3 * self.screen_height
        juno.setWindowTitle(Version + ' - 登录')

        pixmap = QPixmap('static/JunoAi_Login.png')
        pic = QLabel()
        pic.setPixmap(pixmap)
        pic.setScaledContents(True)
        pic.setFixedHeight(height*0.38)
        self.user_login = QLineEdit()
        self.user_login.setObjectName('login')
        self.user_login.setFixedSize(width*0.4, height*0.075)
        self.user_login.setPlaceholderText('用户名')
        self.password_login = QLineEdit()
        self.password_login.setObjectName('login')
        self.password_login.setFixedSize(width*0.4, height*0.075)
        self.password_login.setPlaceholderText('密码')
        login_btn = QPushButton('登 录')
        login_btn.setObjectName('dengluye')
        login_btn.setFixedSize(width*0.185, height*0.08)
        login_btn.clicked.connect(partial(self.login, juno))
        signin_btn = QPushButton('注 册')
        signin_btn.setObjectName('zhuce')
        signin_btn.setFixedSize(width*0.185, height*0.08)
        signin_btn.clicked.connect(partial(self.signin_window, juno))
        btn_content = QWidget(objectName='denglu')
        btn_layout = QHBoxLayout(btn_content)
        btn_layout.addWidget(signin_btn)
        btn_layout.addWidget(login_btn)
        input_content = QWidget(objectName='denglu')
        input_layout = QVBoxLayout(input_content)
        input_layout.setAlignment(QtCore.Qt.AlignCenter)
        input_layout.addWidget(self.user_login)
        input_layout.addWidget(self.password_login)
        input_layout.addWidget(btn_content)

        main_content = QWidget(objectName='denglu')
        main_content.setMinimumSize(width, height)
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(pic)
        main_layout.addWidget(input_content)
        juno.setCentralWidget(main_content)
        juno.resize(width, height)
        juno.move(self.screen_width * 0.35, self.screen_height * 0.3)

    def signin_window(self, juno):
        width = 0.3 * self.screen_width
        height = 0.3 * self.screen_height
        juno.setWindowTitle(Version + ' - 注册')

        pixmap = QPixmap('static/JunoAi_Signin.png')
        pic = QLabel()
        pic.setPixmap(pixmap)
        pic.setScaledContents(True)
        pic.setFixedHeight(height * 0.38)
        self.user_signin = QLineEdit()
        self.user_signin.setObjectName('zhuceyonghu')
        self.user_signin.setFixedSize(width * 0.4, height * 0.075)
        self.user_signin.setPlaceholderText('用户名，6-16位，只能使用字母和数字')
        self.password_signin = QLineEdit()
        self.password_signin.setObjectName('zhuceyonghu')
        self.password_signin.setFixedSize(width * 0.4, height * 0.075)
        self.password_signin.setPlaceholderText('密码，6-16位，只能使用字母和数字')
        self.admin_lineEdit = QLineEdit()
        self.admin_lineEdit.setObjectName('zhuceyonghu')
        self.admin_lineEdit.setFixedSize(width * 0.4, height * 0.075)
        self.admin_lineEdit.setPlaceholderText('管理员令牌。联系：155-5505-9636')
        yes_btn = QPushButton('注 册')
        yes_btn.setObjectName('zhuce')
        yes_btn.setFixedSize(width * 0.185, height * 0.08)
        yes_btn.clicked.connect(partial(self.signin, juno))
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao1')
        no_btn.setFixedSize(width * 0.185, height * 0.08)
        no_btn.clicked.connect(partial(self.login_window, juno))
        btn_content = QWidget()
        btn_layout = QHBoxLayout(btn_content)
        btn_layout.addWidget(yes_btn)
        btn_layout.addWidget(no_btn)
        input_content = QWidget()
        input_layout = QVBoxLayout(input_content)
        input_layout.setAlignment(QtCore.Qt.AlignCenter)
        input_layout.addWidget(self.user_signin)
        input_layout.addWidget(self.password_signin)
        input_layout.addWidget(self.admin_lineEdit)
        input_layout.addWidget(btn_content)

        main_content = QWidget()
        main_content.setMinimumSize(width, height)
        main_layout = QVBoxLayout(main_content)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(pic)
        main_layout.addWidget(input_content)
        juno.setCentralWidget(main_content)
        juno.resize(width, height)

    def login(self, juno):
        # 格式检查
        if self.user_login.text() == '':
            self.user_error_dialog('用户名不能为空!', juno)
            return
        elif self.password_login.text() == '':
            self.user_error_dialog('密码不能为空!', juno)
            return

        user_encrypt = self.md5(self.user_login.text())
        password_encrypt = self.md5(self.password_login.text())
        with open('user.txt', 'r') as f:
            users = f.readlines()
        for i in users:
            role_pos = int(i[:2], 16)
            self.role = i[2 + role_pos]
            account = i[2:role_pos + 2] + i[role_pos + 3:]
            user = account[:32]
            password = account[32:].rstrip('\n')
            if user_encrypt == user and password_encrypt == password:
                self.Main_Window(juno)
                return
            elif user_encrypt == user and password_encrypt != password:
                self.user_error_dialog('密码错误!', juno)
        self.user_error_dialog('用户名错误!', juno)

    def signin(self, juno):
        # 格式检查
        p = re.compile(r'[a-zA-Z0-9]')
        if len(self.user_signin.text()) < 6:
            self.user_error_dialog('用户名不能小于6位!', juno)
            return
        elif len(self.password_signin.text()) < 6:
            self.user_error_dialog('密码不能小于6位!', juno)
            return
        if len(self.user_signin.text()) > 16:
            self.user_error_dialog('用户名不能大于16位!', juno)
            return
        elif len(self.password_signin.text()) > 16:
            self.user_error_dialog('密码不能大于16位!', juno)
            return
        elif not p.match(self.user_signin.text()):
            self.user_error_dialog('用户名含有不支持字符!', juno)
            return
        elif not p.match(self.password_signin.text()):
            self.user_error_dialog('密码含有不支持字符!', juno)
            return

        # admin令牌判断:
        role_total = ['0', '1', '2', '3']
        current_time = time.time()
        current_time = current_time - current_time % 60
        current_time_encrypt = self.md5(int(current_time))
        admin_password = self.admin_lineEdit.text()
        role = admin_password[-1]
        if role not in role_total:
            self.user_error_dialog('权限分配错误!', juno)
            return
        if len(admin_password[:-1]) < 5:
            self.user_error_dialog('令牌长度不足!', juno)
            return
        if len(admin_password[:-1]) > 32:
            self.user_error_dialog('令牌长度超长!', juno)
            return
        if admin_password[:-1] not in current_time_encrypt:
            self.user_error_dialog('令牌错误!', juno)
            return

        # 加密
        role_pos_int = random.sample(range(0, 64), 1)[0]
        role_pos_hex = hex(role_pos_int)[2:] if role_pos_int > 15 else '0' + hex(role_pos_int)[2:]
        user_encrypt = self.md5(self.user_signin.text())
        password_encrypt = self.md5(self.password_signin.text())

        total = user_encrypt + password_encrypt
        account = role_pos_hex + total[:role_pos_int] + role + total[role_pos_int:]

        # 用户名重复性验证
        exists_users = []
        with open('user.txt', 'r') as f:
            users = f.readlines()
        for i in users:
            role_pos = int(i[:2], 16)
            exists_account = i[2:role_pos + 2] + i[role_pos + 3:]
            exists_user = exists_account[:32]
            exists_users.append(exists_user)
        if user_encrypt in exists_users:
            self.user_error_dialog('该用户名已存在!', juno)
        else:
            '''
            头两位是role所在位置的hex，
            去除掉头两位之后，在剩余的字符串中的第hex->int所在位获得role，
            去掉role后剩余的字符对半分，前一半为name，后一半为password
            '''
            with open('user.txt', 'a+') as f:
                f.writelines(account + '\n')
            self.user_error_dialog('注册成功!', juno)
            self.user_signin.clear()
            self.password_signin.clear()
            self.admin_lineEdit.clear()

    def user_error_dialog(self, error, juno):
        self.user_dialog = QDialog(juno)
        self.user_dialog.setFixedSize(self.screen_width*0.1, self.screen_height*0.1)
        self.user_dialog.setWindowTitle('提示')

        label = QLabel(error)

        dialog_content = QWidget(self.user_dialog)
        dialog_content.setFixedSize(self.screen_width*0.1, self.screen_height*0.1)
        dialog_layout = QVBoxLayout(dialog_content)
        dialog_layout.addWidget(label)
        dialog_layout.setAlignment(QtCore.Qt.AlignCenter)

        self.user_dialog.show()

    def Main_Window(self, juno):
        width = 0.3 * self.screen_width
        height = 0.3 * self.screen_height
        juno.setWindowTitle(Version + ' - 项目')

        self.project_table = QTableWidget()
        self.project_table.setColumnCount(1)
        self.project_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.project_table.verticalHeader().setVisible(False)
        self.project_table.horizontalHeader().setVisible(False)
        self.project_table.clicked.connect(partial(self.task_window_pre, juno))

        new_project_btn = QPushButton('新建项目')
        new_project_btn.setObjectName('xinjian')
        new_project_btn.clicked.connect(partial(self.new_project_window, juno))
        main_back_login_btn = QPushButton('返 回')
        main_back_login_btn.setObjectName('fanhui2')
        main_back_login_btn.clicked.connect(partial(self.back_to_login, juno))
        main_btn_content = QWidget()
        main_btn_layout = QHBoxLayout(main_btn_content)
        main_btn_layout.addWidget(main_back_login_btn)
        if self.role == '0':
            main_btn_layout.addWidget(new_project_btn)
        
        main_content = QWidget()
        main_content.setMinimumSize(width, height)
        main_layout = QVBoxLayout(main_content)
        main_layout.addWidget(self.project_table)
        main_layout.addWidget(main_btn_content)
        juno.setCentralWidget(main_content)
        juno.resize(width, height)
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
        width = 0.3 * self.screen_width
        height = 0.3 * self.screen_height
        juno.setWindowTitle(Version + ' - 新建项目')

        setName_label = QLabel('项目名称:')  # 新建项目界面的 ’项目名称:‘ 汉字标签
        setName_label.setFixedHeight(self.screen_height*0.023)

        self.inputName = QLineEdit()  # 新建项目界面的 项目名称文字输入框
        self.inputName.textChanged.connect(self.input_change)
        self.inputName.setFixedHeight(self.screen_height*0.023)

        input_name_content = QWidget()
        input_name_content.setFixedHeight(self.screen_height*0.033)
        input_name_layout = QHBoxLayout(input_name_content)
        input_name_layout.addWidget(setName_label)
        input_name_layout.addWidget(self.inputName)

        self.selectFile = QPushButton('导入新表')  # 新建项目界面的 ’导入新表‘ 按钮
        self.selectFile.setObjectName('daoruxinbiao')
        self.selectFile.clicked.connect(partial(self.load_new_value, juno))

        self.errorText = QTextBrowser()  # 新建项目界面的文字输出框

        noButton = QPushButton('取 消')  # 新建项目界面的 ’取消‘ 按钮
        noButton.setObjectName('quxiao')
        noButton.clicked.connect(partial(self.new_project_no, juno))

        self.yesButton = QPushButton('确 定')  # 新建项目界面的 ’确定‘ 按钮
        self.yesButton.setObjectName('queding')
        self.yesButton.clicked.connect(partial(self.new_project_yes, juno))
        self.yesButton.setDisabled(True)

        YesOrNo_content = QWidget()
        YesOrNo_layout = QHBoxLayout(YesOrNo_content)
        YesOrNo_layout.addWidget(noButton)
        YesOrNo_layout.addWidget(self.yesButton)

        new_project_content = QWidget()
        new_project_content.setMinimumSize(width, height)
        new_project_layout = QVBoxLayout(new_project_content)
        new_project_layout.addWidget(input_name_content)
        new_project_layout.addWidget(self.selectFile)
        new_project_layout.addWidget(self.errorText)
        new_project_layout.addWidget(YesOrNo_content)
        juno.setCentralWidget(new_project_content)

    def input_change(self):
        project_list = os.listdir('./JunoProject')
        if self.inputName.text().strip() != '' and self.value_temp != {} and self.inputName.text() not in project_list:
            self.yesButton.setDisabled(False)
        else:
            self.yesButton.setDisabled(True)

    def new_project_yes(self, juno):
        project_name = self.inputName.text()
        if 'win' in sys.platform:
            os.system('mkdir ' + os.getcwd() + '\JunoProject' + '\\' + project_name)
            os.system('mkdir ' + os.getcwd() + '\JunoProject' + '\\' + project_name + '\\' + 'Log')
            os.system('mkdir ' + os.getcwd() + '\JunoProject' + '\\' + project_name + '\\' + 'Task')
            os.system('copy ' + self.filename.replace('/', '\\') + ' ' + os.getcwd() + '\\' + 'JunoProject' + '\\' + project_name + '\\' + self.filename.split('/')[-1])
        elif 'linux' in sys.platform:
            cmd = ''

        with open('./JunoProject/' + project_name + '/value.json', 'w') as f:
            f.write(json.dumps(self.value_temp))

        data = algo.readJsonData('./JunoProject/港区/value.json')
        trX = algo.knnR(data, verbose=True).tolist()
        with open('./JunoProject/' + project_name + '/trx.txt', 'w') as f:
            f.write(str(trX))

        self.Main_Window(juno)

    def new_project_no(self, juno):
        self.Main_Window(juno)

    def load_kernel(self, wb):
        sheet = wb.sheet_by_index(0)
        # sheet = wb.sheet_by_name('运行数据')

        # 判断一级标题所在行数
        title_1_row = 0
        for title_1_row in range(sheet.nrows):
            count = 0
            for cell in sheet.row_values(title_1_row):
                if cell != '':
                    count += 1
            if count > 3:
                print('一级标题行：', title_1_row + 1)
                break

        # 判断单位所在行数
        unit_row = 0
        for unit_row in range(sheet.nrows):
            count = 0
            for cell in sheet.row_values(unit_row):
                if type(cell) == str and '/' in cell:
                    count += 1
            if count / sheet.row_len(unit_row) > 0.5:
                print('单位行：', unit_row + 1)
                break

        # 从一级标题行开始到单位行之前，智能填充其间空格的原本单位(不能无脑沿用上一次的单位，还要考虑父标题是否相同才行)
        titles = []
        for i in range(title_1_row, unit_row):
            title = []
            last_h = ''
            if i == title_1_row:
                for j in sheet.row_values(i):
                    if j != '':
                        title.append(j)
                        last_h = j
                    else:
                        title.append(last_h)
            else:
                index = 0
                last_v = ''
                for j in sheet.row_values(i):
                    if j != '':
                        title.append(j)
                        last_h = j
                        last_v = titles[-1][index]
                    else:
                        if titles[-1][index] == last_v:
                            title.append(last_h)
                        else:
                            title.append('')
                            last_h = ''
                    index += 1
            titles.append(title)

        # 转置
        titles = np.array(titles).transpose().tolist()

        # 去除二三级标题中重复的名称和空值，以及对标题内字符的规范化
        titles_final = []
        remark_flag = False
        for title in titles:
            title_new = ''
            for cell in title:
                if '\n' in cell:
                    cell = cell.replace("\n", '')
                if '\t' in cell:
                    cell = cell.replace("\t", '')
                if '（' in cell:
                    cell = cell.replace("（", '(')
                if '）' in cell:
                    cell = cell.replace("）", ')')
                if '--' in cell:
                    cell = cell.replace("--", '-')
                if '  ' in cell:
                    cell = re.sub(r'  +', '', cell)
                if cell != '':
                    title_new += cell + "_"
            titles_final.append(title_new[:-1])

        # 添加单位
        unit = []
        for i in sheet.row_values(unit_row):
            unit.append(i)
        for i in range(len(titles_final)):
            if unit[i] != '':
                titles_final[i] = titles_final[i] + " " + unit[i]

        # 去除备注列
        if titles_final[-1] == '备注':
            remark_flag = True
            titles_final.remove('备注')

        def col2az(col):
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
                if row[block_count] == '-' or row[block_count] == '—' or row[block_count] == '——' or row[
                    block_count] == '－' or \
                        row[block_count] == '—' or row[block_count] == '——' or row[block_count] == '-' or row[
                    block_count] == '_' or row[block_count] == '－' or row[block_count] == '＿':
                    row[block_count] = ''
                if row[block_count] == '\\' or row[block_count] == '/' or row[block_count] == '＼' or row[
                    block_count] == '／' or \
                        row[block_count] == '|' or row[block_count] == '｜':
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
                    error = str(row_index + 1) + " 行 " + col2az(col) + " 列，参数 \'" + str(a) + "\' 格式错误！\n"
                    error_per_row += error
            return error_per_row

        # 获取数据部分的开始行
        data_start = 0
        for i in sheet.col_values(0):
            if type(i) == float:
                break
            data_start += 1
        print('数据开始行：', data_start + 1)

        matrix = []
        error_total = ''
        # 将表格的数据部分组合成一个二维list，并对每个数据进行格式检查
        for row_num in range(data_start, sheet.nrows):
            row = [i for i in sheet.row_values(row_num)]
            if remark_flag:
                row = row[:-1]  # 去除备注
            row = value_rectify(row)
            error_total += value_check(row_num, row)
            matrix.append(row)

        # 二维数据表xy转置成二维list: a天每天记录b种数据 变成 b种数据每种记录a天
        matrix_trans = np.array(matrix).transpose().tolist()

        data = {}
        index = 0
        # 去除整列都没有数据的指标
        for a in matrix_trans:
            if ''.join(a).strip():
                data.update({titles_final[index]: a})
            index += 1

        return data, error_total, titles_final

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
            try:
                self.errorText.setText('未读取数据!')
            except:
                print('未重载任何数据!')
            return

        data, error_total, titles_final = self.load_kernel(wb)

        if error_total != '':
            self.errorText.setText(error_total)
            self.errorText.append('\n文件导入失败!')
            return

        self.value_temp = data

        success_text = '数据导入成功!'
        self.errorText.setText(success_text)
        self.selectFile.setText(self.filename)
        self.input_change()
        self.build_tree_combo(self.value_temp)

    def back_to_login(self, juno):
        self.login_window(juno)

    def task_window_pre(self, juno):
        project_name = self.project_table.currentItem().text()

        if self.role == '0' or self.role == '1':
            self.task_window(project_name, juno)
        elif self.role == '2':
            task_dir = os.listdir('./JunoProject/' + project_name + '/Task/')
            if operator_limit_task + '.txt' in task_dir:
                juno.close()
                juno = QMainWindow()
                self.config_window(project_name, juno, operator_limit_task)
            else:
                self.error_dialog(juno, '操作人员命题不存在!')

    def task_window(self, project_name, juno):
        width = 0.3 * self.screen_width
        height = 0.3 * self.screen_height
        juno.setWindowTitle(Version + ' - 命题')

        with open('./JunoProject/' + project_name + '/value.json', 'r') as f:
            self.value = json.load(f)
            self.build_tree_combo(self.value)

        ai_btn = QPushButton('AI生成命题')
        ai_btn.setObjectName('AIyuce')
        ai_btn.clicked.connect(partial(self.ai_task_window, juno, project_name))
        input_btn = QPushButton('手动输入命题')
        input_btn.setObjectName('shoudongshuru')
        input_btn.clicked.connect(partial(self.input_task_window, juno, project_name))
        task_btn_content = QWidget()
        task_btn_layout = QHBoxLayout(task_btn_content)
        task_btn_layout.addWidget(ai_btn)
        task_btn_layout.addWidget(input_btn)

        self.task_table = QTableWidget()  # 命题界面的 命题名称展示表格
        self.task_table.setColumnCount(1)
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.task_table.clicked.connect(partial(self.click_config_window, project_name, juno))
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.horizontalHeader().setVisible(False)

        reload_btn = QPushButton('更新表格')
        reload_btn.setObjectName('gengxin')
        reload_btn.clicked.connect(partial(self.reload_btn, juno, project_name))
        task_back_btn = QPushButton('返 回')  # 命题界面的 ’返回‘ 按钮
        task_back_btn.setObjectName('fanhui3')
        task_back_btn.clicked.connect(partial(self.task_back_btn, juno))
        task_flow_content = QWidget()
        task_flow_layout = QHBoxLayout(task_flow_content)
        task_flow_layout.addWidget(task_back_btn)
        task_flow_layout.addWidget(reload_btn)

        task_content = QWidget()
        task_content.setMinimumSize(width, height)
        task_layout = QVBoxLayout(task_content)
        task_layout.addWidget(task_btn_content)
        task_layout.addWidget(self.task_table)
        task_layout.addWidget(task_flow_content)

        juno.setCentralWidget(task_content)
        juno.setFixedHeight(height)
        juno.setFixedWidth(width)
        self.load_task_table(project_name)

    def ai_task_window(self, juno, project_name):
        self.ai_dialog = QDialog(juno)
        self.ai_dialog.setFixedSize(self.screen_width*0.25, self.screen_height*0.25)
        self.ai_dialog.setWindowTitle('AI预测命题')

        self.ai_combo_tree = QComboBox()
        self.ai_combo_tree.setObjectName('combobox')
        self.ai_combo_tree.setModel(self.treeModel)
        self.ai_combo_tree.setView(self.ai_treeView)
        self.ai_combo_tree.activated.connect(partial(self.ai_combo_change, juno, 0))
        self.ai_table = QTableWidget()
        self.ai_table.setColumnCount(2)
        self.ai_table.verticalHeader().setVisible(False)
        self.ai_table.setHorizontalHeaderLabels(['命题所涉及的调控指标', '清空全部'])
        self.ai_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ai_table.horizontalHeader().sectionDoubleClicked.connect(partial(self.ai_table_clear, juno))
        self.ai_btn = QPushButton('AI预测')
        self.ai_btn.setObjectName('yuce')
        self.ai_btn.clicked.connect(partial(self.ai_calc, project_name))
        ai_content = QWidget(self.ai_dialog)
        ai_content.setFixedSize(self.screen_width*0.25, self.screen_height*0.25)
        ai_layout = QFormLayout(ai_content)
        ai_layout.addRow(self.ai_combo_tree)
        ai_layout.addRow(self.ai_table)
        ai_layout.addRow(self.ai_btn)

        self.ai_dialog.show()

        data = algo.readJsonData('JunoProject/' + project_name + '/value.json')
        vars = algo.detectYvars(data)
        for i in vars:
            self.ai_table_in(juno, i)

    def ai_table_clear(self, juno):
        self.ai_table_clear_dialog = QDialog(juno)
        self.ai_table_clear_dialog.resize(self.screen_width * 0.2, self.screen_height * 0.2)
        self.ai_table_clear_dialog.setWindowTitle('清空?')
        self.ai_table_clear_dialog.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        label = QLabel()
        label.setText('是否确认清空？')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        self.ai_table_clear_dialog.show()

        yes_btn = QPushButton('删 除')
        yes_btn.setObjectName('shanchu')
        yes_btn.clicked.connect(self.ai_table_clear_yes)
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao')
        no_btn.clicked.connect(self.ai_table_clear_no)
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(yes_btn)
        flow_layout.addWidget(no_btn)

        if self.ai_table.currentColumn() == 1:
            layout = QVBoxLayout(self.ai_table_clear_dialog)
            layout.addWidget(label)
            layout.addWidget(flow_content)

    def ai_table_clear_no(self):
        self.ai_table_clear_dialog.close()

    def ai_table_clear_yes(self):
        self.ai_table_clear_dialog.close()
        self.ai_table.setRowCount(0)

    def ai_combo_change(self, juno, ai_current):
        if type(ai_current) is int:
            current_index = []
            current_index.append(self.ai_treeView.currentIndex().row())
            next_item = self.ai_treeView.currentIndex().parent()
            while next_item.row() != -1:
                current_index.append(next_item.row())
                next_item = next_item.parent()
            current_index.reverse()

            datas = []
            item = self.ai_treeView.currentIndex()
            for i in range(len(current_index)):
                datas.append(self.ai_treeView.model().itemData(item)[0])
                item = item.parent()
            datas.reverse()

            ai_current = ''
            for i in datas:
                ai_current += i + '_'
            ai_current = ai_current.rstrip('_')

        table_list = []
        for i in range(self.ai_table.rowCount()):
            table_list.append(self.ai_table.item(i, 0).text())
        if ai_current in table_list:
            self.error_dialog(juno, '重复添加!')
            return

        self.ai_table_in(juno, ai_current)

    def ai_table_in(self, juno, ai_current):
        item = QTableWidgetItem(ai_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.ai_table.insertRow(self.ai_table.rowCount())
        self.ai_table.setItem(self.ai_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删 除')
        cancel_button.setObjectName('shanchu')
        cancel_button.clicked.connect(partial(self.ai_combo_cancel_pre, juno))
        self.ai_table.setCellWidget(self.ai_table.rowCount() - 1, 1, cancel_button)

        self.ai_table.resizeRowsToContents()

    def ai_combo_cancel_pre(self, juno):
        self.ai_combo_cancel_dialog = QDialog(juno)
        self.ai_combo_cancel_dialog.resize(self.screen_width*0.2, self.screen_height*0.2)
        self.ai_combo_cancel_dialog.setWindowTitle('删除?')
        self.ai_combo_cancel_dialog.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        label = QLabel()
        label.setText('是否确认删除？')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        self.ai_combo_cancel_dialog.show()

        yes_btn = QPushButton('删 除')
        yes_btn.setObjectName('shanchu')
        yes_btn.clicked.connect(self.ai_combo_cancel)
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao')
        no_btn.clicked.connect(self.ai_combo_cancel_no)
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(yes_btn)
        flow_layout.addWidget(no_btn)

        layout = QVBoxLayout(self.ai_combo_cancel_dialog)
        layout.addWidget(label)
        layout.addWidget(flow_content)

    def ai_combo_cancel_no(self):
        self.ai_combo_cancel_dialog.close()

    def ai_combo_cancel(self):
        self.ai_combo_cancel_dialog.close()
        index = self.ai_table.currentRow()
        self.ai_table.removeRow(index)

    def ai_calc(self, project_name):
        vars = []
        for i in range(self.ai_table.rowCount()):
            vars.append(self.ai_table.item(i, 0).text())

        data = algo.readJsonData('JunoProject/' + project_name + '/value.json')
        trX = algo.knnR(data, verbose=True)
        morfi = algo.MORFI(trX, data, vars)
        xyz = morfi.XYZ(vars)
        for i in xyz:
            print(i)
            name = 'AI预测_' + i['name']
            if '\\' in name:
                name = name.replace('\\', '')
            if '/' in name:
                name = name.replace('/', '')
            if ':' in name:
                name = name.replace(':', '')
            if '*' in name:
                name = name.replace('*', '')
            if '"' in name:
                name = name.replace('"', '')
            if '<' in name:
                name = name.replace('<', '')
            if '>' in name:
                name = name.replace('>', '')
            if '?' in name:
                name = name.replace('?', '')
            if '|' in name:
                name = name.replace('|', '')

            x = i['inputVars']
            y = i['controlVars']
            z = i['outputVars']
            thresholds = i['thresholds']
            safety = i['safety']

            typedefs = i['typeDefs'].copy()
            while -1 in typedefs:
                typedefs.remove(-1)

            typedefs_cost = typedefs[0:len(y)]
            typedefs_cost_sum = sum(typedefs_cost)
            typedefs_cost_update = []
            for j in typedefs_cost:
                typedefs_cost_update.append(j*100.0/typedefs_cost_sum)

            typedefs_risk = typedefs[len(y):len(y)+len(z)]
            typedefs_risk_update = []
            for j in typedefs_risk:
                typedefs_risk_update.append((j-2)*100.0)

            cost = {}
            risk = {}
            border = {}
            for j in range(len(y)):
                cost.update({y[j]: str(typedefs_cost_update[j])})
            for j in range(len(z)):
                risk.update({z[j]: str(typedefs_risk_update[j])})
                border.update({z[j]: thresholds[j]})

            task_config = {}
            task_config.update({'situation': x})
            task_config.update({'action': y})
            task_config.update({'result': z})
            task_config.update({'cost': cost})
            task_config.update({'risk': risk})
            task_config.update({'border': border})
            task_config.update({'begin': int(float(min(self.value['日期'])))})
            task_config.update({'end': int(float(max(self.value['日期'])))})
            task_config.update({'safety': safety})

            cipher = self.encrypt(task_config)
            with open('./JunoProject/' + project_name + '/Task/' + name + '.txt', 'w+') as f:
                f.write(cipher.decode())

        self.load_task_table(project_name)
        self.ai_dialog.close()

    def input_task_window(self, juno, project_name):
        self.task_input_dialog = QDialog(juno)
        self.task_input_dialog.setFixedSize(self.screen_width * 0.25, self.screen_height * 0.035)
        self.task_input_dialog.setWindowTitle('手动输入命题')

        input_label = QLabel('手动添加: ')  # 命题界面的 ’添加命题:‘ 文字标签
        input_label.setFixedHeight(self.screen_height * 0.023)
        self.task_lineEdit = QLineEdit()  # 命题界面的 命题名称 输入框
        self.task_lineEdit.setFixedHeight(self.screen_height * 0.023)
        self.task_lineEdit.textChanged.connect(partial(self.task_textedit_change, project_name))
        self.submit_task_btn = QPushButton('确定添加')  # 命题界面的 ’确定添加‘ 按钮
        self.submit_task_btn.setObjectName('quedingtianjia')
        self.submit_task_btn.setFixedHeight(self.screen_height * 0.023)
        self.submit_task_btn.clicked.connect(partial(self.task_submit, project_name))
        self.submit_task_btn.setDisabled(True)
        task_input_content = QWidget(self.task_input_dialog)
        task_input_content.setFixedSize(self.screen_width*0.25, self.screen_height*0.035)
        task_input_layout = QHBoxLayout(task_input_content)
        task_input_layout.addWidget(input_label)
        task_input_layout.addWidget(self.task_lineEdit)
        task_input_layout.addWidget(self.submit_task_btn)

        self.task_input_dialog.show()

    def task_textedit_change(self, project_name):
        task_name = self.task_lineEdit.text().strip()
        task_list = os.listdir('./JunoProject/' + project_name + '/Task/')
        if task_name == '':
            self.submit_task_btn.setDisabled(True)
            return
        if task_name + '.txt' in task_list:
            self.submit_task_btn.setDisabled(True)
            return
        self.submit_task_btn.setDisabled(False)

    def task_submit(self, project_name):
        task_name = self.task_lineEdit.text()
        var = {"situation": [], "action": [], "result": [], "cost": {}, "risk": {}, "border": {}, "begin": int(float(min(self.value['日期']))), "end": int(float(max(self.value['日期']))), "safety": "0.5"}
        cipher = self.encrypt(var)
        with open('./JunoProject/' + project_name + '/Task/' + task_name + '.txt', 'w') as f:
            f.write(cipher.decode())
        self.task_lineEdit.clear()
        self.load_task_table(project_name)
        self.task_input_dialog.close()

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

    def reload_btn(self, juno, project_name):
        option = QFileDialog.Option()
        option |= QFileDialog.DontUseNativeDialog
        self.reload_filename, _ = QFileDialog.getOpenFileName(
            juno, '选择运行数据', './SourceTable',
            '所有文件 (*);;xlsx文件 (*.xlsx);;xls文件 (*.xls)',
            options=option
        )

        try:
            wb = xlrd.open_workbook(self.reload_filename, encoding_override='gbk')
        except UnicodeDecodeError:
            wb = xlrd.open_workbook(self.reload_filename, encoding_override='utf-8')
        except FileNotFoundError:
            print('未重载任何数据!')
            return

        data, error_total, titles_final = self.load_kernel(wb)

        if error_total != '':
            self.reload_error_dialog(error_total + '\n文件导入失败!', juno)
            return

        self.reload_value_temp = data
        day_start = min(self.reload_value_temp['日期'])
        day_end = max(self.reload_value_temp['日期'])

        task_error = ''
        task_list = os.listdir('./JunoProject/' + project_name + '/Task/')
        for i in task_list:
            with open('./JunoProject/' + project_name + '/Task/' + i, 'r') as f:
                text = self.decrypt(f.read().encode())
                task_value = json.loads(text.replace('\'', '"'))
            for j in task_value['situation']:
                if j not in titles_final:
                    error = '项目:' + project_name + '\t命题:' + i.split('.')[0] + '\t选项:情况\t指标:' + j + '\n'
                    task_error += (error)
            for j in task_value['action']:
                if j not in titles_final:
                    error = '项目:' + project_name + '\t命题:' + i.split('.')[0] + '\t选项:行为\t指标:' + j + '\n'
                    task_error += (error)
            for j in task_value['result']:
                if j not in titles_final:
                    error = '项目:' + project_name + '\t命题:' + i.split('.')[0] + '\t选项:结果\t指标:' + j + '\n'
                    task_error += (error)

        if task_error != '':
            self.reload_error_dialog(task_error.rstrip('\n'), juno)
        else:
            # 删除project文件夹内的老excel表，导入新excel表
            file_list = os.listdir('./JunoProject/' + project_name)
            for i in file_list:
                if i.endswith('.xlsx') or i.endswith('.xls'):
                    os.remove('./JunoProject/' + project_name + '/' + i)
            shutil.copyfile(self.reload_filename.replace('/', '\\'), os.getcwd() + '\\' + 'JunoProject' + '\\' + project_name + '\\' + self.reload_filename.split('/')[-1])

            # 将新excel表的数据 self.reload_value_temp 覆盖进 value.json
            with open('./JunoProject/' + project_name + '/value.json', 'w') as f:
                f.write(json.dumps(self.reload_value_temp))

            # 覆写新trX
            data = algo.readJsonData('./JunoProject/' + project_name + '/value.json')
            trX = algo.knnR(data, verbose=True).tolist()
            with open('./JunoProject/' + project_name + '/trx.txt', 'w') as f:
                f.write(str(trX))

            # 还要修改数据日期的起始与终止到task的json中！
            for task in task_list:
                with open('./JunoProject/' + project_name + '/Task/' + task, 'r') as f:
                    text = self.decrypt(f.read().encode())
                    temp = json.loads(text.replace('\'', '"'))
                temp.update({'begin': int(float(day_start))})
                temp.update({'end': int(float(day_end))})

                cipher = self.encrypt(temp)
                with open('./JunoProject/' + project_name + '/Task/' + task, 'w') as f:
                    f.write(cipher.decode())

            # 将新excel表的数据 self.reload_value_temp 覆盖给 self.value
            self.value = self.reload_value_temp



    def reload_error_dialog(self, text, juno):
        self.reload_dialog = QDialog(juno)
        self.reload_dialog.setFixedSize(500, 300)
        self.reload_dialog.setWindowTitle('警告!')

        text_broswer = QTextBrowser()
        text_broswer.setText(text)

        dialog_content = QWidget(self.reload_dialog)
        dialog_content.setFixedSize(500, 300)
        dialog_layout = QVBoxLayout(dialog_content)
        dialog_layout.addWidget(text_broswer)

        if '指标' in text:
            text_broswer.setText(text_broswer.toPlainText() + '\n\n以上指标在新表中不存在，是否选择强制更新？\n若选择强制更新，则相关命题将会删除以上指标!')

            yes_btn = QPushButton('强制更新')
            yes_btn.clicked.connect(partial(self.reload_yes, text))
            no_btn = QPushButton('取 消')
            no_btn.setObjectName('quxiao')
            no_btn.clicked.connect(self.reload_no)

            dialog_layout.addWidget(yes_btn)
            dialog_layout.addWidget(no_btn)

        self.reload_dialog.show()

    def reload_yes(self, text):
        day_start = min(self.reload_value_temp['日期'])
        day_end = max(self.reload_value_temp['日期'])
        # 更新project文件夹内的命题json
        project_name = ''
        for i in text.split('\n'):
            info = i.split('\t')
            project_name = info[0].split(':')[1]
            task_name = info[1].split(':')[1] + '.txt'
            target_name = info[3].split(':')[1]
            with open('./JunoProject/' + project_name + '/Task/' + task_name, 'r') as f:
                clear_text = self.decrypt(f.read().encode())
                task_load = json.loads(clear_text.replace('\'', '"'))
            for j in task_load.keys():
                if type(task_load[j]) is not int and type(task_load[j]) is not float and type(task_load[j]) is not str and target_name in task_load[j]:
                    try:
                        task_load[j].pop(target_name)
                    except:
                        task_load[j].remove(target_name)
            task_load.update({'begin': int(float(day_start))})
            task_load.update({'end': int(float(day_end))})
            cipher = self.encrypt(task_load)
            with open('./JunoProject/' + project_name + '/Task/' + task_name, 'w') as f:
                f.write(cipher.decode())

        # 删除project文件夹内的老excel表，导入新excel表
        file_list = os.listdir('./JunoProject/' + project_name)
        for i in file_list:
            if i.endswith('.xlsx') or i.endswith('.xls'):
                os.remove('./JunoProject/' + project_name + '/' + i)
        shutil.copyfile(self.reload_filename.replace('/', '\\'), os.getcwd() + '\\' + 'JunoProject' + '\\' + project_name + '\\' + self.reload_filename.split('/')[-1])

        # 将新excel表的数据 self.reload_value_temp 覆盖进 value.json
        with open('./JunoProject/' + project_name + '/value.json', 'w') as f:
            f.write(json.dumps(self.reload_value_temp))

        # 覆写新trX
        data = algo.readJsonData('./JunoProject/' + project_name + '/value.json')
        trX = algo.knnR(data, verbose=True).tolist()
        with open('./JunoProject/' + project_name + '/trx.txt', 'w') as f:
            f.write(str(trX))

        # 将新excel表的数据 self.reload_value_temp 覆盖给 self.value
        self.value = self.reload_value_temp

        self.build_tree_combo(self.value)

        self.reload_dialog.close()

    def reload_no(self):
        self.reload_dialog.close()

    def task_back_btn(self, juno):
        self.Main_Window(juno)

    def build_tree_combo(self, value):
        global count_2, item_1, count_3, item_2, item_3
        self.situation_treeView = QTreeView()
        self.situation_treeView.setHeaderHidden(True)
        self.action_treeView = QTreeView()
        self.action_treeView.setHeaderHidden(True)
        self.result_treeView = QTreeView()
        self.result_treeView.setHeaderHidden(True)
        self.details_treeView = QTreeView()
        self.details_treeView.setHeaderHidden(True)
        self.ai_treeView = QTreeView()
        self.ai_treeView.setHeaderHidden(True)
        self.treeModel = QStandardItemModel()
        rootNode = self.treeModel.invisibleRootItem()

        count_1 = -1
        for i in value.keys():
            if i == '日期':
                continue

            i = i.split('_')

            dir_1 = []
            for j in range(rootNode.rowCount()):
                dir_1.append(rootNode.child(j, 0).text())
            if i[0] not in dir_1:
                item_1 = QStandardItem()
                item_1.setText(i[0])
                rootNode.appendRow(item_1)
                count_1 += 1
                count_2 = -1

            dir_2 = []
            try:
                item_1 = rootNode.child(count_1, 0)
                item_1.setSelectable(False)
                for j in range(item_1.rowCount()):
                    dir_2.append(item_1.child(j, 0).text())
                if i[1] not in dir_2:
                    item_2 = QStandardItem()
                    item_2.setText(i[1])
                    item_1.appendRow(item_2)
                    count_2 += 1
                    count_3 = -1
            except IndexError:
                item_1.setSelectable(True)
                pass

            dir_3 = []
            try:
                item_2 = item_1.child(count_2, 0)
                item_2.setSelectable(False)
                for j in range(item_2.rowCount()):
                    dir_3.append(item_2.child(j, 0).text())
                if i[2] not in dir_3:
                    item_3 = QStandardItem()
                    item_3.setText(i[2])
                    item_2.appendRow(item_3)
                    count_3 += 1
            except IndexError:
                item_2.setSelectable(True)
                pass
            except AttributeError:
                pass

            dir_4 = []
            try:
                item_3 = item_2.child(count_3, 0)
                item_3.setSelectable(False)
                for j in range(item_3.rowCount()):
                    dir_4.append(item_3.child(j, 0).text())
                if i[3] not in dir_4:
                    item_4 = QStandardItem()
                    item_4.setText(i[3])
                    item_3.appendRow(item_4)
            except IndexError:
                item_3.setSelectable(True)
                pass
            except AttributeError:
                pass

    def click_config_window(self, project_name, juno):
        juno.close()
        juno = QMainWindow()

        self.menubar = QMenuBar(juno)
        self.menuData = QMenu(self.menubar)
        self.menuData.setTitle('数据')
        self.menuData.addAction('查看原始数据', partial(self.details, juno, project_name))
        self.menubar.addAction(self.menuData.menuAction())
        juno.setMenuBar(self.menubar)
        self.config_window(project_name, juno)

    def config_window(self, project_name, juno, *task_name_1):
        juno.setWindowTitle(Version + ' - 配置参数')

        try:
            task_name = self.task_table.currentItem().text()
        except:
            task_name = task_name_1[0]

        with open('./JunoProject/' + project_name + '/value.json', 'r') as f:
            self.value = json.load(f)
            self.build_tree_combo(self.value)

        self.situation_combo_tree = QComboBox()
        self.situation_combo_tree.setObjectName('combobox')
        self.situation_combo_tree.setModel(self.treeModel)
        self.situation_combo_tree.setView(self.situation_treeView)
        self.situation_combo_tree.activated.connect(partial(self.situation_combo_change, juno, project_name, task_name, 0))
        self.situation_table = QTableWidget()
        self.situation_table.setColumnCount(2)
        self.situation_table.verticalHeader().setVisible(False)
        self.situation_table.setHorizontalHeaderLabels(['情况选择', ''])
        self.situation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        situation_content = QWidget()
        situation_layout = QFormLayout(situation_content)
        situation_layout.addRow(self.situation_combo_tree)
        situation_layout.addRow(self.situation_table)

        self.action_combo_tree = QComboBox()
        self.action_combo_tree.setObjectName('combobox')
        self.action_combo_tree.setModel(self.treeModel)
        self.action_combo_tree.setView(self.action_treeView)
        self.action_combo_tree.activated.connect(partial(self.action_combo_change, juno, project_name, task_name, 0))
        self.action_table = QTableWidget()
        self.action_table.setColumnCount(2)
        self.action_table.verticalHeader().setVisible(False)
        self.action_table.setHorizontalHeaderLabels(['行为选择', ''])
        self.action_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        action_content = QWidget()
        action_layout = QFormLayout(action_content)
        action_layout.addRow(self.action_combo_tree)
        action_layout.addRow(self.action_table)

        self.result_combo_tree = QComboBox()
        self.result_combo_tree.setObjectName('combobox')
        self.result_combo_tree.setModel(self.treeModel)
        self.result_combo_tree.setView(self.result_treeView)
        self.result_combo_tree.activated.connect(partial(self.result_combo_change, juno, project_name, task_name, 0))
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setHorizontalHeaderLabels(['结果选择', ''])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_content = QWidget()
        result_layout = QFormLayout(result_content)
        result_layout.addRow(self.result_combo_tree)
        result_layout.addRow(self.result_table)

        var_content = QWidget()
        var_content.setFixedWidth(self.screen_width*0.49)
        var_layout = QHBoxLayout(var_content)
        var_layout.addWidget(situation_content)
        var_layout.addWidget(action_content)
        var_layout.addWidget(result_content)

        cost_table_label = QLabel('成本占比:')

        self.cost_table = QTableWidget()
        self.cost_table.setColumnCount(2)
        self.cost_table.verticalHeader().setVisible(False)
        self.cost_table.setHorizontalHeaderLabels(['行为变量', '成本占比 %'])
        self.cost_table.cellChanged.connect(partial(self.cost_table_change, project_name, task_name))
        self.cost_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        cost_table_content = QWidget()
        cost_table_layout = QVBoxLayout(cost_table_content)
        cost_table_layout.addWidget(cost_table_label)
        cost_table_layout.addWidget(self.cost_table)

        config_content_1 = QWidget()
        config_content_1.setFixedHeight(self.screen_height*0.4)
        config_layout_1 = QHBoxLayout(config_content_1)
        config_layout_1.addWidget(var_content)
        config_layout_1.addWidget(cost_table_content)

        risk_table_label = QLabel('  重要性占比:')
        risk_table_label.setFixedHeight(self.screen_height*0.029)

        self.risk_table = QTableWidget()
        self.risk_table.setColumnCount(2)
        self.risk_table.verticalHeader().setVisible(False)
        self.risk_table.setHorizontalHeaderLabels(['结果变量', '重要性占比 %'])
        self.risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.risk_table.clicked.connect(partial(self.risk_to_border, juno, project_name, task_name))
        self.risk_table.cellChanged.connect(partial(self.risk_table_change, project_name, task_name))

        risk_table_content = QWidget()
        risk_table_layout = QVBoxLayout(risk_table_content)
        risk_table_layout.addWidget(risk_table_label)
        risk_table_layout.addWidget(self.risk_table)

        border_table_label = QLabel('边界设置:')

        self.border_add_btn = QPushButton('添 加 边 界')
        self.border_add_btn.setObjectName('tianjiabianjie')
        self.border_add_btn.setFixedSize(self.screen_width*0.11, self.screen_height*0.025)
        self.border_add_btn.clicked.connect(partial(self.border_table_add_row, juno, project_name, task_name))
        self.border_add_btn.setDisabled(True)

        label_1_content = QWidget()
        label_1_layout = QHBoxLayout(label_1_content)
        label_1_layout.addWidget(border_table_label)
        label_1_layout.addWidget(self.border_add_btn)

        self.border_table = QTableWidget()
        self.border_table.setFixedHeight(self.screen_height*0.17)
        self.border_table.setColumnCount(3)
        self.border_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.border_table.cellChanged.connect(partial(self.borber_table_cell_change, project_name, task_name))
        self.border_table.verticalHeader().setVisible(False)
        self.border_table.setHorizontalHeaderLabels(['', '', ''])
        self.border_table_graph = pyqtgraph.GraphicsLayoutWidget()
        self.border_table_graph.setBackground('#182037') #边界图：背景颜色
        self.border_table_graph_plt = self.border_table_graph.addPlot()

        border_table_content = QWidget()
        border_table_layout = QVBoxLayout(border_table_content)
        border_table_layout.addWidget(label_1_content)
        border_table_layout.addWidget(self.border_table)
        border_table_layout.addWidget(self.border_table_graph)

        config_content_2 = QWidget()
        config_layout_2 = QHBoxLayout(config_content_2)
        config_layout_2.addWidget(risk_table_content)
        config_layout_2.addWidget(border_table_content)

        begin_label = QLabel('起始时间:')
        begin_label.setFixedHeight(self.screen_height*0.023)

        self.date_begin = QDateEdit()
        self.date_begin.setFixedHeight(self.screen_height*0.023)
        self.date_begin.setCalendarPopup(True)
        self.date_begin.setDate(date.fromordinal(int(float(min(self.value['日期']))) + date(1900, 1, 1).toordinal() - 2))

        end_label = QLabel('   结束时间:')
        end_label.setFixedHeight(self.screen_height*0.023)

        self.date_end = QDateEdit()
        self.date_end.setFixedHeight(self.screen_height*0.023)
        self.date_end.setCalendarPopup(True)
        self.date_end.setDate(date.fromordinal(int(float(max(self.value['日期']))) + date(1900, 1, 1).toordinal() - 2))

        label = QLabel('   safety:')
        label.setFixedHeight(self.screen_height*0.023)

        self.safety_lineEdit = QLineEdit('0.5')
        self.safety_lineEdit.setFixedHeight(self.screen_height*0.023)

        content_1 = QWidget()
        content_1.setFixedHeight(self.screen_height*0.033)
        layout_1 = QHBoxLayout(content_1)
        layout_1.addWidget(begin_label)
        layout_1.addWidget(self.date_begin)
        layout_1.addWidget(end_label)
        layout_1.addWidget(self.date_end)
        layout_1.addWidget(label)
        layout_1.addWidget(self.safety_lineEdit)

        back_btn = QPushButton('返 回')
        back_btn.setObjectName('fanhui')
        back_btn.setFixedWidth(self.screen_width * 0.06)
        back_btn.clicked.connect(partial(self.config_back, project_name, juno))

        finish_btn = QPushButton('计 算')
        finish_btn.setObjectName('jisuan')
        finish_btn.setFixedWidth(self.screen_width * 0.06)
        finish_btn.clicked.connect(partial(self.config_calc, juno, project_name, task_name))

        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(back_btn)
        flow_layout.addWidget(finish_btn)

        config_content = QWidget()
        config_layout = QVBoxLayout(config_content)
        config_layout.addWidget(config_content_1)
        config_layout.addWidget(config_content_2)
        config_layout.addWidget(content_1)
        config_layout.addWidget(flow_content)

        juno.showMaximized()

        self.load_config_window(juno, project_name, task_name)

        if self.risk_table.rowCount() > 0:
            self.risk_to_border_first(juno, project_name, task_name)

        if self.role == '0' or self.role == '1':
            juno.setCentralWidget(config_content)
        elif self.role == '2':
            self.config_calc(juno, project_name, task_name)

    def load_config_window(self, juno, project_name, task_name):
        with open('./JunoProject/' + project_name + '/Task/' + task_name + '.txt', 'r') as f:
            text = self.decrypt(f.read().encode())
            self.config_data = json.loads(text.replace('\'', '"'))
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
            self.situation_combo_change(juno, project_name, task_name, i)
        for i in self.action_data:
            self.action_combo_change(juno, project_name, task_name, i)
        for i in self.result_data:
            self.result_combo_change(juno, project_name, task_name, i)

        if self.begin_data is not None:
            self.date_begin.setDate(date.fromordinal(int(float(self.begin_data)) + date(1900, 1, 1).toordinal() - 2))
        if self.end_data is not None:
            self.date_end.setDate(date.fromordinal(int(float(self.end_data)) + date(1900, 1, 1).toordinal() - 2))
        if self.safety_data is not None:
            self.safety_lineEdit.setText(self.safety_data)

    def risk_to_border_first(self, juno, project_name, task_name):
        border_name = self.risk_table.item(0, 0).text()
        self.border_add_btn.setDisabled(False)

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

        self.border_table.clear()
        self.border_table.setRowCount(0)
        self.border_table.setHorizontalHeaderLabels([border_name + '--分布 %', border_name + '--边界 (值不要相同)', ''])

        if border_name in self.border_data:
            border_info = self.border_data[border_name]
            count = 0
            for i in border_info.keys():
                self.border_table.setRowCount(self.border_table.rowCount() + 1)
                item = QTableWidgetItem(i)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.border_table.setItem(count, 0, item)
                item = QTableWidgetItem(border_info[i])
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.border_table.setItem(count, 1, item)
                delete_btn = QPushButton('删 除')
                delete_btn.setObjectName('shanchu')
                delete_btn.clicked.connect(partial(self.border_table_delete_pre, juno, project_name, task_name))
                self.border_table.setCellWidget(count, 2, delete_btn)
                count += 1

    def situation_combo_change(self, juno, project_name, task_name, situation_current):
        if type(situation_current) is int:
            current_index = []
            current_index.append(self.situation_treeView.currentIndex().row())
            next_item = self.situation_treeView.currentIndex().parent()
            while next_item.row() != -1:
                current_index.append(next_item.row())
                next_item = next_item.parent()
            current_index.reverse()

            datas = []
            item = self.situation_treeView.currentIndex()
            for i in range(len(current_index)):
                datas.append(self.situation_treeView.model().itemData(item)[0])
                item = item.parent()
            datas.reverse()

            situation_current = ''
            for i in datas:
                situation_current += i + '_'
            situation_current = situation_current.rstrip('_')

        table_list = []
        for i in range(self.situation_table.rowCount()):
            table_list.append(self.situation_table.item(i, 0).text())
        for i in range(self.action_table.rowCount()):
            table_list.append(self.action_table.item(i, 0).text())
        for i in range(self.result_table.rowCount()):
            table_list.append(self.result_table.item(i, 0).text())
        if situation_current in table_list:
            self.error_dialog(juno, '重复添加!')
            return

        item = QTableWidgetItem(situation_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.situation_table.insertRow(self.situation_table.rowCount())
        self.situation_table.setItem(self.situation_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删 除')
        cancel_button.setObjectName('shanchu')
        cancel_button.clicked.connect(partial(self.situation_combo_cancel_pre, juno, project_name, task_name))
        self.situation_table.setCellWidget(self.situation_table.rowCount() - 1, 1, cancel_button)

        self.situation_table.resizeRowsToContents()
        self.config_save(project_name, task_name)

    def situation_combo_cancel_pre(self, juno, project_name, task_name):
        self.situation_combo_cancel_dialog = QDialog(juno)
        self.situation_combo_cancel_dialog.resize(self.screen_width*0.2, self.screen_height*0.2)
        self.situation_combo_cancel_dialog.setWindowTitle('删除!')

        label = QLabel()
        label.setText('是否确认删除？')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        self.situation_combo_cancel_dialog.show()

        yes_btn = QPushButton('删 除')
        yes_btn.setObjectName('shanchu')
        yes_btn.clicked.connect(partial(self.situation_combo_cancel, project_name, task_name))
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao')
        no_btn.clicked.connect(self.situation_combo_cancel_no)
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(yes_btn)
        flow_layout.addWidget(no_btn)

        layout = QVBoxLayout(self.situation_combo_cancel_dialog)
        layout.addWidget(label)
        layout.addWidget(flow_content)

    def situation_combo_cancel_no(self):
        self.situation_combo_cancel_dialog.close()

    def situation_combo_cancel(self, project_name, task_name):
        self.situation_combo_cancel_dialog.close()
        index = self.situation_table.currentRow()
        self.situation_table.removeRow(index)
        self.config_save(project_name, task_name)

    def action_combo_change(self, juno, project_name, task_name, action_current):
        if type(action_current) is int:
            current_index = []
            current_index.append(self.action_treeView.currentIndex().row())
            next_item = self.action_treeView.currentIndex().parent()
            while next_item.row() != -1:
                current_index.append(next_item.row())
                next_item = next_item.parent()
            current_index.reverse()

            datas = []
            item = self.action_treeView.currentIndex()
            for i in range(len(current_index)):
                datas.append(self.action_treeView.model().itemData(item)[0])
                item = item.parent()
            datas.reverse()

            action_current = ''
            for i in datas:
                action_current += i + '_'
            action_current = action_current.rstrip('_')
            print('行为: ' + action_current)

        table_list = []
        for i in range(self.situation_table.rowCount()):
            table_list.append(self.situation_table.item(i, 0).text())
        for i in range(self.action_table.rowCount()):
            table_list.append(self.action_table.item(i, 0).text())
        for i in range(self.result_table.rowCount()):
            table_list.append(self.result_table.item(i, 0).text())
        if action_current in table_list:
            self.error_dialog(juno, '重复添加!')
            return

        item = QTableWidgetItem(action_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.action_table.insertRow(self.action_table.rowCount())
        self.action_table.setItem(self.action_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删 除')
        cancel_button.setObjectName('shanchu')
        cancel_button.clicked.connect(partial(self.action_combo_cancel_pre, juno, project_name, task_name))
        self.action_table.setCellWidget(self.action_table.rowCount() - 1, 1, cancel_button)

        self.action_table.resizeRowsToContents()

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

        self.config_save(project_name, task_name)

    def action_combo_cancel_pre(self, juno, project_name, task_name):
        self.action_combo_cancel_dialog = QDialog(juno)
        self.action_combo_cancel_dialog.resize(self.screen_width*0.2, self.screen_height*0.2)
        self.action_combo_cancel_dialog.setWindowTitle('删除!')

        label = QLabel()
        label.setText('是否确认删除？')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        self.action_combo_cancel_dialog.show()

        yes_btn = QPushButton('删 除')
        yes_btn.setObjectName('shanchu')
        yes_btn.clicked.connect(partial(self.action_combo_cancel, project_name, task_name))
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao')
        no_btn.clicked.connect(self.action_combo_cancel_no)
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(yes_btn)
        flow_layout.addWidget(no_btn)

        layout = QVBoxLayout(self.action_combo_cancel_dialog)
        layout.addWidget(label)
        layout.addWidget(flow_content)

    def action_combo_cancel_no(self):
        self.action_combo_cancel_dialog.close()

    def action_combo_cancel(self, project_name, task_name):
        self.action_combo_cancel_dialog.close()
        index = self.action_table.currentRow()
        self.action_table.removeRow(index)
        self.cost_table.removeRow(index)
        self.config_save(project_name, task_name)

    def result_combo_change(self, juno, project_name, task_name, result_current):
        if type(result_current) is int:
            current_index = []
            current_index.append(self.result_treeView.currentIndex().row())
            next_item = self.result_treeView.currentIndex().parent()
            while next_item.row() != -1:
                current_index.append(next_item.row())
                next_item = next_item.parent()
            current_index.reverse()

            datas = []
            item = self.result_treeView.currentIndex()
            for i in range(len(current_index)):
                datas.append(self.result_treeView.model().itemData(item)[0])
                item = item.parent()
            datas.reverse()

            result_current = ''
            for i in datas:
                result_current += i + '_'
            result_current = result_current.rstrip('_')
            print('行为: ' + result_current)

        table_list = []
        for i in range(self.situation_table.rowCount()):
            table_list.append(self.situation_table.item(i, 0).text())
        for i in range(self.action_table.rowCount()):
            table_list.append(self.action_table.item(i, 0).text())
        for i in range(self.result_table.rowCount()):
            table_list.append(self.result_table.item(i, 0).text())
        if result_current in table_list:
            self.error_dialog(juno, '重复添加!')
            return

        item = QTableWidgetItem(result_current)
        item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.result_table.insertRow(self.result_table.rowCount())
        self.result_table.setItem(self.result_table.rowCount() - 1, 0, item)

        cancel_button = QPushButton('删 除')
        cancel_button.setObjectName('shanchu')
        cancel_button.clicked.connect(partial(self.result_combo_cancel_pre, juno, project_name, task_name))
        self.result_table.setCellWidget(self.result_table.rowCount() - 1, 1, cancel_button)

        self.result_table.resizeRowsToContents()

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

        self.config_save(project_name, task_name)

    def result_combo_cancel_pre(self, juno, project_name, task_name):
        self.result_combo_cancel_dialog = QDialog(juno)
        self.result_combo_cancel_dialog.resize(self.screen_width*0.2, self.screen_height*0.2)
        self.result_combo_cancel_dialog.setWindowTitle('删除!')

        label = QLabel()
        label.setText('是否确认删除？')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        self.result_combo_cancel_dialog.show()

        yes_btn = QPushButton('删 除')
        yes_btn.setObjectName('shanchu')
        yes_btn.clicked.connect(partial(self.result_combo_cancel, project_name, task_name))
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao')
        no_btn.clicked.connect(self.result_combo_cancel_no)
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(yes_btn)
        flow_layout.addWidget(no_btn)

        layout = QVBoxLayout(self.result_combo_cancel_dialog)
        layout.addWidget(label)
        layout.addWidget(flow_content)

    def result_combo_cancel_no(self):
        self.result_combo_cancel_dialog.close()

    def result_combo_cancel(self, project_name, task_name):
        self.result_combo_cancel_dialog.close()
        index = self.result_table.currentRow()
        name = self.result_table.item(index, 0).text()
        self.result_table.removeRow(index)
        self.risk_table.removeRow(index)

        if self.border_table.rowCount() > 0:
            if name == self.border_table.item(0, 0).text().split('--')[0]:
                self.border_table.clear()
                self.border_table.setRowCount(0)
                self.border_add_btn.setDisabled(True)

        if name in self.config_data['border']:
            self.config_data['border'].pop(name)

        self.config_save(project_name, task_name)

    def cost_table_change(self, project_name, task_name):
        self.config_save(project_name, task_name)

    def risk_table_change(self, project_name, task_name):
        self.config_save(project_name, task_name)

    def risk_to_border(self, juno, project_name, task_name):
        row_count = self.risk_table.currentRow()
        border_name = self.risk_table.item(row_count, 0).text()
        self.border_add_btn.setDisabled(False)

        self.border_table_graph_plt.clear()
        self.border_table_graph_plt.addLegend(brush=(255, 255, 255, 120), labelTextColor='555', pen={'color': "ccc", 'width': 0}) #边界图：图例说明-背景颜色（brush），字颜色（labelTextColor），边框（pen）
        border_data = self.value[border_name]
        border_data_1 = []
        for i in border_data:
            if i != '':
                border_data_1.append(float(i))
        border_data_1 = np.array(border_data_1)
        y, x = np.histogram(border_data_1, bins=30)
        self.border_table_graph_plt.plot(x, y, stepMode='center', fillLevel=0, fillOutLine=False, brush=(152, 152, 153, 255), name='人') #边界图：柱颜色（brush）

        self.border_table.clear()
        self.border_table.setRowCount(0)
        self.border_table.setHorizontalHeaderLabels([border_name + '--分布 %', border_name + '--边界 (值不要相同)', ''])

        if border_name in self.border_data:
            border_info = self.border_data[border_name]
            count = 0
            for i in border_info.keys():
                self.border_table.setRowCount(self.border_table.rowCount() + 1)
                item = QTableWidgetItem(i)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.border_table.setItem(count, 0, item)
                item = QTableWidgetItem(border_info[i])
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.border_table.setItem(count, 1, item)
                delete_btn = QPushButton('删 除')
                delete_btn.setObjectName('shanchu')
                delete_btn.clicked.connect(partial(self.border_table_delete_pre, juno, project_name, task_name))
                self.border_table.setCellWidget(count, 2, delete_btn)
                count += 1

    def border_table_delete_pre(self, juno, project_name, task_name):
        self.border_table_delete_dialog = QDialog(juno)
        self.border_table_delete_dialog.resize(self.screen_width*0.2, self.screen_height*0.2)
        self.border_table_delete_dialog.setWindowTitle('删除!')

        label = QLabel()
        label.setText('是否确认删除？')
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        self.border_table_delete_dialog.show()

        yes_btn = QPushButton('删 除')
        yes_btn.setObjectName('shanchu')
        yes_btn.clicked.connect(partial(self.border_table_delete_row, project_name, task_name))
        no_btn = QPushButton('取 消')
        no_btn.setObjectName('quxiao')
        no_btn.clicked.connect(self.border_table_delete_no)
        flow_content = QWidget()
        flow_layout = QHBoxLayout(flow_content)
        flow_layout.addWidget(yes_btn)
        flow_layout.addWidget(no_btn)

        layout = QVBoxLayout(self.border_table_delete_dialog)
        layout.addWidget(label)
        layout.addWidget(flow_content)

    def border_table_delete_no(self):
        self.border_table_delete_dialog.close()

    def border_table_delete_row(self, project_name, task_name):
        self.border_table_delete_dialog.close()
        delete_row = self.border_table.currentRow()
        self.border_table.removeRow(delete_row)
        self.borber_table_cell_change(project_name, task_name)

    def borber_table_cell_change(self, project_name, task_name):
        current_border_name = self.border_table.horizontalHeaderItem(0).text().rstrip('--' + self.border_table.horizontalHeaderItem(0).text().split('--')[-1])
        row_count = self.border_table.rowCount()

        if row_count == 0:
            try:
                self.border_data.pop(current_border_name)
                self.config_save(project_name, task_name)
            except:
                pass
            return

        c = {}
        for i in range(row_count):
            if self.border_table.item(i, 0) is None:
                border_name = ''
            elif self.border_table.item(i, 0).text() == '':
                border_name = ''
            else:
                border_name = self.border_table.item(i, 0).text()

            if self.border_table.item(i, 1) is None:
                border_value = ''
            elif self.border_table.item(i, 1).text() == '':
                border_value = ''
            else:
                border_value = self.border_table.item(i, 1).text()

            c.update({border_name: border_value})

        if c == {}:
            return
        b = {current_border_name: c}
        self.border_data.update(b)

        self.config_save(project_name, task_name)

    def border_table_add_row(self, juno, project_name, task_name):
        self.border_table.setRowCount(self.border_table.rowCount() + 1)

        item = QTableWidgetItem('')
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(self.border_table.rowCount() - 1, 0, item)

        item = QTableWidgetItem('')
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.border_table.setItem(self.border_table.rowCount() - 1, 1, item)

        delete_btn = QPushButton('删 除')
        delete_btn.setObjectName('shanchu')
        delete_btn.clicked.connect(partial(self.border_table_delete_pre, juno, project_name, task_name))
        self.border_table.setCellWidget(self.border_table.rowCount() - 1, 2, delete_btn)

    def config_back(self, project_name, juno):
        self.menubar.setVisible(False)
        juno.move(self.screen_width*0.35, self.screen_height*0.3)
        self.task_window(project_name, juno)

    def config_save(self, project_name, task_name):
        situation_temp = []
        for i in range(self.situation_table.rowCount()):
            situation_temp.append(self.situation_table.item(i, 0).text())
        self.config_data.update({'situation': situation_temp})

        action_temp = []
        for i in range(self.action_table.rowCount()):
            action_temp.append(self.action_table.item(i, 0).text())
        self.config_data.update({'action': action_temp})

        result_temp = []
        for i in range(self.result_table.rowCount()):
            result_temp.append(self.result_table.item(i, 0).text())
        self.config_data.update({'result': result_temp})

        cost_temp = {}
        for i in range(self.cost_table.rowCount()):
            cost_name = self.cost_table.item(i, 0).text()
            if self.cost_table.item(i, 1) is None:
                cost_value = ''
            else:
                cost_value = self.cost_table.item(i, 1).text()
            cost_temp.update({cost_name: cost_value})
        self.config_data.update({'cost': cost_temp})

        risk_temp = {}
        for i in range(self.risk_table.rowCount()):
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
        safety = self.safety_lineEdit.text()
        self.config_data.update({'begin': Begin_datestamp})
        self.config_data.update({'end': End_datestamp})
        self.config_data.update({'safety': safety})

        data = self.config_data.copy()
        cipher = self.encrypt(data)

        with open('./JunoProject/' + project_name + '/Task/' + task_name + '.txt', 'w') as f:
            f.write(cipher.decode())

    def config_check(self, value, config_data):
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
            data.append(i - len(riqi_temp))
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

        if not config_data['situation']:
            return '情况变量为空!'
        if not config_data['action']:
            return '行为变量为空!'
        if not config_data['result']:
            return '结果变量为空!'

        if len(config_data['risk'].keys()) != len(config_data['border'].keys()):
            return '结果变量未定义边界!'
        try:
            border = []
            for i in config_data['border']:
                data = {}
                for j in config_data['border'][i]:
                    if j == '':
                        return '结果变量边界定义缺失!'
                    data.update({j: float(config_data['border'][i][j])})
                border.append(data)
        except ValueError:
            return '结果变量边界定义缺失!'

        typedefs = []
        for i in range(len(config_data['situation'])):
            typedefs.append(-1)

        count = 0
        for i in config_data['cost']:
            if not config_data['cost'][i]:
                return '成本占比数据缺失!'
            cost_data = float(config_data['cost'][i])
            if 0.0 > cost_data or cost_data > 100.1:
                return '成本占比数据超限! ' + str(cost_data)
            count += cost_data
            typedefs.append(cost_data / 100.0)
        if count > 100.1:
            return '成本占比数据超限! ' + str(count)

        count = 0
        for i in config_data['risk']:
            if not config_data['risk'][i]:
                return '重要性占比数据缺失!'
            risk_data = float(config_data['risk'][i])
            if 0.0 > risk_data or risk_data > 100.1:
                return '重要性占比数据超限! ' + str(risk_data)
            count += risk_data
            typedefs.append(risk_data / 10.0 + 2)
        if count > 100.1:
            return '重要性占比数据超限! ' + str(count)

        Begin_datestamp = config_data['begin']
        End_datestamp = config_data['end']
        startIndex = float(Begin_datestamp) - float(max(riqi)) - 1
        endIndex = float(End_datestamp) - float(max(riqi)) - 1

        return values, border, typedefs, startIndex, endIndex

    def config_calc(self, juno, project_name, task_name):
        self.wait_dialog = QDialog(juno)
        self.wait_dialog.resize(self.screen_width * 0.3, self.screen_height * 0.3)
        self.wait_dialog.move(self.screen_width * 0.35, self.screen_height * 0.3)
        self.wait_dialog.setWindowTitle('请稍等...')
        self.wait_dialog.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.wait_label = QLabel(self.wait_dialog)
        self.wait_label.setFixedSize(self.screen_width * 0.3, self.screen_height * 0.3)
        self.wait_label.setAlignment(QtCore.Qt.AlignCenter)
        self.wait_label.show()
        self.wait_dialog.show()

        self.config_save(project_name, task_name)

        value = self.value.copy()
        config_data = self.config_data.copy()
        args = self.config_check(value, config_data)
        if type(args) == str:
            self.error_dialog(juno, args)
            return
        else:
            values = args[0]
            border = args[1]
            typedefs = args[2]
            startIndex = args[3]
            endIndex = args[4]

        self.calc_juno = juno
        self.calc_project_name = project_name
        self.calc_task_name = task_name

        self.waitthread = WaitThread()
        self.waitthread.signal.connect(self.wait_callback)
        self.waitthread.start()

        calcthread = CalcThread()
        calcthread.signal.connect(self.result_callback)
        calcthread.values = values
        calcthread.x = self.config_data['situation']
        calcthread.y = self.config_data['action']
        calcthread.z = self.config_data['result']
        calcthread.border = border
        calcthread.typedefs = typedefs
        calcthread.safety = float(self.config_data['safety'])
        calcthread.startIndex = int(startIndex)
        calcthread.endIndex = int(endIndex)
        calcthread.start()

    def result_callback(self, result):
        self.result_window(self.calc_juno, result, self.calc_project_name, self.calc_task_name)
        self.waitthread.stop()
        self.wait_dialog.close()

    def wait_callback(self, str):
        self.wait_label.setText(str)

    def error_dialog(self, juno, error_text):
        dialog = QDialog(juno)
        dialog.resize(self.screen_width*0.2, self.screen_height*0.2)
        dialog.setWindowTitle('错误!')
        dialog.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        label = QLabel(dialog)
        label.setText(error_text)
        label.setFixedSize(self.screen_width*0.2, self.screen_height*0.2)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.show()
        dialog.show()

    def result_window(self, juno, result, project_name, task_name):
        juno.setWindowTitle(Version + ' - 结果展示')

        self.table1_label = QLabel('项目: ' + project_name + '\n\n命题: ' + task_name + '\n\n各种边界的AI策略结果\n')
        self.table1 = QTableWidget()
        self.table1.verticalHeader().setVisible(False)

        result_1_content = QWidget()
        result_1_content.setFixedHeight(0.3*self.screen_height)
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
        self.graph1.setBackground('#182037')  # 柱状图：背景颜色
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
        self.graph2.setBackground('#182037')  # 散点图：背景颜色
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

        self.table7_add = QPushButton('添 加')
        self.table7_add.setObjectName('tianjia')
        self.table7_add.clicked.connect(partial(self.table7_add_btn, result))

        table7_content = QWidget()
        table7_layout = QVBoxLayout(table7_content)
        table7_layout.addWidget(self.table7)
        table7_layout.addWidget(self.table7_add)

        self.table8 = QTableWidget()
        self.table8.horizontalHeader().setVisible(False)
        self.table8.cellChanged.connect(partial(self.table8_change, result, project_name, task_name))

        self.table8_add = QPushButton('添 加')
        self.table8_add.setObjectName('tianjia')
        self.table8_add.clicked.connect(partial(self.table8_add_btn, result))

        table8_content = QWidget()
        table8_layout = QVBoxLayout(table8_content)
        table8_layout.addWidget(self.table8)
        table8_layout.addWidget(self.table8_add)

        self.table9 = QTableWidget()
        self.table9.horizontalHeader().setVisible(False)

        self.table9_export = QPushButton('导出预测结果')
        self.table9_export.setObjectName('tianjia')
        self.table9_export.clicked.connect(partial(self.forecast_export, juno))

        table9_content = QWidget()
        table9_layout = QVBoxLayout(table9_content)
        table9_layout.addWidget(self.table9)
        table9_layout.addWidget(self.table9_export)

        self.tab3 = QWidget()
        result_4_layout = QHBoxLayout(self.tab3)
        result_4_layout.addWidget(table7_content)
        result_4_layout.addWidget(table8_content)
        result_4_layout.addWidget(table9_content)

        tabs = QTabWidget()
        if self.role == '0' or self.role == '1':
            tabs.addTab(self.tab1, '策略概览')
            tabs.addTab(self.tab2, '策略分析依据')
        tabs.addTab(self.tab3, '策略应用')

        back_btn = QPushButton('返 回')
        back_btn.setObjectName('fanhui1')
        back_btn.setFixedWidth(self.screen_width * 0.06)
        back_btn.clicked.connect(partial(self.result_back_to_config, project_name, task_name, juno))

        export_btn = QPushButton('导 出')
        export_btn.setObjectName('daochu')
        export_btn.setFixedWidth(self.screen_width*0.06)
        export_btn.clicked.connect(partial(self.export_to_excel, project_name, task_name, result))

        btn_content = QWidget()
        btn_layout = QHBoxLayout(btn_content)
        btn_layout.addWidget(back_btn)
        if self.role == '0' or self.role == '1':
            btn_layout.addWidget(export_btn)

        result_content = QWidget()
        result_layout = QVBoxLayout(result_content)
        if self.role == '0' or self.role == '1':
            result_layout.addWidget(result_1_content)
        result_layout.addWidget(tabs)
        result_layout.addWidget(btn_content)
        juno.setCentralWidget(result_content)

        if self.role == '0' or self.role == '1':
            self.load_table1(result)
            self.load_table2(result)
            self.load_table3(result)
            self.load_table4()
            self.load_table5(result)
            self.load_table6(result)
            self.load_graph1_combo(result)
            self.load_graph2_combo(result)
            self.graph1_combo_change(result)
            self.graph2_combo_change(result)

        self.load_table7(result)
        self.load_table8()
        self.load_table9()

        self.table7_add_lastday(result, project_name)

    def result_back_to_config(self, project_name, task_name, juno):
        if self.role == '0' or self.role == '1':
            self.config_window(project_name, juno, task_name)
        else:
            juno.move(self.screen_width * 0.35, self.screen_height * 0.3)
            self.Main_Window(juno)

    def load_table1(self, result):
        table_title = []
        count = 0
        for i in result.risks.keys():
            count += 1
            for j in result.risks[i]['threshold']:
                table_title.append('Z' + str(count) + ' - ' + i + ' - ' + j)
        count = 0
        for i in result.consumptions.keys():
            count += 1
            table_title.append('Y' + str(count) + ' - ' + i + ' 节约率(%)')
        count = 0
        for i in result.risks.keys():
            count += 1
            table_title.append('Z' + str(count) + ' - ' + i + ' 达标优势(%)')

        self.table1.setColumnCount(len(table_title))
        self.table1.setRowCount(1)

        index = 0
        for i in table_title:
            if '节约率' in i:
                brush = QtGui.QBrush(QtGui.QColor(76, 140, 255, 255))
            elif '达标优势' in i:
                brush = QtGui.QBrush(QtGui.QColor(255, 140, 76, 255))
            else:
                brush = QtGui.QBrush(QtGui.QColor(0, 0, 0, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            item = QTableWidgetItem(i)
            item.setForeground(brush)
            self.table1.setHorizontalHeaderItem(index, item)
            index += 1

        value = []
        for i in result.risks.keys():
            for j in result.risks[i]['threshold']:
                value.append(round(result.risks[i]['threshold'][j], 2))
        for i in result.consumptions.keys():
            value.append(round(result.consumptions[i]['saving_rate'], 2))
        for i in result.risks.keys():
            value.append(round(result.risks[i]['ai_advantage'], 2))
        for i in range(len(value)):
            item = QTableWidgetItem(str(value[i]))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table1.setItem(0, i, item)

        self.table1.resizeColumnsToContents()

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
        self.plt1.addLegend(brush=(255, 255, 255, 120), labelTextColor='333', pen={'color': "ccc", 'width': 0}) #柱状图-图例说明： 背景颜色 = brush (尾值代表透明度)，字颜色 = labelTextColor， 边框 = pen

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

        self.plt1.plot(x_hm, y_hm, stepMode='center', fillLevel=0, fillOutLine=False, brush=(152, 152, 153, 150), name='人 - ' + self.graph1_combo.currentText())
        self.plt1.plot(x_ai, y_ai, stepMode='center', fillLevel=0, fillOutLine=False, brush=(65, 88, 158, 150), name='AI - ' + self.graph1_combo.currentText())

    def graph2_combo_change(self, result):

        x = self.graph2_combo_x.currentText()
        y = self.graph2_combo_y.currentText()

        self.graph2.clear()
        self.graph2.setTitle(x + " 和 " + y + " 之间的关系")
        self.graph2.setLabel('left', y)
        self.graph2.setLabel('bottom', x)
        self.graph2.addLegend(brush=(255, 255, 255, 120), labelTextColor='333', pen={'color': "ccc", 'width': 1}) #散点图-图例说明： 背景颜色 = brush (尾值代表透明度)，字颜色 = labelTextColor， 边框 = pen

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

        scatter_hm = pyqtgraph.ScatterPlotItem(size=5, brush=pyqtgraph.mkBrush(152, 152, 153, 150)) #散点图-散点颜色
        scatter_hm.setData(pos=zip(axis_x_hm, axis_y_hm), alpha=1, name='人')
        self.graph2.addItem(scatter_hm)

        scatter_ai = pyqtgraph.ScatterPlotItem(size=7, brush=pyqtgraph.mkBrush(68, 88, 158, 250))
        scatter_ai.setData(pos=zip(axis_x_ai, axis_y_ai), alpha=1, name='AI') #散点图-散点颜色
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
        x_title = ['当天情况']
        count_x = 1
        for i in self.config_data['situation']:
            x_title.append('X' + str(count_x) + ' - ' + i)
            count_x += 1
        x_title.append('')
        self.table7.setVerticalHeaderLabels(x_title)
        self.table7.resizeRowsToContents()

    def table7_add_lastday(self, result, project_name):
        self.today = str(date.fromordinal(int(float(self.value['日期'][-1]) + date(1900, 1, 1).toordinal() - 2)))
        self.tomorrow = str(date.fromordinal(int(float(self.value['日期'][-1]) + date(1900, 1, 1).toordinal() - 1)))

        with open('./JunoProject/' + project_name + '/value.json', 'r') as f:
            data = json.load(f)
        with open('./JunoProject/' + project_name + '/trx.txt', 'r') as f:
            trX = np.array(eval(f.read()))

        today_data = []
        x_title = []
        for i in range(self.table7.rowCount()):
            x_title.append(self.table7.verticalHeaderItem(i).text())
        for i in x_title:
            if ' - ' in i:
                name = i.split(' - ')[1]
                ind = algo.var2ind(name, data)
                val = trX[ind, -1, 1]
                today_data.append(val)
        today_data = [today_data]
        today_x = ['']
        for i in today_data[0]:
            today_x.append(round(i, 2))

        self.table7_add_btn(result)

        for i in range(1, self.table7.rowCount() - 1):
            item = QTableWidgetItem(str(today_x[i]))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table7.setItem(i, self.table7.columnCount() - 1, item)

        self.table7.item(0, 0).setText(self.today)
        self.table8.item(0, 0).setText(self.today + '  人')
        self.table8.item(0, 1).setText(self.today + '  AI')
        self.table9.item(0, 0).setText(self.tomorrow + '  人')
        self.table9.item(0, 1).setText(self.tomorrow + '  AI')

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

        table7_delete_column_btn = QPushButton('删 除')
        table7_delete_column_btn.setObjectName('shanchu')
        table7_delete_column_btn.clicked.connect(partial(self.table7_delete_column_btn, result))
        self.table7.setCellWidget(self.table7.rowCount() - 1, self.table7.columnCount() - 1, table7_delete_column_btn)
        self.table7.resizeRowsToContents()
        self.table7.resizeColumnsToContents()

    def table7_delete_column_btn(self, result):
        delete_index = self.table7.currentColumn()
        self.table7.removeColumn(delete_index)
        self.table7_change(result)

        column_count = self.table7.columnCount()
        for i in range(column_count):
            if '自定义输入' not in self.table7.item(0, i).text():
                continue
            item = QTableWidgetItem('自定义输入 - ' + str(i + 1))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table7.setItem(0, i, item)

    def table7_change(self, result):
        row_count = self.table7.rowCount()
        column_count = self.table7.columnCount()
        if row_count == 0 or column_count == 0:
            self.table8.setColumnCount(0)
            self.table9.setColumnCount(0)
            self.table8_add.setDisabled(True)
            self.table9_export.setDisabled(True)
            return
        for i in range(1, row_count - 1):
            for j in range(column_count):
                try:
                    float(self.table7.item(i, j).text())
                except:
                    self.table8_add.setDisabled(True)
                    self.table9_export.setDisabled(True)
                    return
        self.table8_add.setDisabled(False)
        self.table8_add.setText('添加')
        self.table8.setColumnCount(0)
        self.forecast_action(result)

        if '自定义输入' not in self.table7.item(0, 0).text():
            self.table8.item(0, 0).setText(self.today + '  人')
            self.table8.item(0, 1).setText(self.today + '  AI')

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
        count_ai = 1
        for i in range(len(y_hm)):
            y = [str(count_hm) + ' - 人']
            for j in y_hm[i]:
                y.append(round(j, 2))
            count_hm += 1
            count_y = 0
            for j in y:
                item = QTableWidgetItem(str(j))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table8.setItem(count_y, count_table8, item)
                count_y += 1
            count_table8 += 1

            y = [str(count_ai) + ' - AI']
            for j in y_ai[i]:
                y.append(round(j, 2))
            count_ai += 1
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
        y_title = ['当天行为']
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
            table8_delete_column_btn = QPushButton('删 除')
            table8_delete_column_btn.setObjectName('shanchu')
            table8_delete_column_btn.clicked.connect(partial(self.table8_delete_column_btn, result))
            self.table8.setCellWidget(self.table8.rowCount() - 1, self.table8.columnCount() - 1, table8_delete_column_btn)
            self.table8.resizeRowsToContents()
        self.table8_add.setText('更改')

    def table8_delete_column_btn(self, result):
        delete_columns_count = self.table7.columnCount()
        for i in range(delete_columns_count):
            self.table8.removeColumn(delete_columns_count*2)
        self.table8_add.setText('添加')
        self.table8_change(result)

    def table8_change(self, result, project_name=0, task_name=0):
        row_count = self.table8.rowCount()
        column_count = self.table8.columnCount()

        for j in range(column_count):
            for i in range(1, row_count - 1):
                try:
                    float(self.table8.item(i, j).text())
                except:
                    self.table9_export.setDisabled(True)
                    return
        self.forecast_result(result, project_name, task_name)

        if '自定义输入' not in self.table7.item(0, 0).text():
            self.table9.item(0, 0).setText(self.tomorrow + '  人')
            self.table9.item(0, 1).setText(self.tomorrow + '  AI')

    def forecast_result(self, result, project_name, task_name):
        rowCount_x = self.table7.rowCount()
        columnCount_x = self.table7.columnCount()
        x = []
        for i in range(columnCount_x):
            x1 = []
            for j in range(1, rowCount_x - 1):
                try:
                    x1.append(float(self.table7.item(j, i).text()))
                except ValueError:
                    print('情况变量不完全！')
                    return
            x.append(x1)

        rowCount_y = self.table8.rowCount()
        columnCount_y = self.table8.columnCount()
        y_recive = []
        for i in range(self.table7.columnCount()*2):
            y1 = []
            for j in range(1, rowCount_y - 1):
                y1.append(float(self.table8.item(j, i).text()))
            y_recive.append(y1)
        y_send = []
        for i in range(self.table7.columnCount() * 2, columnCount_y):
            y1 = []
            for j in range(1, rowCount_y - 1):
                y1.append(float(self.table8.item(j, i).text()))
            y_send.append(y1)

        T_input = []
        for i in range(len(x)):
            T_input.append(x[i] + y_recive[i * 2])
            T_input.append(x[i] + y_recive[i * 2 + 1])
        for i in range(len(x)):
            if y_send:
                T_input.append(x[i] + y_send[i])

        z = result.T(T_input).tolist()

        self.table9.setColumnCount(len(z))
        count_table9 = 0

        count_hm = 1
        count_ai = 1
        for i in range(self.table7.columnCount()):
            z1 = [str(count_hm) + ' - 人']
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

            z2 = [str(count_ai) + ' - AI']
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
        self.table9_change()

        # log 记录 table 7 8 9 三张表的数据
        if project_name != 0 and task_name != 0:
            with open('./JunoProject/' + project_name + '/Log/' + task_name + '.log', 'a+') as f:
                table789_log = []

                table7_row_count = self.table7.rowCount()
                table7_column_count = self.table7.columnCount()
                table7_log_columns = []
                for j in range(table7_column_count):
                    table7_log_column = []
                    for i in range(1, table7_row_count - 1):
                        table7_log_column.append(self.table7.item(i, j).text())
                    table7_log_columns.append(table7_log_column)
                table789_log.append(table7_log_columns)

                table8_row_count = self.table8.rowCount()
                table8_column_count = self.table8.columnCount()
                table8_log_columns = []
                for j in range(len(table7_log_columns) * 2, table8_column_count):
                    table8_log_column = []
                    for i in range(1, table8_row_count - 1):
                        table8_log_column.append(self.table8.item(i, j).text())
                    table8_log_columns.append(table8_log_column)
                table789_log.append(table8_log_columns)

                f.write(str(datetime.datetime.now()) + '\t' + str(table789_log) + '\n')

    def load_table9(self):
        self.table9.setRowCount(len(self.config_data['result']) + 1)
        z_title = ['一天后结果']
        count_z = 1
        for i in self.config_data['result']:
            z_title.append('Z' + str(count_z) + ' - ' + i)
            count_z += 1
        self.table9.setVerticalHeaderLabels(z_title)
        self.table9.resizeColumnsToContents()
        self.table9_change()

    def table9_change(self):
        if self.table9.columnCount() == 0:
            self.table9_export.setDisabled(True)
        else:
            self.table9_export.setDisabled(False)

    def forecast_export(self, juno):
        option = QFileDialog.Option()
        option |= QFileDialog.DontUseNativeDialog
        path, _ = QFileDialog.getSaveFileName(
            juno, '导出这些表格', desktop_path + str(datetime.date.today()) + ".xlsx",
            'xlsx文件 (*.xlsx);;xls文件 (*.xls)',
            options=option
        )

        xlsx = xlsxwriter.Workbook(path)
        text_format = xlsx.add_format({'text_wrap': 'True'})
        table = xlsx.add_worksheet(time.asctime().split(' ')[3].replace(':', '时', 1).replace(':', '分', 2) + '秒')

        y = ['指标名称']
        for i in range(1, self.table7.rowCount() - 1):
            y.append(self.table7.verticalHeaderItem(i).text())
        for i in range(1, self.table8.rowCount() - 1):
            y.append(self.table8.verticalHeaderItem(i).text())
        for i in range(1, self.table9.rowCount()):
            y.append(self.table9.verticalHeaderItem(i).text())
        for i in range(len(y)):
            table.write(i, 0, y[i], text_format)

        x = []
        for i in range(self.table8.columnCount()):
            x.append(self.table8.item(0, i).text())
        for i in range(len(x)):
            table.write(0, i + 1, x[i], text_format)

        table7_data = []
        for i in range(self.table7.columnCount()):
            column_data = []
            for j in range(1, self.table7.rowCount() - 1):
                column_data.append(self.table7.item(j, i).text())
            table7_data.append(column_data)
            column_data = []
            for j in range(1, self.table7.rowCount() - 1):
                column_data.append('')
            table7_data.append(column_data)
        for i in range(self.table8.columnCount() - self.table7.columnCount() * 2):
            column_data = []
            for j in range(1, self.table7.rowCount() - 1):
                column_data.append('')
            table7_data.append(column_data)

        table8_data = []
        for i in range(self.table8.columnCount()):
            column_data = []
            for j in range(1, self.table8.rowCount() - 1):
                column_data.append(self.table8.item(j, i).text())
            table8_data.append(column_data)

        table9_data = []
        for i in range(self.table9.columnCount()):
            column_data = []
            for j in range(1, self.table9.rowCount()):
                column_data.append(self.table9.item(j, i).text())
            table9_data.append(column_data)

        table7_data = np.array(table7_data)
        table8_data = np.array(table8_data)
        table9_data = np.array(table9_data)
        data = np.hstack((table7_data, table8_data, table9_data)).tolist()

        for i in range(len(data)):
            for j in range(len(data[i])):
                item = '' if data[i][j] == '' else float(data[i][j])
                table.write(j + 1, i + 1, item, text_format)

        try:
            xlsx.close()
        except:
            pass

    def export_to_excel(self, project_name, task_name, result):
        # exportthread = ExportThread()
        # exportthread.project_name = project_name
        # exportthread.task_name = task_name
        # exportthread.result = result

        str = "./JunoProject/" + project_name + "/Output/"
        str_image_1 = "./JunoProject/" + project_name + "/Output/" + task_name + "/人机策略分布"
        str_image_2 = "./JunoProject/" + project_name + "/Output/" + task_name + "/两色散点图"
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
            a = 'JunoProject/' + project_name + '/Output/' + task_name + '/人机策略分布/' + i.replace(' ', '').replace('/', '') + '.png'
            self.plt1.enableAutoRange()
            pyqtgraph.exporters.ImageExporter(self.plt1).export(fileName=a)
        for i in self.graph2_combo_value:
            for j in self.graph2_combo_value:
                if i != j:
                    self.graph2_combo_x.setCurrentText(i)
                    self.graph2_combo_y.setCurrentText(j)
                    self.graph2_combo_change(result)
                    a = 'JunoProject/' + project_name + '/Output/' + task_name + '/两色散点图/' + i.replace(' ', '').replace('/', '').split('(')[0] + '&' + j.replace(' ', '').replace('/', '').split('(')[0] + '.png'
                    self.graph2.getPlotItem().enableAutoRange()
                    pyqtgraph.exporters.ImageExporter(self.graph2.plotItem).export(fileName=a)
        self.graph1_combo.setCurrentText(current_graph1_combo)
        self.graph1_combo_change(result)
        self.graph2_combo_x.setCurrentText(current_graph2_combo_x)
        self.graph2_combo_y.setCurrentText(current_graph2_combo_y)
        self.graph2_combo_change(result)

        xlsx = xlsxwriter.Workbook(str + task_name + '/table.xlsx')
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

    def details(self, juno, project_name):
        dialog = QDialog(juno)
        dialog.resize(self.screen_width * 0.5, self.screen_height * 0.5)
        dialog.setWindowTitle('数据查看')

        combo = QComboBox()
        combo.setObjectName('combobox')
        combo.setModel(self.treeModel)
        combo.setView(self.details_treeView)
        combo.activated.connect(self.details_combo_click)
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(['日期', '数据'])
        self.details_table.verticalHeader().setVisible(False)
        self.details_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        details_left_content = QWidget()
        details_left_layout = QVBoxLayout(details_left_content)
        details_left_layout.addWidget(combo)
        details_left_layout.addWidget(self.details_table)

        date_axis = pyqtgraph.DateAxisItem(orientation='bottom')
        self.plot_graph = pyqtgraph.PlotWidget(axisItems={'bottom': date_axis})
        self.plot_graph.setBackground('#182037')
        histo_graph = pyqtgraph.GraphicsLayoutWidget()
        histo_graph.setBackground('#182037')
        self.plt1 = histo_graph.addPlot()
        details_right_content = QWidget()
        details_right_layout = QVBoxLayout(details_right_content)
        details_right_layout.addWidget(self.plot_graph)
        details_right_layout.addWidget(histo_graph)

        details_content = QWidget(dialog)
        details_content.setFixedSize(self.screen_width * 0.5, self.screen_height * 0.5)
        details_layout = QHBoxLayout(details_content)
        details_layout.addWidget(details_left_content)
        details_layout.addWidget(details_right_content)

        dialog.show()

    def details_combo_click(self):
        current_index = []
        current_index.append(self.details_treeView.currentIndex().row())
        next_item = self.details_treeView.currentIndex().parent()
        while next_item.row() != -1:
            current_index.append(next_item.row())
            next_item = next_item.parent()
        current_index.reverse()

        datas = []
        item = self.details_treeView.currentIndex()
        for i in range(len(current_index)):
            datas.append(self.details_treeView.model().itemData(item)[0])
            item = item.parent()
        datas.reverse()

        details_current = ''
        for i in datas:
            details_current += i + '_'
        details_current = details_current.rstrip('_')

        row_count = len(self.value['日期'])
        self.details_table.setRowCount(row_count)
        for i in range(row_count):
            item = QTableWidgetItem(str(date.fromordinal(int(float(self.value['日期'][i]) + date(1900, 1, 1).toordinal() - 2))))
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.details_table.setItem(i, 0, item)

            item = QTableWidgetItem(self.value[details_current][i])
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.details_table.setItem(i, 1, item)

        x = []
        y = []
        for i in range(row_count):
            if self.value[details_current][i] == '':
                continue
            x.append(int(self.date_convert(float(self.value['日期'][i]))))
            y.append(float(self.value[details_current][i]))
        scatter = pyqtgraph.ScatterPlotItem(size=5, brush=pyqtgraph.mkBrush(0, 0, 0, 80))
        scatter.setData(pos=zip(x, y), alpha=0.5, name='历史数据')
        self.plot_graph.clear()
        self.plot_graph.setLabel('left', details_current)
        self.plot_graph.setLabel('bottom', str(date.fromordinal(int(float(min(self.value['日期'])) + date(1900, 1, 1).toordinal() - 2))) + " 到 " + str(date.fromordinal(int(float(max(self.value['日期'])) + date(1900, 1, 1).toordinal() - 2))))
        self.plot_graph.addItem(scatter)

        self.plt1.clear()
        self.plt1.addLegend(brush=(255, 255, 255, 120), labelTextColor='555', pen={'color': "ccc", 'width': 1})
        y = np.array(y)
        y, x = np.histogram(y)
        y = y / np.sum(y)
        self.plt1.plot(x, y, stepMode='center', fillLevel=0, fillOutLine=False, brush=(161, 164, 167, 150), name=details_current)

    def date_convert(self, excel_date):
        excel_date = date.fromordinal(int(float(excel_date)) + date(1900, 1, 1).toordinal() - 2)
        time_struct = time.strptime(str(excel_date), "%Y-%m-%d")
        timestamp = time.mktime(time_struct)
        return timestamp

    def md5(self, s, salt='huang5201314'):
        return hashlib.md5((str(s) + salt).encode()).hexdigest()

    def encrypt(self, data):
        data = str.encode(str(data), encoding='utf-8')
        diff = 16 - len(data) % 16
        data += diff * chr(ord(' ')).encode()
        aes = AES.new(str.encode(encrypt_decrypt_key), AES.MODE_ECB)
        cipher = base64.encodebytes(aes.encrypt(data))
        return cipher

    def decrypt(self, data):
        aes = AES.new(str.encode(encrypt_decrypt_key), AES.MODE_ECB)
        text = aes.decrypt(base64.decodebytes(data)).decode().rstrip()
        return text


# class ExportThread(QtCore.QThread):
#     def __init__(self, parent=None):
#         super(ExportThread, self).__init__(parent)
#
#     def run(self):
#         pass
#
#     def __del__(self):
#         self.wait(1)


class CalcThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super(CalcThread, self).__init__(parent)

    def run(self):
        result = algo.analysis(self.values, self.x, self.y, self.z, self.border, self.typedefs, safety=self.safety, startIndex=self.startIndex, endIndex=self.endIndex, verbose=True)
        self.signal.emit(result)

    def __del__(self):
        self.wait(1)


class WaitThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(WaitThread, self).__init__(parent)
        self.flag = 1

    def run(self):
        while self.flag:
            self.signal.emit('请稍等')
            time.sleep(0.2)
            self.signal.emit('请稍等.')
            time.sleep(0.2)
            self.signal.emit('请稍等..')
            time.sleep(0.2)
            self.signal.emit('请稍等...')
            time.sleep(0.2)

    def stop(self):
        self.flag = 0

    def __del__(self):
        self.wait(1)


if __name__ == '__main__':
    print(algo.about())
    # QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    qssFile = 'texstudio.qss'

    try:
        app.setStyleSheet(open('QSS\\' + qssFile).read())
    except UnicodeDecodeError:
        app.setStyleSheet(open('QSS\\' + qssFile, encoding='utf-8').read())
    current_screen_width = app.desktop().screenGeometry().width()
    current_screen_height = app.desktop().screenGeometry().height()
    MainWindow = QMainWindow()
    window = JunoUI(MainWindow, current_screen_width, current_screen_height)
    MainWindow.show()
    sys.exit(app.exec_())
