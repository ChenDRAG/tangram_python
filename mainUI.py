# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore, QtGui
import Window
import os
import sys
from PyQt5.QtWidgets import QMessageBox
from solver import solve_result, process_generator
import solver_nine_pieces
import numpy as np
import cv2

class MainWindow(QMainWindow, Window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.data_num = 21
        self.game_mode = 0
        self.results_list = None
        self.select.setMinimum(0)
        self.select.setMaximum(self.data_num)
        self.select.setValue(0)

        self.select.valueChanged.connect(self.change)
        self.solve.clicked.connect(self.solve_tangram)
        self.result.clicked.connect(self.tangram_result)
        self.plus.clicked.connect(self.plus_action)
        self.minus.clicked.connect(self.minus_action)
        self.filee.clicked.connect(self.fi)
        self.mode.currentIndexChanged.connect(self.change_mode)
        
#         # set init image
        img = QtGui.QPixmap("Tangram_init.jpeg")
        img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        self.display.setPixmap(img)

    @staticmethod
    def img2pixmap(image):
        Y, X = image.shape[:2]
        _bgra = np.zeros((Y, X, 4), dtype=np.uint8, order='C')
        _bgra[..., 0] = image[..., 2]
        _bgra[..., 1] = image[..., 1]
        _bgra[..., 2] = image[..., 0]
        qimage = QtGui.QImage(_bgra.data, X, Y, QtGui.QImage.Format_RGB32)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap

    def change_mode(self):
        self.game_mode = self.mode.currentIndex()
        if self.game_mode == 1:
            filenamet  = '9tangram.png'
            try:
                img = QtGui.QPixmap(filenamet)
                img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
                if img is None:
                    QMessageBox.information(self, 'info', 'Fail to load the image.', QMessageBox.Ok)
                    return
                self.display.setPixmap(img)
                self.filename = filenamet
                self.results_list = None
                self.i = 0
            except AttributeError:
                QMessageBox.information(self, 'info', 'Fail to load the image.', QMessageBox.Ok)
            _translate = QtCore.QCoreApplication.translate
            self.info.setText(_translate("MainWindow", "9 blocks, enjoy!"))
        else:
            self.change()
            self.info.setText(_translate("MainWindow", "Welcome!"))

            

    def change(self):
        if self.game_mode == 1:
            return
        filenamet  = 'tangram' + str(self.select.value()) + '.png'
        try:
            img = QtGui.QPixmap(filenamet)
            img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
            if img is None:
                QMessageBox.information(self, 'info', 'Fail to load the image.', QMessageBox.Ok)
                return
            self.display.setPixmap(img)
            self.filename = filenamet
            self.results_list = None
            self.i = 0
        except AttributeError:
            QMessageBox.information(self, 'info', 'Fail to load the image.', QMessageBox.Ok)
        
        return 
    def fi(self):
        filenamet  = self.load.text()
        try:
            img = QtGui.QPixmap(filenamet)
            img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
            self.display.setPixmap(img)
            self.filename = filenamet
            self.results_list = None
            self.i = 0
        except AttributeError:
            QMessageBox.information(self, 'info', 'Fail to load the image. Please check your path.', QMessageBox.Ok)
        return
        
    def tangram_result(self):
        try:
            result = None
            if self.game_mode == 0:
                result = solve_result(self.filename)
            else:
                result = solver_nine_pieces.solve_result(self.filename)
            if result is None:
                QMessageBox.information(self, 'info', '失败  Sorry, fail to solve the puzzle.', QMessageBox.Ok)
                return                
        except AssertionError:
            QMessageBox.information(self, 'info', '失败  Sorry, fail to solve the puzzle.', QMessageBox.Ok)
            return
        if result is None:
            QMessageBox.information(self, 'info', '失败  Sorry, fail to solve the puzzle.', QMessageBox.Ok)
            return
        QMessageBox.information(self, 'info', '成功  Successful！', QMessageBox.Ok)
        # img = self.img2pixmap(result)
        # img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        cv2.imwrite('./debug/debug.png',result)
        img = QtGui.QPixmap('./debug/debug.png')
        img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        self.display.setPixmap(img)
        self.results_list = None
        self.i = 0

    def solve_tangram(self):
        try:
            if self.game_mode == 0:
                results = process_generator(self.filename)
            else:
                results = solver_nine_pieces.process_generator(self.filename)
            if results is None or len(results) == 0:
                QMessageBox.information(self, 'info', '失败  Sorry, fail to solve the puzzle.', QMessageBox.Ok)
                return                
        except AssertionError:
            QMessageBox.information(self, 'info', '失败  Sorry, fail to solve the puzzle.', QMessageBox.Ok)
            return
        if results is None or len(results) < 10:
            QMessageBox.information(self, 'info', '失败  Sorry, fail to solve the puzzle.', QMessageBox.Ok)
            return
        self.i = 0
        self.results_list = results
        QMessageBox.information(self, 'info', '成功!', QMessageBox.Ok)
        _translate = QtCore.QCoreApplication.translate
        self.info.setText(_translate("MainWindow", "点击 > 或 < 观察    Select > or < to view"))
        # img = self.img2pixmap()
        # img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        # self.display.setPixmap(self.results_list[self.i])
        cv2.imwrite('./debug/debug.png',self.results_list[self.i])
        img = QtGui.QPixmap('./debug/debug.png')
        img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        self.display.setPixmap(img)        

    def plus_action(self):
        if self.results_list is None or self.i == len(self.results_list) - 1:
            return
        self.i = self.i + 1
        cv2.imwrite('./debug/debug.png',self.results_list[self.i])
        img = QtGui.QPixmap('./debug/debug.png')
        img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        self.display.setPixmap(img)
        self.changeinfo()      
    def minus_action(self):
        if self.results_list is None or self.i == 0:
            return
        self.i = self.i - 1
        cv2.imwrite('./debug/debug.png',self.results_list[self.i])
        img = QtGui.QPixmap('./debug/debug.png')
        img = img.scaled(self.display.size(), QtCore.Qt.KeepAspectRatio)
        self.display.setPixmap(img)
        self.changeinfo()
    def changeinfo(self):
        _translate = QtCore.QCoreApplication.translate
        if self.i == 0:      
            self.info.setText(_translate("MainWindow", "点击 > 或 < 观察    Select > or < to view"))
        elif self.i == 1:      
            self.info.setText(_translate("MainWindow", "corner_points detected"))
        elif self.i == 2:      
            self.info.setText(_translate("MainWindow", "iteratvely found node_points"))
        elif self.i == 3:      
            self.info.setText(_translate("MainWindow", "legal edges connected"))
        elif self.i == len(self.results_list) - 1:      
            self.info.setText(_translate("MainWindow", "Finished"))            
        else:      
            self.info.setText(_translate("MainWindow", "DFS searching"))     

if __name__ == '__main__':
    if not os.path.exists('debug/'):
        os.makedirs('debug/')    
    app = QApplication(sys.argv)
    display = MainWindow()
    display.show()
    sys.exit(app.exec_())
