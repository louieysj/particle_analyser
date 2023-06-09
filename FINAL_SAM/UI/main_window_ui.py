# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(977, 306)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 601, 111))
        self.groupBox.setObjectName("groupBox")
        self.label_current_folder = QtWidgets.QLabel(self.groupBox)
        self.label_current_folder.setGeometry(QtCore.QRect(130, 40, 251, 20))
        self.label_current_folder.setObjectName("label_current_folder")
        self.btn_detect_contour = QtWidgets.QPushButton(self.groupBox)
        self.btn_detect_contour.setGeometry(QtCore.QRect(399, 66, 191, 28))
        self.btn_detect_contour.setObjectName("btn_detect_contour")
        self.btn_goto_previous = QtWidgets.QPushButton(self.groupBox)
        self.btn_goto_previous.setGeometry(QtCore.QRect(400, 30, 93, 28))
        self.btn_goto_previous.setObjectName("btn_goto_previous")
        self.btn_goto_next = QtWidgets.QPushButton(self.groupBox)
        self.btn_goto_next.setGeometry(QtCore.QRect(500, 30, 93, 28))
        self.btn_goto_next.setObjectName("btn_goto_next")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 40, 121, 16))
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setObjectName("label")
        self.label_current_file = QtWidgets.QLabel(self.groupBox)
        self.label_current_file.setGeometry(QtCore.QRect(130, 70, 251, 20))
        self.label_current_file.setText("")
        self.label_current_file.setObjectName("label_current_file")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 111, 16))
        self.label_3.setObjectName("label_3")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 130, 601, 91))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_scale = QtWidgets.QLabel(self.groupBox_3)
        self.label_scale.setGeometry(QtCore.QRect(10, 20, 181, 21))
        self.label_scale.setObjectName("label_scale")
        self.slider_scale = QtWidgets.QSlider(self.groupBox_3)
        self.slider_scale.setGeometry(QtCore.QRect(200, 20, 391, 21))
        self.slider_scale.setMaximum(100)
        self.slider_scale.setSingleStep(5)
        self.slider_scale.setPageStep(5)
        self.slider_scale.setProperty("value", 25)
        self.slider_scale.setOrientation(QtCore.Qt.Horizontal)
        self.slider_scale.setObjectName("slider_scale")
        self.label_actual_scale = QtWidgets.QLabel(self.groupBox_3)
        self.label_actual_scale.setGeometry(QtCore.QRect(10, 50, 341, 16))
        self.label_actual_scale.setObjectName("label_actual_scale")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(620, 10, 341, 171))
        self.groupBox_4.setObjectName("groupBox_4")
        self.label_minmaxArea = QtWidgets.QLabel(self.groupBox_4)
        self.label_minmaxArea.setGeometry(QtCore.QRect(10, 50, 321, 41))
        self.label_minmaxArea.setObjectName("label_minmaxArea")
        self.checkBox_delete_edge_contour = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBox_delete_edge_contour.setGeometry(QtCore.QRect(10, 140, 211, 19))
        self.checkBox_delete_edge_contour.setObjectName("checkBox_delete_edge_contour")
        self.slider_area = QtWidgets.QSlider(self.groupBox_4)
        self.slider_area.setGeometry(QtCore.QRect(10, 100, 321, 21))
        self.slider_area.setMaximum(100)
        self.slider_area.setSingleStep(1)
        self.slider_area.setPageStep(1)
        self.slider_area.setProperty("value", 25)
        self.slider_area.setOrientation(QtCore.Qt.Horizontal)
        self.slider_area.setObjectName("slider_area")
        self.checkBox_draw_contours = QtWidgets.QCheckBox(self.groupBox_4)
        self.checkBox_draw_contours.setGeometry(QtCore.QRect(10, 30, 131, 21))
        self.checkBox_draw_contours.setChecked(True)
        self.checkBox_draw_contours.setObjectName("checkBox_draw_contours")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(800, 190, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 977, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTool = QtWidgets.QMenu(self.menubar)
        self.menuTool.setObjectName("menuTool")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.toolBar_2 = QtWidgets.QToolBar(MainWindow)
        self.toolBar_2.setObjectName("toolBar_2")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar_2)
        self.actionOpenFolder = QtWidgets.QAction(MainWindow)
        self.actionOpenFolder.setObjectName("actionOpenFolder")
        self.actionExportData = QtWidgets.QAction(MainWindow)
        self.actionExportData.setObjectName("actionExportData")
        self.actionModifyMode = QtWidgets.QAction(MainWindow)
        self.actionModifyMode.setCheckable(True)
        self.actionModifyMode.setObjectName("actionModifyMode")
        self.actionMeasureScale = QtWidgets.QAction(MainWindow)
        self.actionMeasureScale.setCheckable(True)
        self.actionMeasureScale.setObjectName("actionMeasureScale")
        self.actionImportScale = QtWidgets.QAction(MainWindow)
        self.actionImportScale.setObjectName("actionImportScale")
        self.menuFile.addAction(self.actionOpenFolder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExportData)
        self.menuTool.addAction(self.actionModifyMode)
        self.menuTool.addSeparator()
        self.menuTool.addAction(self.actionMeasureScale)
        self.menuTool.addSeparator()
        self.menuTool.addAction(self.actionImportScale)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTool.menuAction())
        self.toolBar.addAction(self.actionOpenFolder)
        self.toolBar_2.addSeparator()
        self.toolBar_2.addAction(self.actionModifyMode)
        self.toolBar_2.addSeparator()
        self.toolBar_2.addAction(self.actionMeasureScale)
        self.toolBar_2.addAction(self.actionImportScale)
        self.toolBar_2.addSeparator()
        self.toolBar_2.addSeparator()
        self.toolBar_2.addAction(self.actionExportData)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Detector-SAM"))
        self.groupBox.setTitle(_translate("MainWindow", "File Control Panel"))
        self.label_current_folder.setText(_translate("MainWindow", "Not Open Yet"))
        self.btn_detect_contour.setText(_translate("MainWindow", "Detect Contour"))
        self.btn_goto_previous.setText(_translate("MainWindow", "<"))
        self.btn_goto_next.setText(_translate("MainWindow", ">"))
        self.label.setText(_translate("MainWindow", "CurrentFolder:"))
        self.label_3.setText(_translate("MainWindow", "CurrentImage:"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Arguments Control"))
        self.label_scale.setText(_translate("MainWindow", "ShowScale:0.25"))
        self.label_actual_scale.setText(_translate("MainWindow", "ActualScale (um/pix): Not Set Yet"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Contour Control"))
        self.label_minmaxArea.setText(_translate("MainWindow", "MinArea:   , MaxArea:"))
        self.checkBox_delete_edge_contour.setText(_translate("MainWindow", "Delete Edge Contours"))
        self.checkBox_draw_contours.setText(_translate("MainWindow", "Draw Contour"))
        self.label_2.setText(_translate("MainWindow", "UoM 10652989"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuTool.setTitle(_translate("MainWindow", "Tool"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolBar_2.setWindowTitle(_translate("MainWindow", "toolBar_2"))
        self.actionOpenFolder.setText(_translate("MainWindow", "OpenFolder"))
        self.actionExportData.setText(_translate("MainWindow", "ExportData"))
        self.actionModifyMode.setText(_translate("MainWindow", "EditMode"))
        self.actionMeasureScale.setText(_translate("MainWindow", "MeasureScale"))
        self.actionImportScale.setText(_translate("MainWindow", "ImportScale"))
