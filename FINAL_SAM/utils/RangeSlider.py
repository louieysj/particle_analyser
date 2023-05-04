from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
from PyQt5.QtWidgets import QWidget
import os
import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import pyqtSignal
import win32api, win32con
from collections import Counter


class RangeSlider_Visual(QWidget):
    rangeChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        
        self.data = None
        self.dragging = None
    
    def setData(self, data):
        # 输入的data是list，代表了每一个轮廓的面积area, 即[area1, area2, ...]
        # 因此需要根据area大小，统计每个area的轮廓数量，得到转换后的数据{area1: cnt1, area2:cnt2, ...}
        # RangeSlider里面调整的就是area的上下限，目的是筛选掉过大和过小的面积，只有range范围内的才通过
        # 转换后，横坐标为area的index，纵坐标为该area的数量
        self.areas = sorted(data, )
        self.count_dict = Counter(self.areas)
        self.data = list(self.count_dict.values())

        # self.data = data
        # self.left_handle = 0
        # self.right_handle = len(data) - 1
        self.left_handle = int(0.1*len(self.data))
        self.right_handle = len(self.data) - 1#max(1, int(0.1*len(self.data)))

        l_idx = int(self.left_handle)
        l_value = list(self.count_dict.keys())[l_idx]
        r_idx = int(self.right_handle)
        r_value = list(self.count_dict.keys())[r_idx]
        self.rangeChanged.emit(l_value, r_value)

        self.update()

    def paintEvent(self, event):
        if not self.data:
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.red)
            return
        
        painter = QPainter(self)

        # 绘制数据分布的直方图
        bin_count = len(self.data)
        bin_size = len(self.data) // bin_count
        bins = [sum(self.data[i:i+bin_size]) for i in range(0, len(self.data), bin_size)]
        max_bin = max(bins)
        bar_width = self.width() / bin_count

        brush = QBrush(Qt.SolidPattern)
        for i, bin_value in enumerate(bins):
            bar_height = self.height() * bin_value / max_bin
            x = i * bar_width
            y = self.height() - bar_height
            rect = QRectF(x, y, bar_width, bar_height)
            color = Qt.red if self.left_handle <= i <= self.right_handle else Qt.gray
            brush.setColor(color)
            painter.fillRect(rect, brush)

        # 绘制指针
        pen = QPen(Qt.blue, 3)
        painter.setPen(pen)
        painter.drawLine(self.left_handle * bar_width, 0, self.left_handle * bar_width, self.height())
        painter.drawLine(self.right_handle * bar_width, 0, self.right_handle * bar_width, self.height())

        # 绘制边框
        painter.setPen(QPen(Qt.black, 2))
        painter.drawRect(self.rect())

        painter.end()

    def mousePressEvent(self, event):
        if not self.data:
            return
        
        handle_width = 15
        bar_width = self.width() / len(self.data)

        # 判断点击位置是否在左侧或右侧的指针上
        left_handle_rect = QRectF(self.left_handle * bar_width - handle_width/2, 0, handle_width, self.height())
        right_handle_rect = QRectF(self.right_handle * bar_width - handle_width/2, 0, handle_width, self.height())
        if left_handle_rect.contains(event.pos()):
            self.dragging = "left"
        elif right_handle_rect.contains(event.pos()):
            self.dragging = "right"
        else:
            self.dragging = None

    def mouseMoveEvent(self, event):
        if not self.data:
            return
        
        bar_width = self.width() / len(self.data)
        handle_position = event.pos().x() / bar_width
        
        # 左右指针吸附
        l_dis = abs(event.pos().x() - self.left_handle * bar_width)
        r_dis = abs(event.pos().x() - self.right_handle * bar_width)

        if min(l_dis, r_dis)<8:
            win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_SIZEWE))
        else:
            win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_ARROW))
        
        if self.dragging is not None:
            if self.dragging == "left":
                self.left_handle = max(0, min(handle_position, self.right_handle - 1))
            elif self.dragging == "right":
                self.right_handle = min(len(self.data) - 1, max(handle_position, self.left_handle + 1))
            
            l_idx = int(self.left_handle)
            l_value = list(self.count_dict.keys())[l_idx]
            r_idx = int(self.right_handle)
            r_value = list(self.count_dict.keys())[r_idx]
            self.rangeChanged.emit(l_value, r_value)

            self.update()
    
    def leaveEvent(self, event):
        win32api.SetCursor(win32api.LoadCursor(0, win32con.IDC_ARROW))

    def mouseReleaseEvent(self, event):
        if not self.data:
            return
        
        if self.dragging is not None:
            l_idx = int(self.left_handle)
            l_value = list(self.count_dict.keys())[l_idx]
            r_idx = int(self.right_handle)
            r_value = list(self.count_dict.keys())[r_idx]
            self.rangeChanged.emit(l_value, r_value)

        self.dragging = None


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QVBoxLayout, QMainWindow

    data = [3, 5, 7, 2, 1, 8, 4, 6, 10, 90]*10

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            self.slider = RangeSlider_Visual()
            self.slider.setData(data)
            layout = QVBoxLayout()
            layout.addWidget(self.slider)
            central_widget = QWidget()
            central_widget.setLayout(layout)
            self.setCentralWidget(central_widget)

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
