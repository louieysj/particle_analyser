from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton

class ActualScaleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.num_edit = QLineEdit()
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['um', 'mm', ])
        
        ok_button = QPushButton('OK')
        ok_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.reject)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        form_layout = QVBoxLayout()
        form_layout.addWidget(QLabel('Number:'))
        form_layout.addWidget(self.num_edit)
        form_layout.addWidget(QLabel('Unit:'))
        form_layout.addWidget(self.unit_combo)
        form_layout.addLayout(button_layout)
        
        self.setLayout(form_layout)
        
    def get_data(self):
        if self.exec_() == QDialog.Accepted:
            num = float(self.num_edit.text())
            k = 1
            if self.unit_combo.currentText() == 'mm':
                k=1e3
            
            return num*k
        else:
            return None