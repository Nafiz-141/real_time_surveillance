from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSlider, QPushButton)
from PyQt5.QtCore import Qt

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Surveillance System Settings")
        
        layout = QVBoxLayout()
        
        # GPU Settings
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("GPU Acceleration:")
        self.gpu_checkbox = QCheckBox("Enable GPU")
        self.gpu_checkbox.setChecked(config['system']['gpu_enabled'])
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_checkbox)
        layout.addLayout(gpu_layout)
        
        # Confidence Threshold
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Object Detection Confidence:")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(int(config['models']['object_detection']['confidence_threshold'] * 100))
        confidence_layout.addWidget(confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        layout.addLayout(confidence_layout)
        
        # Theme Selection
        theme_layout = QHBoxLayout()
        theme_label = QLabel("Interface Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        current_theme = config['interface']['theme']
        self.theme_combo.setCurrentText(current_theme.capitalize())
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)
        
        # Save and Cancel Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def save_settings(self):
        # Update configuration
        self.config['system']['gpu_enabled'] = self.gpu_checkbox.isChecked()
        self.config['models']['object_detection']['confidence_threshold'] = self.confidence_slider.value() / 100
        self.config['interface']['theme'] = self.theme_combo.currentText().lower()
        
        # Close dialog with acceptance
        self.accept()