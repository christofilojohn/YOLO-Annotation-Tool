# Copyright (c) 2025, Hellenic Center for Marine Research,
# Ioannis Christofilogiannis, Dimitra Georgopoulou and Nikos Papandroulakis.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QRadioButton, QButtonGroup, QSlider, QMessageBox, QProgressBar, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import (QPixmap, QPainter, QPen, QColor, QImage, QKeySequence,
                         QShortcut)

DATASET_GOOD_FIN_CLASS_ID = 1
YOLO_GOOD_FIN_CLASS_ID = 1


class AnnotationCanvas(QLabel):
    def __init__(self, annotator):
        super().__init__()
        self.annotator = annotator
        self.drawing = False
        self.dragging = False
        self.resizing = False
        self.resize_enabled = False
        self.drag_start = None
        self.drag_box_idx = -1
        self.resize_corner = None
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.current_rect = None
        self.scale_factor = 1.0
        self.zoom_level = 1  # 1 for normal, 2 for 2x zoom
        self.boxes = []
        self.labels = []
        self.selected_box = -1
        self.hovered_box = -1
        self.corner_size = 10
        self.base_image = None
        self.scaled_image = None
        self.original_size = None

        self.setMouseTracking(True)
        self.installEventFilter(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image):
        """Set new base image and store original dimensions"""
        self.base_image = image
        self.original_size = (image.width(), image.height())
        self.update_scaled_image()

    def update_scaled_image(self):
        """Update the scaled image based on zoom level while maintaining aspect ratio"""
        if not self.base_image or not self.original_size:
            return

        # Calculate new dimensions based on zoom level
        new_width = self.original_size[0] * self.zoom_level
        new_height = self.original_size[1] * self.zoom_level

        # Scale the image using integer scaling
        self.scaled_image = self.base_image.scaled(
            new_width,
            new_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        # Update scale factor based on actual scaled dimensions
        self.scale_factor = self.scaled_image.width() / self.original_size[0]

        # Set fixed size to prevent any automatic scaling
        self.setFixedSize(self.scaled_image.size())
        self.setPixmap(self.scaled_image)
        self.update()

    def get_scaled_point(self, point):
        """Convert screen coordinates to original image coordinates"""
        if not self.scale_factor:
            return point
        return QPoint(
            int(point.x() / self.scale_factor),
            int(point.y() / self.scale_factor)
        )

    def get_display_point(self, point):
        """Convert original image coordinates to screen coordinates"""
        return QPoint(
            int(point.x() * self.scale_factor),
            int(point.y() * self.scale_factor)
        )

    def get_corner_rect(self, box, corner):
        """Return the rect for a resize handle with proper scaling"""
        x, y = 0, 0
        if 'top' in corner:
            y = box.top()
        if 'bottom' in corner:
            y = box.bottom()
        if 'left' in corner:
            x = box.left()
        if 'right' in corner:
            x = box.right()

        handle_size = self.corner_size * self.zoom_level
        scaled_x = int(x * self.scale_factor)
        scaled_y = int(y * self.scale_factor)

        return QRect(
            scaled_x - handle_size // 2,
            scaled_y - handle_size // 2,
            handle_size,
            handle_size
        )

    def get_resize_corner(self, pos, box_idx):
        """Determine if pos is over a resize handle of the given box"""
        if not self.resize_enabled or box_idx < 0:
            return None

        box = self.boxes[box_idx]
        corners = ['topleft', 'topright', 'bottomleft', 'bottomright']

        for corner in corners:
            if self.get_corner_rect(box, corner).contains(pos):
                return corner
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()

            if self.resize_enabled and self.hovered_box != -1:
                resize_corner = self.get_resize_corner(pos, self.hovered_box)
                if resize_corner:
                    self.resizing = True
                    self.resize_corner = resize_corner
                    self.drag_box_idx = self.hovered_box
                    self.drag_start = pos
                    return

            if self.hovered_box != -1:
                self.dragging = True
                self.drag_box_idx = self.hovered_box
                self.drag_start = pos
                self.original_box_pos = self.boxes[self.drag_box_idx].topLeft()
            else:
                self.drawing = True
                self.start_point = pos
                self.end_point = pos

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if self.drawing:
            self.end_point = pos
            self.update()
        elif self.dragging and self.drag_box_idx >= 0:
            delta = pos - self.drag_start
            current_box = self.boxes[self.drag_box_idx]
            new_pos = current_box.topLeft() + self.get_scaled_point(delta)
            self.boxes[self.drag_box_idx].moveTopLeft(new_pos)
            self.drag_start = pos
            self.annotator.modified = True
            self.update()
        elif self.resizing and self.drag_box_idx >= 0:
            box = self.boxes[self.drag_box_idx]
            scaled_delta = self.get_scaled_point(pos - self.drag_start)

            if 'top' in self.resize_corner:
                box.setTop(box.top() + scaled_delta.y())
            if 'bottom' in self.resize_corner:
                box.setBottom(box.bottom() + scaled_delta.y())
            if 'left' in self.resize_corner:
                box.setLeft(box.left() + scaled_delta.x())
            if 'right' in self.resize_corner:
                box.setRight(box.right() + scaled_delta.x())

            self.drag_start = pos
            self.annotator.modified = True
            self.update()
        else:
            # Hover detection
            old_hovered = self.hovered_box
            self.hovered_box = -1

            scaled_pos = self.get_scaled_point(pos)

            for i, box in enumerate(self.boxes):
                # Use unscaled coordinates for hit testing
                if box.contains(scaled_pos):
                    self.hovered_box = i
                    break

            if old_hovered != self.hovered_box:
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                rect = QRect(self.start_point, self.end_point).normalized()
                # Convert to image coordinates
                unscaled_rect = QRect(
                    self.get_scaled_point(rect.topLeft()),
                    self.get_scaled_point(rect.bottomRight())
                )
                if unscaled_rect.width() > 5 and unscaled_rect.height() > 5:
                    self.annotator.add_box(unscaled_rect)

            self.drawing = False
            self.dragging = False
            self.resizing = False
            self.resize_corner = None
            self.drag_box_idx = -1
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.scaled_image:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw existing boxes
        for i, (box, label) in enumerate(zip(self.boxes, self.labels)):
            # Scale the box coordinates precisely
            scaled_box = QRect(
                int(box.x() * self.scale_factor),
                int(box.y() * self.scale_factor),
                int(box.width() * self.scale_factor),
                int(box.height() * self.scale_factor)
            )

            is_good_fin = label == "good_fin"

            if i == self.hovered_box:
                color = QColor(0, 255, 0) if is_good_fin else QColor(255, 0, 0)
                fill_color = QColor(0, 255, 0, 50) if is_good_fin else QColor(255, 0, 0, 50)
                pen_width = max(4, 4 * self.zoom_level)
                pen = QPen(color, pen_width)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.fillRect(scaled_box, fill_color)

                if self.resize_enabled:
                    corner_pen = QPen(QColor(0, 0, 0), max(2, 2 * self.zoom_level))
                    corner_pen.setCosmetic(True)
                    painter.setPen(corner_pen)
                    for corner in ['topleft', 'topright', 'bottomleft', 'bottomright']:
                        corner_rect = self.get_corner_rect(box, corner)
                        painter.fillRect(corner_rect, QColor(255, 255, 255))
                        painter.drawRect(corner_rect)
            else:
                color = QColor(0, 200, 0) if is_good_fin else QColor(200, 0, 0)
                pen = QPen(color, max(2, 2 * self.zoom_level))
                pen.setCosmetic(True)
                painter.setPen(pen)

            painter.drawRect(scaled_box)

        # Draw current box being drawn
        if self.drawing:
            pen = QPen(QColor("yellow"), max(1, 1 * self.zoom_level))
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_point, self.end_point).normalized())

    def set_zoom_level(self, level):
        """Set zoom level and update display"""
        if self.zoom_level != level:
            self.zoom_level = level
            if self.base_image:
                self.update_scaled_image()
class AssistatedAnnotator(QMainWindow):
    def __init__(self, model_path=None, confidence=0.4):
        super().__init__()
        self.model = None
        self.model_path = model_path
        if model_path:
            self.model = YOLO(model_path)
        self.confidence = confidence
        self.current_image = None
        self.current_image_path = None
        self.modified = False
        self.dataset_mode = False
        self.current_idx = 0
        self.total_images = 0
        self.image_paths = []
        self.setup_ui()
        self.setup_shortcuts()

        # Disable buttons initially
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.resize_toggle.setEnabled(False)
        self.clear_annotations_btn.setEnabled(False)

    def setup_ui(self):
        self.setWindowTitle("Assisted Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Canvas
        self.canvas = AnnotationCanvas(self)
        self.canvas.setMinimumSize(800, 600)
        layout.addWidget(self.canvas)
        # Right sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        layout.addWidget(sidebar)

        # Mode selection group
        mode_group = QButtonGroup(self)
        self.predict_mode_radio = QRadioButton("YOLO Prediction Mode")
        self.dataset_mode_radio = QRadioButton("Dataset Mode")
        self.predict_mode_radio.setChecked(True)
        mode_group.addButton(self.predict_mode_radio)
        mode_group.addButton(self.dataset_mode_radio)
        sidebar_layout.addWidget(self.predict_mode_radio)
        sidebar_layout.addWidget(self.dataset_mode_radio)

        # Connect mode change signals
        self.predict_mode_radio.toggled.connect(self.toggle_mode)

        # Dataset split selection
        split_group = QButtonGroup(self)
        self.split_label = QLabel("Dataset Split:")
        sidebar_layout.addWidget(self.split_label)

        self.train_radio = QRadioButton("Train")
        self.valid_radio = QRadioButton("Valid")
        self.test_radio = QRadioButton("Test")
        self.train_radio.setChecked(True)

        split_group.addButton(self.train_radio)
        split_group.addButton(self.valid_radio)
        split_group.addButton(self.test_radio)

        sidebar_layout.addWidget(self.train_radio)
        sidebar_layout.addWidget(self.valid_radio)
        sidebar_layout.addWidget(self.test_radio)
        # Connect split selection signals
        self.train_radio.toggled.connect(self.split_changed)
        self.valid_radio.toggled.connect(self.split_changed)
        self.test_radio.toggled.connect(self.split_changed)

        # Image filename display
        # self.filename_label = QLabel("Current Image:")
        # self.filename_display = QLabel()
        # self.filename_display.setStyleSheet("color: blue; font-weight: bold;")
        # self.filename_display.setWordWrap(True)
        # sidebar_layout.addWidget(self.filename_label)
        # sidebar_layout.addWidget(self.filename_display)

        # Buttons and their shortcut labels
        self.load_dir_btn = QPushButton("Load Dataset Directory")  # Updated text
        self.load_dir_btn.clicked.connect(self.load_directory)
        sidebar_layout.addWidget(self.load_dir_btn)

        # Model selection button
        self.load_model_btn = QPushButton("Load YOLO Model")
        self.load_model_btn.clicked.connect(self.load_model)
        sidebar_layout.addWidget(self.load_model_btn)

        self.model_label = QLabel("Model: None loaded")
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(self.model_label)

        self.next_btn = QPushButton("Next Image")
        self.next_btn.clicked.connect(self.next_image)
        sidebar_layout.addWidget(self.next_btn)
        next_shortcut = QLabel("→ (Right Arrow)")
        next_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        next_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(next_shortcut)
        
        self.refresh_btn = QPushButton("Refresh Image")
        self.refresh_btn.clicked.connect(self.refresh_image)
        sidebar_layout.addWidget(self.refresh_btn)
        refresh_shortcut = QLabel("Tab")
        refresh_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        refresh_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(refresh_shortcut)

        self.prev_btn = QPushButton("Previous Image")
        self.prev_btn.clicked.connect(self.previous_image)
        sidebar_layout.addWidget(self.prev_btn)
        prev_shortcut = QLabel("← (Left Arrow)")
        prev_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        prev_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(prev_shortcut)

        self.delete_btn = QPushButton("Delete Selected Box")
        self.delete_btn.clicked.connect(lambda: self.delete_box(-1))
        sidebar_layout.addWidget(self.delete_btn)
        delete_shortcut = QLabel("D")
        delete_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        delete_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(delete_shortcut)

        self.delete_dialog_checkbox = QCheckBox("Show Delete Confirmation")
        self.delete_dialog_checkbox.setChecked(True)  # Default to showing dialog
        sidebar_layout.addWidget(self.delete_dialog_checkbox)

        # Add clear annotations button
        self.clear_annotations_btn = QPushButton("Clear All Annotations")
        self.clear_annotations_btn.clicked.connect(self.clear_annotations)
        self.clear_annotations_btn.setEnabled(False)  # Disabled by default
        sidebar_layout.addWidget(self.clear_annotations_btn)

        self.save_btn = QPushButton("Save Annotations")
        self.save_btn.clicked.connect(self.save_annotations)
        sidebar_layout.addWidget(self.save_btn)
        save_shortcut = QLabel("Ctrl+S")
        save_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        save_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(save_shortcut)

        # Add refresh shortcut
        refresh_shortcut = QShortcut(QKeySequence("Q"), self)
        refresh_shortcut.activated.connect(self.refresh_image)

        # Resize toggle
        self.resize_toggle = QPushButton("Enable Box Resizing")
        self.resize_toggle.setCheckable(True)
        self.resize_toggle.clicked.connect(self.toggle_resize_mode)
        sidebar_layout.addWidget(self.resize_toggle)
        resize_shortcut = QLabel("R (Toggle Resize Mode)")
        resize_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        resize_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(resize_shortcut)

        # Class selection
        class_group = QButtonGroup(self)
        self.good_fin_radio = QRadioButton("Good Fin")
        self.bad_fin_radio = QRadioButton("Bad Fin")
        self.good_fin_radio.setChecked(True)
        class_group.addButton(self.good_fin_radio)
        class_group.addButton(self.bad_fin_radio)
        sidebar_layout.addWidget(self.good_fin_radio)
        sidebar_layout.addWidget(self.bad_fin_radio)
        toggle_shortcut = QLabel("W (Toggle Classification)")
        toggle_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toggle_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(toggle_shortcut)

        # Confidence threshold
        self.conf_label = QLabel("Confidence Threshold")
        sidebar_layout.addWidget(self.conf_label)
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.confidence * 100))
        self.conf_slider.valueChanged.connect(self.confidence_changed)
        sidebar_layout.addWidget(self.conf_slider)

        # Dataset progress section
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)

        # Image count and progress display
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("""
            color: #333;
            font-weight: bold;
            font-size: 14px;
            padding: 5px;
            background-color: #f0f0f0;
            border-radius: 4px;
        """)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setMinimumHeight(30)
        progress_layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
                   QProgressBar {
                       border: 2px solid grey;
                       border-radius: 5px;
                       text-align: center;
                   }
                   QProgressBar::chunk {
                       background-color: #4CAF50;
                       width: 10px;
                   }
               """)
        progress_layout.addWidget(self.progress_bar)

        # Add go to image control
        goto_widget = QWidget()
        goto_layout = QHBoxLayout(goto_widget)
        goto_layout.setContentsMargins(0, 0, 0, 0)

        goto_label = QLabel("Go to image:")
        goto_layout.addWidget(goto_label)

        self.goto_input = QLineEdit()
        self.goto_input.setPlaceholderText("Enter image #")
        self.goto_input.setFixedWidth(80)
        self.goto_input.returnPressed.connect(self.goto_image)  # Add Enter key support
        goto_layout.addWidget(self.goto_input)

        goto_btn = QPushButton("Go")
        goto_btn.clicked.connect(self.goto_image)
        goto_layout.addWidget(goto_btn)

        sidebar_layout.addWidget(goto_widget)

        # Add zoom controls after the resize toggle
        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)

        zoom_label = QLabel("Zoom:")
        zoom_layout.addWidget(zoom_label)

        self.zoom_1x = QRadioButton("1x")
        self.zoom_2x = QRadioButton("2x")
        self.zoom_1x.setChecked(True)

        zoom_layout.addWidget(self.zoom_1x)
        zoom_layout.addWidget(self.zoom_2x)

        self.zoom_1x.toggled.connect(self.update_zoom)
        self.zoom_2x.toggled.connect(self.update_zoom)

        sidebar_layout.addWidget(zoom_widget)

        # Add zoom keyboard shortcut hint
        zoom_shortcut = QLabel("Z (Toggle Zoom)")
        zoom_shortcut.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_shortcut.setStyleSheet("color: gray; font-size: 10px;")
        sidebar_layout.addWidget(zoom_shortcut)

        # Current image filename display
        self.filename_label = QLabel("Current Image:")
        self.filename_display = QLabel()
        self.filename_display.setStyleSheet("color: blue; font-weight: bold;")
        self.filename_display.setWordWrap(True)

        progress_layout.addWidget(self.filename_label)
        progress_layout.addWidget(self.filename_display)
        sidebar_layout.addWidget(progress_widget)

        sidebar_layout.addStretch()

    def load_model(self):
        """Open file dialog to select a YOLO model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            "",
            "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_path = file_path
                # Display just the filename, not the full path
                model_name = Path(file_path).name
                self.model_label.setText(f"Model: {model_name}")
                self.model_label.setStyleSheet("color: green; font-size: 10px;")
                QMessageBox.information(self, "Success", f"Model loaded: {model_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
                self.model = None
                self.model_label.setText("Model: None loaded")
                self.model_label.setStyleSheet("color: red; font-size: 10px;")

    def split_changed(self):
        """Handle changing dataset split"""
        if hasattr(self, 'image_dir') and self.image_dir:
            # Get the root dataset directory (two levels up from image_dir)
            dataset_root = self.image_dir.parent.parent

            # Get new split
            split = self.get_selected_split()

            # Update paths
            new_images_dir = dataset_root / split / "images"
            new_labels_dir = dataset_root / split / "labels"

            if not new_images_dir.exists():
                QMessageBox.warning(self, "Error",
                                    f"Split directory not found: {new_images_dir}")
                return

            # Save current annotations if modified
            if self.modified:
                reply = QMessageBox.question(
                    self,
                    'Unsaved Changes',
                    'Do you want to save changes before switching splits?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )

                if reply == QMessageBox.StandardButton.Cancel:
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self.save_current_annotations()

            # Update directories and load new images
            self.image_dir = new_images_dir
            self.labels_dir = new_labels_dir
            # Only include real image files, excluding dotfiles like ._filename.jpg
            self.image_paths = sorted([
                p for p in new_images_dir.glob("*.jpg")
                if not p.name.startswith("._") and p.is_file()
            ])

            self.current_idx = 0

            # Update progress tracking
            self.progress_bar.setMaximum(len(self.image_paths))
            self.update_progress_display()
            self.total_images = len(self.image_paths)
            # Load first image of new split
            if self.image_paths:
                self.load_current_image()
            else:
                QMessageBox.warning(self, "Error", f"No images found in {split} split")
                
    def setup_shortcuts(self):
        # Setup keyboard shortcuts
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_annotations)

        delete_shortcut = QShortcut(QKeySequence("D"), self)
        delete_shortcut.activated.connect(self.delete_hovered_box)

        toggle_shortcut = QShortcut(QKeySequence("W"), self)
        toggle_shortcut.activated.connect(self.toggle_hovered_box_class)

        # Add next/previous image shortcuts
        next_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        next_shortcut.activated.connect(self.next_image)

        prev_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        prev_shortcut.activated.connect(self.previous_image)

        # Add resize toggle shortcut
        resize_shortcut = QShortcut(QKeySequence("R"), self)
        resize_shortcut.activated.connect(self.toggle_resize_mode)

        mode_toggle_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        mode_toggle_shortcut.activated.connect(self.toggle_mode_shortcut)

        # Add zoom toggle shortcut
        zoom_shortcut = QShortcut(QKeySequence("Z"), self)
        zoom_shortcut.activated.connect(self.toggle_zoom)

    def get_selected_split(self):
        """Get the currently selected dataset split"""
        if self.train_radio.isChecked():
            return "train"
        elif self.valid_radio.isChecked():
            return "valid"
        else:
            return "test"

    def toggle_mode(self, checked):
        """Handle switching between prediction and dataset modes"""
        if self.modified:
            reply = QMessageBox.question(
                self,
                'Unsaved Changes',
                'Do you want to save changes before switching modes?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Cancel:
                # Revert radio button selection
                self.predict_mode_radio.setChecked(not self.dataset_mode)
                self.dataset_mode_radio.setChecked(self.dataset_mode)
                return
            elif reply == QMessageBox.StandardButton.Yes:
                self.save_current_annotations()

        self.dataset_mode = not checked
        self.conf_slider.setEnabled(not self.dataset_mode)
        self.conf_label.setEnabled(not self.dataset_mode)

        # Refresh the current image if one is loaded
        if hasattr(self, 'current_image_path') and self.current_image_path:
            self.load_current_image()

    def toggle_mode_shortcut(self):
        if self.predict_mode_radio.isChecked():
            self.dataset_mode_radio.setChecked(True)
        else:
            self.predict_mode_radio.setChecked(True)

    def toggle_resize_mode(self):
        """Toggle box resize mode"""
        # Update the button state
        self.resize_toggle.setChecked(not self.resize_toggle.isChecked())
        # Update the canvas resize state
        self.canvas.resize_enabled = self.resize_toggle.isChecked()
        # Update the button text
        self.resize_toggle.setText("Disable Box Resizing" if self.canvas.resize_enabled else "Enable Box Resizing")
        self.canvas.update()

    def toggle_zoom(self):
        """Toggle between 1x and 2x zoom"""
        if self.zoom_1x.isChecked():
            self.zoom_2x.setChecked(True)
        else:
            self.zoom_1x.setChecked(True)

    def update_zoom(self):
        """Update the canvas zoom level"""
        zoom_level = 2 if self.zoom_2x.isChecked() else 1
        self.canvas.set_zoom_level(zoom_level)

    def display_image(self):
        if self.current_image is None:
            return

        height, width = self.current_image.shape[:2]
        bytes_per_line = 3 * width
        qt_image = QImage(self.current_image.data, width, height,
                          bytes_per_line, QImage.Format.Format_RGB888)

        # Create base pixmap
        base_pixmap = QPixmap.fromImage(qt_image)

        # Set the base image in the canvas
        self.canvas.set_image(base_pixmap)

    def confidence_changed(self, value):
        self.confidence = value / 100
        if self.current_image_path:
            self.load_current_image()

    def load_directory(self):
        """Handle directory loading for both modes"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Root Directory")
        if not dir_path:
            return

        dataset_path = Path(dir_path)
        # TODO
        split = self.get_selected_split()

        # Check for proper dataset structure
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            QMessageBox.warning(self, "Error",
                                f"Invalid dataset structure. Expected: {dataset_path}/{split}/images/")
            return

        self.image_dir = images_dir
        self.labels_dir = labels_dir
        # Remove hidden mac OS files
        # Only include real image files, excluding dotfiles like ._filename.jpg
        self.image_paths = sorted([
            p for p in images_dir.glob("*.jpg")
            if not p.name.startswith("._") and p.is_file()
        ])
        # TODO
        self.total_images = len(self.image_paths)
        self.progress_bar.setMaximum(self.total_images)
        self.update_progress_display()

        if not self.image_paths:
            QMessageBox.warning(self, "Error", "No images found in the selected directory")
            return

        # Enable all buttons after successful loading
        self.enable_buttons()

        self.current_idx = 0
        self.load_current_image()

    # Add new method to enable/disable buttons
    def enable_buttons(self):
        """Enable all buttons after loading directory"""
        self.next_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.resize_toggle.setEnabled(True)
        self.clear_annotations_btn.setEnabled(True)

    def load_prediction_directory(self):
        """Original directory loading with YOLO predictions"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.image_dir = Path(dir_path)
            # Only include real image files, excluding dotfiles like ._filename.jpg
            self.image_paths = sorted([
                p for p in self.image_dir.glob("*.jpg")
                if not p.name.startswith("._") and p.is_file()
            ])

            self.current_idx = 0
            self.load_current_image()

    def refresh_image(self):
        """Reload the current image and its annotations or predictions"""
        if hasattr(self, 'current_idx') and self.current_image_path:
            if self.modified:
                reply = QMessageBox.question(
                    self,
                    'Unsaved Changes',
                    'Do you want to save changes before refreshing?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )

                if reply == QMessageBox.StandardButton.Cancel:
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self.save_current_annotations()

            self.load_current_image()

    # TODO
    def load_dataset_directory(self):
        """Load directory in dataset mode"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            dataset_path = Path(dir_path)

            # Look for images and labels in Roboflow/YOLO format
            images_dir = dataset_path / "train" / "images"
            if images_dir.exists():
                self.image_dir = images_dir
                # Only include real image files, excluding dotfiles like ._filename.jpg
                self.image_paths = sorted([
                    p for p in images_dir.glob("*.jpg")
                    if not p.name.startswith("._") and p.is_file()
                ])

                self.current_idx = 0
                self.load_current_image()
            else:
                QMessageBox.warning(self, "Error",
                                    "Invalid dataset directory structure. Expected: dataset_dir/train/images/")

    def toggle_hovered_box_class(self):
        if self.canvas.hovered_box != -1:
            current_label = self.canvas.labels[self.canvas.hovered_box]
            # Toggle between good_fin and bad_fin
            new_label = "bad_fin" if current_label == "good_fin" else "good_fin"
            self.canvas.labels[self.canvas.hovered_box] = new_label
            self.modified = True
            self.canvas.update()

    def delete_hovered_box(self):
        if self.canvas.hovered_box != -1:
            if self.delete_dialog_checkbox.isChecked():
                reply = QMessageBox.question(
                    self,
                    'Delete Box',
                    'Do you want to delete this box?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.delete_box(self.canvas.hovered_box)
            else:
                # Delete without confirmation
                self.delete_box(self.canvas.hovered_box)

    def clear_annotations(self):
        if not self.canvas.boxes:
            return

        if self.delete_dialog_checkbox.isChecked():
            reply = QMessageBox.question(
                self,
                'Clear Annotations',
                'Do you want to remove all annotations from this image?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.canvas.boxes = []
        self.canvas.labels = []
        self.modified = True
        self.canvas.update()

    def load_current_image(self):
        """Load image based on current mode"""
        if not hasattr(self, 'image_paths') or not self.image_paths:
            return

        image_path = self.image_paths[self.current_idx]
        self.current_image_path = str(image_path)

        # Update filename display
        self.filename_display.setText(image_path.name)
        self.update_progress_display()

        self.current_image = cv2.imread(self.current_image_path)
        if self.current_image is None:
            QMessageBox.critical(
                self,
                "Image Load Error",
                f"Failed to load image:\n{self.current_image_path}"
            )
            return
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)

        # Clear existing boxes
        self.canvas.boxes = []
        self.canvas.labels = []

        if self.dataset_mode:
            # Load existing annotations
            label_path = self.labels_dir / image_path.with_suffix('.txt').name
            if label_path.exists():
                with open(label_path, 'r') as f:
                    img_height, img_width = self.current_image.shape[:2]
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())

                        # Convert YOLO format to pixel coordinates
                        x = int((x_center - width / 2) * img_width)
                        y = int((y_center - height / 2) * img_height)
                        w = int(width * img_width)
                        h = int(height * img_height)

                        self.canvas.boxes.append(QRect(x, y, w, h))
                        self.canvas.labels.append("good_fin" if class_id == DATASET_GOOD_FIN_CLASS_ID else "bad_fin")
        else:
            if not self.model:
                QMessageBox.warning(self, "No Model", "Please load a YOLO model first to use prediction mode.")
                self.display_image()
                return
            # Get predictions from YOLO model
            results = self.model.predict(self.current_image_path, conf=self.confidence)
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    rect = QRect(
                        int(xyxy[0]), int(xyxy[1]),
                        int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])
                    )
                    self.canvas.boxes.append(rect)
                    self.canvas.labels.append(
                        "good_fin" if box.cls[0] == YOLO_GOOD_FIN_CLASS_ID else "bad_fin"
                    )

        self.display_image()

    def add_box(self, rect):
        self.canvas.boxes.append(rect)
        self.canvas.labels.append(
            "good_fin" if self.good_fin_radio.isChecked() else "bad_fin"
        )
        self.modified = True
        self.canvas.update()

    def delete_box(self, index):
        if index >= 0 and index < len(self.canvas.boxes):
            del self.canvas.boxes[index]
            del self.canvas.labels[index]
            self.modified = True
            self.canvas.update()

    def next_image(self):
        if self.modified:
            self.save_current_annotations()
        if hasattr(self, 'current_idx'):
            self.current_idx = (self.current_idx + 1) % len(self.image_paths)
            self.update_progress_display()
            self.load_current_image()

    def previous_image(self):
        if self.modified:
            self.save_current_annotations()
        if hasattr(self, 'current_idx'):
            self.current_idx = (self.current_idx - 1) % len(self.image_paths)
            self.update_progress_display()
            self.load_current_image()

    def save_current_annotations(self):
        """Save annotations in YOLO format"""
        if not self.current_image_path:
            return

        try:
            # Ensure labels directory exists
            if not hasattr(self, 'labels_dir'):
                # If labels_dir is not set, create it parallel to images
                image_path = Path(self.current_image_path)
                self.labels_dir = image_path.parent.parent / "labels"

            # Create labels directory if it doesn't exist
            self.labels_dir.mkdir(parents=True, exist_ok=True)

            # Convert to YOLO format
            img_height, img_width = self.current_image.shape[:2]
            yolo_annotations = []

            for box, label in zip(self.canvas.boxes, self.canvas.labels):
                # Convert Qt rect to YOLO format
                x_center = (box.x() + box.width() / 2) / img_width
                y_center = (box.y() + box.height() / 2) / img_height
                width = box.width() / img_width
                height = box.height() / img_height

                class_id = 1 if label == "good_fin" else 0
                yolo_annotations.append(
                    f"{class_id} {x_center} {y_center} {width} {height}"
                )

            # Save to labels directory
            image_path = Path(self.current_image_path)
            label_path = self.labels_dir / image_path.with_suffix('.txt').name

            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            self.modified = False

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Annotations",
                f"Failed to save annotations: {str(e)}\nPath: {label_path}"
            )
            return False

        return True

    def save_annotations(self):
        """Save all annotations and maintain dataset structure"""
        if self.modified:
            self.save_current_annotations()

        # Provide feedback about save location
        if self.dataset_mode:
            message = "Annotations saved in dataset structure at:\n"
            message += str(Path(self.current_image_path).parent.parent)
        else:
            dataset_path = Path(self.current_image_path).parent / "dataset"
            message = "Dataset saved with YOLO format at:\n"
            message += str(dataset_path)
            message += "\n\nDirectory structure created:\n"
            message += "- train/\n  - images/\n  - labels/"

        QMessageBox.information(self, "Success", message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image is not None:
            self.display_image()

    def goto_image(self):
        """Navigate to a specific image number entered by the user"""
        if not hasattr(self, 'image_paths') or not self.image_paths:
            QMessageBox.warning(self, "Error", "No images loaded. Please load a dataset first.")
            return

        try:
            # Get the entered image number (1-based index for user, 0-based for code)
            image_number = int(self.goto_input.text())

            # Validate the number is within range
            if image_number < 1 or image_number > len(self.image_paths):
                QMessageBox.warning(
                    self,
                    "Invalid Image Number",
                    f"Please enter a number between 1 and {len(self.image_paths)}."
                )
                return

            # Check for unsaved changes
            if self.modified:
                reply = QMessageBox.question(
                    self,
                    'Unsaved Changes',
                    'Do you want to save changes before navigating to another image?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
                )

                if reply == QMessageBox.StandardButton.Cancel:
                    return
                elif reply == QMessageBox.StandardButton.Yes:
                    self.save_current_annotations()

            # Navigate to the requested image (convert to 0-based index)
            self.current_idx = image_number - 1
            self.load_current_image()

            # Clear the input field after successful navigation
            self.goto_input.clear()

        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid number."
            )

    def update_progress_display(self):
        if hasattr(self, 'total_images') and self.total_images > 0:
            current = self.current_idx + 1
            self.progress_label.setText(f"Image {current} of {self.total_images}")
            self.progress_bar.setValue(current)

def main():
    app = QApplication(sys.argv)
    annotator = AssistatedAnnotator(confidence=0.4)

    annotator.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
