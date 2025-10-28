# YOLO Ai Assisted Annotation Tool
## Built by the Hellenic Center for Marine Research
### Ioannis Christofilogiannis, Dimitra Georgopoulou and Nikos Papandroulakis
### [Cure4Aqua EU project](https://cure4aqua-project.eu)

A PyQt6-based annotation tool for binary classification tasks with YOLO model integration. Originally designed for classifying Mediterranean fish fin quality (caudal and pectoral fins), this tool is highly adaptable for any binary classification annotation project.

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

![Tool Preview](preview.png)

## Features

### Core Functionality
- **AI-Assisted Annotation**: Load a pre-trained YOLO model to generate initial predictions
- **Manual Drawing**: Create bounding boxes with click-and-drag interface
- **Interactive Editing**: 
  - Drag boxes to reposition
  - Resize boxes using corner handles
  - Delete boxes with keyboard shortcuts
- **Dual Classification**: Quick toggle between two classes (e.g., "good_fin" and "bad_fin")
- **YOLO Format**: Automatic saving in YOLO-compatible format

### Advanced Features
- **Zoom Support**: 1x and 2x zoom levels for detailed annotation (toggle with `Z` key)
- **Dataset Navigation**: 
  - Previous/Next image navigation with arrow keys
  - Jump to specific image number
  - Progress tracking with visual progress bar
- **Smart Loading**: Auto-loads existing annotations from YOLO `.txt` files
- **Confidence Threshold**: Adjustable slider for model predictions (0.0-1.0)
- **Keyboard Shortcuts**: Efficient annotation workflow with hotkeys
- **Auto-save**: Changes automatically saved when navigating between images
- **Dual Mode Operation**: 
  - **YOLO Prediction Mode**: Generate AI-assisted predictions
  - **Dataset Mode**: Review and edit existing annotations
- **Dataset Split Support**: Easy switching between train/valid/test splits
- **Box Resizing**: Toggle-able corner handles for precise box adjustment
- **Hover-to-Edit**: Hover over any box to select it for deletion or class toggle
- **Clear All**: Remove all annotations from current image with one click

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qt-assisted-annotation.git
cd qt-assisted-annotation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your pre-trained YOLO model in the project directory
   - The default model name expected is `best_tail.pt`
   - Or modify the `model_path` parameter in the `main()` function to point to your model

## Usage

### Launching the Tool

Run the tool from the command line:
```bash
python qt_assisted_annotation.py
```

**Note**: By default, the tool expects a YOLO model file named `best_tail.pt` in the project directory with a confidence threshold of 0.4.

To use a different model or default confidence, modify the `main()` function in `qt_assisted_annotation.py`:
```python
# At the bottom of qt_assisted_annotation.py (lines 1104-1108)
annotator = AssistatedAnnotator(
    model_path='your_model.pt',  # Change to your model path
    confidence=0.4                # Adjust default confidence threshold
)
```

### Workflow

1. **Select Mode**:
   - **YOLO Prediction Mode**: Use AI to generate initial predictions
   - **Dataset Mode**: Work with existing annotations only

2. **Load Dataset**: Click "Load Dataset Directory" and select your dataset root folder
   - The tool expects a structure like: `dataset/train/images/` and `dataset/valid/images/`
   - Select your split (Train/Valid/Test) using the radio buttons

3. **Set Confidence** (Prediction Mode only): Adjust the slider to control model prediction sensitivity

4. **Annotate**:
   - Draw boxes by clicking and dragging on the image
   - Select class using radio buttons (Good Fin / Bad Fin)
   - **Hover over any box** to select it
   - Press `W` to toggle class of hovered box
   - Press `D` to delete hovered box
   - Enable resize mode with `R` key, then drag corner handles to adjust boxes

5. **Navigate**:
   - Use `←` `→` arrow keys or Previous/Next buttons
   - Enter image number in "Go to image" field and press Enter or click "Go"
   - Progress bar shows completion status
   - Press `Q` to refresh current image and reload predictions/annotations

6. **Save**: Annotations auto-save when navigating; use "Save Annotations" button or `Ctrl+S` for manual save

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` | Previous image |
| `→` | Next image |
| `D` | Delete hovered box |
| `W` | Toggle class of hovered box (Good/Bad) |
| `R` | Toggle resize mode on/off |
| `Z` | Toggle zoom (1x ↔ 2x) |
| `Q` | Refresh current image |
| `Tab` | Toggle between YOLO Prediction and Dataset modes |
| `Ctrl+S` | Save annotations |
| `Enter` | (in "Go to image" field) Jump to specific image |

## Dataset Structure

### Required Directory Structure

**The tool expects a standard YOLO dataset structure:**

```
my_dataset/                # Root dataset directory (select this in the tool)
├── train/
│   ├── images/           # Training images
│   │   ├── fish001.jpg
│   │   ├── fish002.jpg
│   │   └── ...
│   └── labels/           # Annotations (auto-created if missing)
│       ├── fish001.txt
│       ├── fish002.txt
│       └── ...
├── valid/                # Validation split
│   ├── images/
│   └── labels/
└── test/                 # Test split (optional)
    ├── images/
    └── labels/
```

### How to Load Your Dataset

1. **Click "Load Dataset Directory"**
2. **Select the ROOT dataset folder** (e.g., `my_dataset/` in the example above)
3. **Choose your split** using the radio buttons (Train/Valid/Test)
4. The tool will automatically look for `[split]/images/` and create/use `[split]/labels/`

### Supported Image Formats
- `.jpg` / `.jpeg`
- `.png` 
- `.bmp`
- `.tiff`

### Annotation Format

Annotations are saved in YOLO format (`.txt` files):
```
class_id x_center y_center width height
```

**Format specifications:**
- Each line represents one bounding box
- Values are space-separated
- All coordinates are normalized (0.0 to 1.0 relative to image dimensions)
- `class_id`: 0 for bad_fin, 1 for good_fin (configurable)
- `x_center`, `y_center`: Center point of the bounding box
- `width`, `height`: Dimensions of the bounding box

**Example annotation file** (`fish001.txt`):
```
1 0.5234 0.6123 0.1234 0.2341
0 0.7821 0.3456 0.0987 0.1654
1 0.2341 0.8234 0.1456 0.2123
```

### Important Notes

- **Filename matching**: Label files must have the same name as their corresponding image (e.g., `fish001.jpg` → `fish001.txt`)
- **Automatic directory creation**: If the `labels/` directory doesn't exist, it will be created automatically when you save
- **Path structure preservation**: When loading from `dataset/train/images/`, labels are saved to `dataset/train/labels/`
- **Non-destructive editing**: Existing annotations are preserved unless explicitly deleted or modified

## Customization

### Adapting for Your Use Case

The tool can be easily adapted for other binary classification tasks:

1. **Modify Class Names**: Update the class labels in the code:
```python
# In add_box() and relevant sections
self.canvas.labels.append(
    "class_positive" if self.good_fin_radio.isChecked() else "class_negative"
)
```

2. **Change Class IDs**: Adjust the constants at the top:
```python
DATASET_GOOD_FIN_CLASS_ID = 1  # Your positive class ID
YOLO_GOOD_FIN_CLASS_ID = 1     # Model's positive class ID
```

3. **Modify UI Labels**: Change radio button text:
```python
self.good_fin_radio = QRadioButton("Positive Class")
self.bad_fin_radio = QRadioButton("Negative Class")
```

### Future Enhancements
- Multi-class support (beyond binary classification)
- Additional annotation formats (COCO, Pascal VOC)
- Batch annotation features
- Annotation quality metrics

## Technical Details

### Dependencies
- **PyQt6**: GUI framework
- **OpenCV**: Image processing
- **Ultralytics YOLO**: Object detection model integration
- **NumPy**: Numerical operations
- **Pillow**: Image handling

### Performance
- Handles large image datasets efficiently
- Real-time zoom and pan with maintained annotation accuracy
- Coordinate scaling ensures precision across zoom levels

## Use Case: Fish Fin Classification

Originally developed for Mediterranean fish species fin quality assessment:
- **Good Fin**: Intact, undamaged fins suitable for analysis
- **Bad Fin**: Damaged, degraded, or unsuitable fins

This specialized use case demonstrates the tool's capability for detailed morphological annotation tasks in biological research.

## Contributing

Contributions are welcome! Areas for improvement:
- Multi-class annotation support
- Additional export formats
- Enhanced visualization options
- Annotation validation tools

## License

Apache License 2.0 - See LICENSE file for details

## Acknowledgments

Developed for Mediterranean fish morphology research, with applications across computer vision annotation tasks. Research funded by the Cure4Aqua EU project.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: Ensure your YOLO model is compatible with Ultralytics YOLO format. The tool expects a `.pt` model file.
