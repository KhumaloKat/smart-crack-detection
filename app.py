# importing libraries
from operator import truediv
import warnings
from PyQt6 import QtCore, QtWidgets
from tkinter import scrolledtext
from PyQt6.QtWidgets import *
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import gdown
import sys
import os
import re
import time
import sys
import threading
from datetime import datetime
import csv
import io
# For Extracting Metadata
from PIL import Image
from PIL.ExifTags import TAGS
# For quantification
import numpy as np
# Generating Report
from showinfm import show_in_file_manager

# Detectron 2 libraries
import cv2
import os
import numpy as np
# Ignore depreciated warnings
warnings.filterwarnings('ignore')

# For Progressbar


class PercentageWorker(QtCore.QObject):
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    percentageChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percentage = 0

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value):
        if self._percentage == value:
            return
        self._percentage = value
        self.percentageChanged.emit(self.percentage)

    def start(self):
        self.started.emit()

    def finish(self):
        self.finished.emit()


class FakeWorker:
    def start(self):
        pass

    def finish(self):
        pass

    @property
    def percentage(self):
        return 0

    @percentage.setter
    def percentage(self, value):
        pass


CRACK_CLASSES = ['diagonal_crack', 'horizontal_crack', 'vertical_crack']
YOLO_WEIGHTS_NAME = "yolov8m-crack-seg.pt"
YOLO_HF_REPO = "OpenSistemas/YOLOv8-crack-seg"
YOLO_HF_FILE = "yolov8m/weights/best.pt"
DETECTRON_WEIGHTS_NAME = "model_final.pth"


def model_path(filename):
    return os.path.join(os.getcwd(), "output", filename)


def sidecar_path(image_path, suffix):
    root, ext = os.path.splitext(image_path)
    return root + suffix + ext


def skeletonize(binary_mask):
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary_mask)
    except Exception:
        pass
    skeleton = np.zeros(binary_mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_mask.copy()
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def classify_crack_orientation(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return "diagonal_crack"
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    angle = abs(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle > 90:
        angle = 180 - angle
    if angle <= 20:
        return "horizontal_crack"
    if angle >= 70:
        return "vertical_crack"
    return "diagonal_crack"


def draw_detections(im, instances, show_boxes=True):
    overlay = im.copy()
    colors = {
        "diagonal_crack": (0, 0, 180),
        "horizontal_crack": (0, 80, 200),
        "vertical_crack": (0, 120, 40),
    }
    color_layer = overlay.astype(np.float32)
    for inst in instances:
        color = np.array(colors.get(inst["class"], (40, 40, 40)), dtype=np.float32)
        mask = inst["mask"].astype(bool)
        color_layer[mask] = color_layer[mask] * 0.30 + color * 0.70
    overlay = color_layer.astype(np.uint8)
    for inst in instances:
        color = colors.get(inst["class"], (40, 40, 40))
        mask = inst["mask"].astype(bool)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if show_boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{inst['class'].replace('_', ' ')} {inst['score'] * 100:.0f}%"
        origin = (x1, max(20, y1 - 8))
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return overlay


def resize_mask(mask, image_shape):
    height, width = image_shape[:2]
    if mask.shape[0] == height and mask.shape[1] == width:
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def yolo_to_instances(result, image_shape):
    instances = []
    if result.boxes is None or len(result.boxes) == 0:
        return instances
    scores = result.boxes.conf.cpu().numpy()
    masks = None
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
    for i, score in enumerate(scores):
        if masks is not None and i < len(masks):
            mask = resize_mask(masks[i], image_shape)
        else:
            x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().astype(int)
            mask = np.zeros(image_shape[:2], dtype=bool)
            mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = True
        instances.append({
            "mask": mask,
            "score": float(score),
            "class": classify_crack_orientation(mask),
        })
    return instances


def detectron_to_instances(outputs, image_shape):
    instances = []
    pred = outputs["instances"].to("cpu")
    if len(pred) == 0:
        return instances
    scores = pred.scores.numpy()
    masks = pred.pred_masks.numpy()
    for i, score in enumerate(scores):
        mask = resize_mask(masks[i], image_shape)
        instances.append({
            "mask": mask,
            "score": float(score),
            "class": classify_crack_orientation(mask),
        })
    return instances


def load_crack_detector(thresholdLower):
    conf = max(0.01, thresholdLower / 100.0)
    yolo_weights = model_path(YOLO_WEIGHTS_NAME)
    if os.path.exists(yolo_weights):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(yolo_weights)

            def predict_yolo(im):
                results = yolo_model.predict(
                    im, conf=conf, verbose=False, device="cpu", retina_masks=True)
                return yolo_to_instances(results[0], im.shape)

            print("Using YOLOv8-crack-seg detector")
            return predict_yolo
        except Exception as e:
            print(f"YOLO detector failed ({e}). Falling back to Mask R-CNN.")

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo
    from detectron2.utils.logger import setup_logger
    setup_logger()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.WEIGHTS = model_path(DETECTRON_WEIGHTS_NAME)
    predictor = DefaultPredictor(cfg)

    def predict_detectron(im):
        return detectron_to_instances(predictor(im), im.shape)

    print("Using Mask R-CNN detector")
    return predict_detectron


def analyseImage(foo, dir_path, thresholdLower, thresholdUpper, ResultPossibleFolder, ResultConfidentFolder, baz="1", worker=None, show_boxes=True):
    # Get list of images to process
    imageFileList = []
    for images in os.listdir(dir_path):
        if images.lower().endswith(('.png', '.jpg', '.jpeg')):
            imageFileList.append(images)

    total_images = len(imageFileList)
    if total_images == 0:
        print("\nNo images to analyze in", dir_path)
        if worker:
            worker.percentage = 100
            worker.finish()
        return

    print(f"Analyzing {total_images} Images...")
    if worker is None:
        worker = FakeWorker()
    worker.start()

    detect_cracks = load_crack_detector(thresholdLower)

    for idx, image_name in enumerate(imageFileList):
        percent = int(((idx + 1) / total_images) * 100)
        worker.percentage = percent
        print(f"[{percent}%] Analyzing {image_name}")

        try:
            im = cv2.imread(os.path.join(dir_path, image_name))
            if im is None:
                print(f"Failed to read image: {image_name}")
                continue

            instances = detect_cracks(im)
            scores = [inst["score"] for inst in instances]
            max_score = max(scores) * 100 if scores else 0

            if max_score > thresholdUpper:
                output_folder = ResultConfidentFolder
                confidence_type = "Confident Crack"
            elif max_score >= thresholdLower:
                output_folder = ResultPossibleFolder
                confidence_type = "Possible Crack"
            else:
                print(f"No cracks above {thresholdLower}% in {image_name} (max score {max_score:.1f}%)")
                continue

            annotated = draw_detections(im, instances, show_boxes=show_boxes)
            image_output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_output_path, annotated)
            print(f"Saved {confidence_type} result: {image_output_path}")

            mask_output = np.zeros_like(im)
            for inst in instances:
                mask_output[inst["mask"]] = 255
            mask_path = sidecar_path(image_output_path, "_mask")
            cv2.imwrite(mask_path, mask_output)

            gray = cv2.cvtColor(mask_output, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, 0)
            skeleton = skeletonize(thresh)
            cv2.imwrite(sidecar_path(image_output_path, "_length_estimation"), skeleton)

            date_taken = ""
            try:
                with Image.open(os.path.join(dir_path, image_name)) as img:
                    exif = img.getexif()
                    date_taken = exif.get(306, "")
            except Exception as e:
                print(f"Error reading metadata: {e}")

            class_names = sorted(set(inst["class"] for inst in instances))
            white_pixels = np.sum(mask_output == 255)
            total_pixels = mask_output.size
            coverage = round((white_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0
            length = int(np.sum(skeleton == 255))

            data_path = os.path.splitext(image_output_path)[0] + "_data.txt"
            with open(data_path, "w") as f:
                f.write(f"{image_name},{confidence_type},{date_taken},")
                f.write(f"\"{', '.join(class_names)}\",")
                f.write(f"{int(max_score)},{coverage},{length}")

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")

    worker.percentage = 100
    worker.finish()
    print("\nCrack Analysis Completed. You may generate a report.")


class InspectionCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._has_image = False
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#1a2533"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(360)
        self.setPlaceholder()

    def setPlaceholder(self, text=None):
        self._scene.clear()
        self._pixmap_item = None
        self._has_image = False
        self.resetTransform()
        message = text or "Select a folder and run crack analysis to inspect detections."
        text_item = self._scene.addText(message)
        text_item.setDefaultTextColor(QColor("#8ea0b5"))
        self.setSceneRect(text_item.boundingRect())
        self.fitInView(text_item, Qt.AspectRatioMode.KeepAspectRatio)

    def setImage(self, pixmap):
        self._scene.clear()
        self.resetTransform()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._has_image = True
        self.setSceneRect(self._pixmap_item.boundingRect())
        QTimer.singleShot(0, self.fitToView)

    def fitToView(self):
        if not self._has_image or self._pixmap_item is None:
            return
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self._has_image:
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            delta = event.pixelDelta().y()
        if delta == 0:
            return
        factor = 1.18 if delta > 0 else 1 / 1.18
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if self._has_image:
            self.fitToView()
        super().mouseDoubleClickEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        self.dir_path = ""
        self.console_out = ""
        self.thresholdUpper = 85
        self.thresholdLower = 60
        self.showBoundingBoxes = True
        super().__init__()
        # Variables for image processing
        self.imageFileList = []
        self.total_images = 0
        # Verification Process
        self.verificationImageFileList = []
        self.total_verification_images = 0
        self.current_image_id = 0
        self.verify_mode = ""
        # Methods of Main Class
        self.initUI()

    def consoleUpdate(self, message):
        self.console_out = str(message).strip()
        print(self.console_out)
        if hasattr(self, "Console"):
            self.Console.setText(self.console_out)
        if hasattr(self, "labelActivity"):
            self.labelActivity.setText(self.console_out)

    def countSourceImages(self):
        if not self.dir_path or not os.path.isdir(self.dir_path):
            return 0
        return len([
            name for name in os.listdir(self.dir_path)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def parseResultLine(self, line):
        try:
            row = next(csv.reader(io.StringIO(line.strip())))
            while len(row) < 7:
                row.append("")
            return {
                "filename": row[0],
                "confidence": row[1],
                "taken": row[2],
                "types": row[3],
                "score": row[4],
                "coverage": row[5],
                "length": row[6],
            }
        except Exception:
            return None

    def loadResultRecords(self):
        records = []
        if not self.dir_path:
            return records
        for mode in ("Confident", "Possible"):
            folder = os.path.join(self.dir_path, "Crack_Analysis", mode)
            if not os.path.exists(folder):
                continue
            for name in os.listdir(folder):
                if not name.endswith(".txt"):
                    continue
                path = os.path.join(folder, name)
                try:
                    with open(path, "r") as handle:
                        parsed = self.parseResultLine(handle.read())
                    if parsed:
                        records.append(parsed)
                except Exception:
                    continue
        return records

    def refreshDashboardStats(self):
        source_count = self.countSourceImages()
        possible_count = len(self.listResultImages("Possible")) if self.dir_path else 0
        confident_count = len(self.listResultImages("Confident")) if self.dir_path else 0
        self.labelSourceCount.setText(str(source_count))
        self.labelPossibleCount.setText(str(possible_count))
        self.labelConfidentCount.setText(str(confident_count))
        skipped = max(0, source_count - possible_count - confident_count)
        self.labelSkippedCount.setText(str(skipped) if source_count else "0")
        yolo_ready = os.path.exists(model_path(YOLO_WEIGHTS_NAME))
        self.labelDetector.setText("YOLOv8-crack-seg" if yolo_ready else "Mask R-CNN fallback")
        self.labelDetectorState.setText("Online" if yolo_ready else "Fallback")
        self.labelDetectorState.setObjectName("pill" if yolo_ready else "pillWarn")
        self.labelDetectorState.style().unpolish(self.labelDetectorState)
        self.labelDetectorState.style().polish(self.labelDetectorState)
        if self.dir_path:
            self.labelHelp.setText(self.dir_path)
        else:
            self.labelHelp.setText("No folder selected")
        self.populateResultsTable()

    def populateResultsTable(self):
        records = self.loadResultRecords()
        self.resultsTable.setRowCount(len(records))
        for row, rec in enumerate(records):
            values = [
                rec["filename"], rec["confidence"], rec["types"],
                rec["score"], rec["coverage"], rec["length"]
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.resultsTable.setItem(row, col, item)

    def updateCurrentCrackInfo(self, image_path=""):
        placeholder = (
            "No inspection selected.\nRun analysis, then browse Possible or Confident results."
        )
        if not image_path:
            self.labelCrackDetails.setText(placeholder)
            self.labelInspectionMeta.setText("Inspection idle")
            return
        data_path = os.path.splitext(image_path)[0] + "_data.txt"
        position = ""
        if self.total_verification_images:
            position = f"{self.current_image_id + 1} / {self.total_verification_images}  ·  {self.verify_mode}"
        self.labelInspectionMeta.setText(position or os.path.basename(image_path))
        parsed = None
        if os.path.exists(data_path):
            try:
                with open(data_path, "r") as handle:
                    parsed = self.parseResultLine(handle.read())
            except Exception:
                parsed = None
        if not parsed:
            self.labelCrackDetails.setText(
                f"File: {os.path.basename(image_path)}\nNo sidecar metrics found."
            )
            return
        self.labelCrackDetails.setText(
            f"File: {parsed['filename']}\n"
            f"Confidence: {parsed['confidence']}\n"
            f"Types: {parsed['types']}\n"
            f"Score: {parsed['score']}%\n"
            f"Coverage: {parsed['coverage']}%\n"
            f"Length: {parsed['length']} px\n"
            f"Captured: {parsed['taken'] or 'n/a'}"
        )

    def showPage(self, name):
        pages = {"dashboard": 0, "inspection": 1, "results": 2, "reports": 3}
        if name not in pages or not hasattr(self, "pageStack"):
            return
        self.pageStack.setCurrentIndex(pages[name])
        for key, button in self.navButtons.items():
            button.setObjectName("navActive" if key == name else "navItem")
            button.style().unpolish(button)
            button.style().polish(button)
        if name == "inspection":
            QTimer.singleShot(0, self.imageHolder.fitToView)

    def openResultFromTable(self, item):
        row = item.row()
        name_item = self.resultsTable.item(row, 0)
        conf_item = self.resultsTable.item(row, 1)
        if name_item is None:
            return
        filename = name_item.text()
        confidence = conf_item.text() if conf_item else ""
        mode = "Confident" if "Confident" in confidence else "Possible"
        self.showPage("inspection")
        if not self.showResultViewer(mode):
            return
        if filename in self.verificationImageFileList:
            self.current_image_id = self.verificationImageFileList.index(filename)
            self.showImage(os.path.join(
                self.dir_path, "Crack_Analysis", mode, filename))
            self.setInspectionButtons(
                True,
                has_prev=self.current_image_id > 0,
                has_next=self.current_image_id < self.total_verification_images - 1)

    def setInspectionButtons(self, enabled, has_prev=False, has_next=False):
        self.btnRemoveImage.setEnabled(enabled)
        self.btnZoomImage.setEnabled(enabled)
        self.btnPrevImage.setEnabled(enabled and has_prev)
        self.btnNextImage.setEnabled(enabled and has_next)

    def clearInspectionView(self):
        self.imageHolder.setPlaceholder()
        self.setInspectionButtons(False)
        self.updateCurrentCrackInfo()

    def changeThresholdUpper(self):
        if(self.thresholdUpper < self.thresholdLower):
            QMessageBox.critical(self, "Threshold Value Error.",
                                 "The Upper Threshold must be greater than Lower threshold and vice versa.")
            self.thresholdUpper = self.thresholdLower + 10
            self.thresholdSliderUpper.setValue(self.thresholdUpper)
            self.labelThresholdUpperValue.setText(str(self.thresholdUpper) + "%")
            self.consoleUpdate("Confidence Score Upper Threshold set at " +
                               str(self.thresholdUpper)+" %")
        else:
            self.thresholdUpper = self.sender().value()
            self.labelThresholdUpperValue.setText(str(self.thresholdUpper) + "%")
            self.consoleUpdate("Confidence Score Upper Threshold set at " +
                               str(self.thresholdUpper)+" %")

    def changeThresholdLower(self):
        if(self.thresholdUpper < self.thresholdLower):
            QMessageBox.critical(self, "Threshold Value Error.",
                                 "The Upper Threshold must be greater than Lower threshold and vice versa.")
            self.thresholdLower = self.thresholdUpper - 10
            self.thresholdSliderLower.setValue(self.thresholdLower)
            self.labelThresholdLowerValue.setText(str(self.thresholdLower) + "%")
            self.consoleUpdate("Confidence Score Lower Threshold set at " +
                               str(self.thresholdLower)+" %")
        else:
            self.thresholdLower = self.sender().value()
            self.labelThresholdLowerValue.setText(str(self.thresholdLower) + "%")
            self.consoleUpdate("Confidence Score Lower Threshold set at " +
                               str(self.thresholdLower)+" %")

    def CheckVerifyFolder(self):
        # This function checks whether there exists the analyzed images folder for verification
        analyzedFolderPath = os.path.join(self.dir_path, "Crack_Analysis")
        ResultPossibleFolder = os.path.join(
            analyzedFolderPath, "Possible")
        ResultConfidentFolder = os.path.join(
            analyzedFolderPath, "Confident")
        # Check whether the specified path exists or not
        ResultFolderExist = os.path.exists(analyzedFolderPath)
        ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
        ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
        if(ResultFolderExist and ResultPossibleFolderExists and ResultConfidentFolderExists):
            return True
        else:
            return False

    # Image Analysis Functions
    def PrevImage(self):
        if(self.current_image_id > 0):
            self.current_image_id -= 1
        if(self.total_verification_images > 0):
            self.showImage(os.path.join(
                self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id]))
            self.setInspectionButtons(
                True,
                has_prev=self.current_image_id > 0,
                has_next=self.current_image_id < self.total_verification_images - 1)
            self.consoleUpdate("Image "+str(self.current_image_id+1) +
                               " of "+str(self.total_verification_images))

    def NextImage(self):
        if(self.current_image_id < self.total_verification_images-1):
            self.current_image_id += 1
        if(self.total_verification_images > 0):
            self.showImage(os.path.join(
                self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id]))
            self.setInspectionButtons(
                True,
                has_prev=self.current_image_id > 0,
                has_next=self.current_image_id < self.total_verification_images - 1)
            self.consoleUpdate("Image "+str(self.current_image_id+1) +
                               " of "+str(self.total_verification_images))

    def RemoveImage(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirm removal")
        dlg.setText("Are you sure you want to remove this result?")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()
        if button == QMessageBox.StandardButton.Yes:
            if(len(self.verificationImageFileList) > 0):
                image_path = os.path.join(
                    self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id])
                try:
                    os.remove(image_path)
                except:
                    self.consoleUpdate("Error removing file "+image_path)
                image_output_path_modified = image_path
                image_output_path_modified = image_output_path_modified.replace(
                    '.png', '_mask.png')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpg', '_mask.jpg')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpeg', '_mask.jpeg')
                try:
                    os.remove(image_output_path_modified)
                except:
                    self.consoleUpdate("Error removing file" +
                                       image_output_path_modified)
                # Remove Text Data
                image_output_path_modified = image_path
                image_output_path_modified = image_output_path_modified.replace(
                    '.png', '_data.txt')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpg', '_data.txt')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpeg', '_data.txt')
                try:
                    os.remove(image_output_path_modified)
                except:
                    self.consoleUpdate("Error removing file" +
                                       image_output_path_modified)
                try:
                    self.verificationImageFileList.remove(
                        self.verificationImageFileList[self.current_image_id])
                    # print(self.verificationImageFileList)
                except:
                    self.consoleUpdate("Error processing removal.")
                self.total_verification_images = len(
                    self.verificationImageFileList)
                # Stop display if none left
                if(self.total_verification_images == 0):
                    self.clearInspectionView()
                self.refreshDashboardStats()
                self.consoleUpdate("Image Removed from Results")
                if(self.current_image_id > self.total_verification_images-1):
                    self.current_image_id = 0
                else:
                    self.NextImage()
        else:
            return

    def ZoomImage(self):
        try:
            self.imageHolder.fitToView()
        except Exception:
            QMessageBox.critical(self, "Error Opening Image.",
                                 "There was an error opening the image.")

    def showImage(self, image_path):
        try:
            os.environ['QT_IMAGEIO_MAXALLOC'] = str(
                os.path.getsize(image_path))
            self.pixmap = QPixmap(image_path)
            if self.pixmap.isNull():
                raise ValueError("Could not load image")
            self.imageHolder.setImage(self.pixmap)
            self.updateCurrentCrackInfo(image_path)
        except:
            QMessageBox.critical(self, "Error Opening Image.",
                                 "There was an error opening the image.")

    def listResultImages(self, mode):
        folder_dir = os.path.join(self.dir_path, "Crack_Analysis", mode)
        result_images = []
        if not os.path.exists(folder_dir):
            return result_images
        for images in os.listdir(folder_dir):
            if images.lower().endswith((".png", ".jpg", ".jpeg")):
                if "mask" not in images and "length_estimation" not in images:
                    result_images.append(images)
        return result_images

    def showResultViewer(self, mode):
        self.current_image_id = 0
        self.verify_mode = mode
        self.verificationImageFileList = self.listResultImages(mode)
        self.total_verification_images = len(self.verificationImageFileList)
        self.consoleUpdate(
            str(self.total_verification_images)+" images found for verification.")
        if self.total_verification_images > 0:
            self.showImage(os.path.join(
                self.dir_path, "Crack_Analysis", mode, self.verificationImageFileList[0]))
            self.setInspectionButtons(
                True,
                has_prev=False,
                has_next=self.total_verification_images > 1)
            return True
        self.clearInspectionView()
        return False

    def showAnalysisResults(self):
        self.consoleUpdate("Crack Analysis Completed. You may generate a report.")
        self.refreshDashboardStats()
        self.showPage("inspection")
        if self.dir_path == "" or not self.CheckVerifyFolder():
            return
        if self.showResultViewer("Confident"):
            self.consoleUpdate("Showing Confident Results")
            return
        if self.showResultViewer("Possible"):
            self.consoleUpdate("Showing Possible Results")
            return
        QMessageBox.information(
            self, "No detections.",
            "No cracks were detected above the current confidence thresholds. Try lowering the lower threshold and run analysis again.")

    def VerifyPossible(self):
        self.consoleUpdate("Verifying Possible Results")
        if(self.dir_path == ""):
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images of analysis.")
        elif not self.CheckVerifyFolder() or not self.showResultViewer("Possible"):
            QMessageBox.critical(self, "No Analyzed Results.",
                                 "The analyzed folders contain no Possible images. Try running Crack Analysis.")
        else:
            self.showPage("inspection")

    def VerifyConfident(self):
        self.consoleUpdate("Verifying Confident Results")
        if(self.dir_path == ""):
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images of analysis.")
        elif not self.CheckVerifyFolder() or not self.showResultViewer("Confident"):
            QMessageBox.critical(self, "No Analyzed Results.",
                                 "The analyzed folders contain no Confident images. Try running Crack Analysis.")
        else:
            self.showPage("inspection")

    def generateReport(self):
        try:
            text_file_paths = []
            # Create Result Directory
            analyzedFolderPath = os.path.join(self.dir_path, "Crack_Analysis")
            ResultPossibleFolder = os.path.join(
                analyzedFolderPath, "Possible")
            ResultConfidentFolder = os.path.join(
                analyzedFolderPath, "Confident")
            # Check whether the specified path exists or not
            ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
            ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
            # Get all the text files analyzed
            if ResultPossibleFolderExists:
                for result_text in os.listdir(ResultPossibleFolder):
                    # check file types and process
                    if (result_text.endswith(".txt")):
                        text_file_paths.append(os.path.join(
                            ResultPossibleFolder, result_text))
            if ResultConfidentFolderExists:
                for result_text in os.listdir(ResultConfidentFolder):
                    # check file types and process
                    if (result_text.endswith(".txt")):
                        text_file_paths.append(os.path.join(
                            ResultConfidentFolder, result_text))
            self.consoleUpdate("Generating Report")
            # Auto Filename
            now = datetime.now()
            date_time = now.strftime("%b_%d_%Y-%H_%M_%S")
            Report_Filename = "Crack_Analysis_Report_"+date_time+".csv"
            f = open(os.path.join(self.dir_path, Report_Filename), "w")
            output_data = ""
            for data in text_file_paths:
                data_file = open(data, "r")
                output_data += str(data_file.read()) + "\n"
                data_file.close()
            f.write("Filename"+","+"Confidence Type"+","+"Date/Time Taken"+"," +
                    "Crack Types"+","+"Maximum Confidence Score"+","+"Crack Coverage %"+","+"Total Crack Length (pixels)"+"\n"+output_data+"\nTotal Files:"+str(len(text_file_paths)))
            f.close()
            # Open Report Containing Folder
            show_in_file_manager(os.path.join(self.dir_path, Report_Filename))
            self.consoleUpdate("Report Successfully Generated")
        except Exception as e:
            QMessageBox.critical(self, "Error Generating Report.",
                                 "Something went wrong." + str(e))

    def openFolder(self):
        self.dir_path = QFileDialog.getExistingDirectory(
            self, "Choose Directory", "")
        if(self.dir_path == ""):
            # Throw warning when no folder is selected
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images for crack analysis.")
        else:
            self.labelHelp.setText(self.dir_path)
            self.consoleUpdate("Folder Selected at "+str(self.dir_path))
            self.clearInspectionView()
            self.refreshDashboardStats()

    def downloadPretrainedModel(self):
        os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
        yolo_path = model_path(YOLO_WEIGHTS_NAME)
        detectron_path = model_path(DETECTRON_WEIGHTS_NAME)

        self.consoleUpdate("Checking if pretrained models exist.")
        if not os.path.exists(yolo_path):
            self.consoleUpdate("Downloading YOLOv8-crack-seg model.")
            try:
                from huggingface_hub import hf_hub_download
                import shutil
                cached = hf_hub_download(repo_id=YOLO_HF_REPO, filename=YOLO_HF_FILE)
                shutil.copyfile(cached, yolo_path)
                self.consoleUpdate("YOLOv8-crack-seg model downloaded.")
            except Exception as e:
                self.consoleUpdate("Could not download YOLOv8 model: " + str(e))
        else:
            self.consoleUpdate("YOLOv8-crack-seg model exists.")

        if not os.path.exists(detectron_path):
            self.consoleUpdate("Mask R-CNN fallback model does not exist. Downloading.")
            url = "https://drive.google.com/uc?id=1V5biplCaJYHTxIp8achw0KKpm52qPLIa"
            gdown.download(url, detectron_path, quiet=False)
            self.consoleUpdate("Mask R-CNN model downloaded.")
        else:
            self.consoleUpdate("Mask R-CNN fallback model exists.")
        self.refreshDashboardStats()

    def runAnalysis(self):
        # Main Crack Detection Algorithm
        if(self.dir_path == ""):
            # Throw warning when no folder is selected
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images for crack analysis.")
        else:
            QMessageBox.information(self, "Running Analysis.",
                                    "Crack Analysis will begin and will take some time. Plese don't close any windows. Look at the terminal for progress.")
            # get the path/directory
            self.imageFileList = []
            folder_dir = self.dir_path
            for images in os.listdir(folder_dir):
                if images.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.imageFileList.append(images)
            # Report Image Number
            self.total_images = len(self.imageFileList)
            self.progress.setValue(0)
            self.refreshDashboardStats()
            self.consoleUpdate(str(self.total_images)+" images found in the folder " +
                               str(folder_dir))
            # Create Result Directory
            analyzedFolderPath = os.path.join(folder_dir, "Crack_Analysis")
            ResultPossibleFolder = os.path.join(
                analyzedFolderPath, "Possible")
            ResultConfidentFolder = os.path.join(
                analyzedFolderPath, "Confident")
            # Check whether the specified path exists or not
            ResultFolderExist = os.path.exists(analyzedFolderPath)
            ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
            ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
            # If not exists, create result directories
            if not ResultFolderExist:
                os.makedirs(analyzedFolderPath)
                self.consoleUpdate("Created folder "+str(analyzedFolderPath))
            if not ResultPossibleFolderExists:
                os.makedirs(ResultPossibleFolder)
                self.consoleUpdate("Created folder "+str(ResultPossibleFolder))
            if not ResultConfidentFolderExists:
                os.makedirs(ResultConfidentFolder)
                self.consoleUpdate("Created folder " +
                                   str(ResultConfidentFolder))
            self.analysis_worker = PercentageWorker(self)
            self.analysis_worker.percentageChanged.connect(self.progress.setValue)
            self.analysis_worker.finished.connect(self.showAnalysisResults)
            threading.Thread(
                target=analyseImage,
                args=("foo", self.dir_path, self.thresholdLower,
                      self.thresholdUpper, ResultPossibleFolder, ResultConfidentFolder),
                kwargs=dict(baz="baz", worker=self.analysis_worker,
                            show_boxes=self.chkShowBoundingBoxes.isChecked()),
                daemon=True,
            ).start()

    def initUI(self):
        self.setWindowTitle("Smart Crack Detection v.1.0")
        self.resize(1280, 820)
        self.setMinimumSize(1100, 720)
        self.setObjectName("dashboardRoot")
        self.setStyleSheet("""
            QWidget#dashboardRoot { background: #15202e; color: #d7e0ea; font-size: 13px; }
            QLabel { color: #d7e0ea; }
            QLabel#brand { font-size: 20px; font-weight: 700; letter-spacing: 1px; color: #ffffff; }
            QLabel#navActive { color: #3ddc84; font-weight: 700; padding: 8px 14px; }
            QLabel#navItem { color: #8ea0b5; padding: 8px 14px; }
            QLabel#cardTitle { color: #8ea0b5; font-size: 11px; font-weight: 700; letter-spacing: 1.4px; }
            QLabel#statCaption { color: #8ea0b5; font-size: 11px; }
            QLabel#statValue { font-size: 28px; font-weight: 700; color: #ffffff; }
            QLabel#statValueGood { font-size: 28px; font-weight: 700; color: #3ddc84; }
            QLabel#statValueWarn { font-size: 28px; font-weight: 700; color: #f0b429; }
            QLabel#statValueBad { font-size: 28px; font-weight: 700; color: #ff5c5c; }
            QLabel#pill { background: #1f3d32; color: #3ddc84; border-radius: 10px; padding: 4px 10px; }
            QLabel#pillWarn { background: #3d3220; color: #f0b429; border-radius: 10px; padding: 4px 10px; }
            QLabel#muted { color: #8ea0b5; }
            QFrame#card, QFrame#statTile {
                background: #1e2a3a;
                border: 1px solid #2c3b50;
                border-radius: 10px;
            }
            QPushButton#navItem, QPushButton#navActive {
                background: transparent;
                border: none;
                border-radius: 0;
                padding: 8px 14px;
                font-size: 14px;
            }
            QPushButton#navItem { color: #8ea0b5; font-weight: 500; }
            QPushButton#navItem:hover { color: #ffffff; background: transparent; }
            QPushButton#navActive {
                color: #3ddc84;
                font-weight: 700;
                border-bottom: 2px solid #3ddc84;
            }
            QPushButton#navActive:hover { color: #3ddc84; background: transparent; }
            QPushButton {
                background: #243447;
                color: #d7e0ea;
                border: 1px solid #3a4d66;
                border-radius: 6px;
                padding: 8px 12px;
            }
            QPushButton:hover { background: #2d4158; }
            QPushButton:disabled { color: #66788c; background: #1a2635; }
            QPushButton#primaryButton { background: #1f6b45; border: 1px solid #3ddc84; color: #ffffff; font-weight: 700; }
            QPushButton#primaryButton:hover { background: #258155; }
            QSlider::groove:horizontal { height: 6px; background: #2c3b50; border-radius: 3px; }
            QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; background: #3ddc84; border-radius: 7px; }
            QProgressBar { background: #2c3b50; border: none; border-radius: 4px; color: #d7e0ea; text-align: center; height: 16px; }
            QProgressBar::chunk { background: #3ddc84; border-radius: 4px; }
            QCheckBox { color: #d7e0ea; }
            QTableWidget {
                background: #1a2533;
                alternate-background-color: #1e2c3d;
                color: #d7e0ea;
                gridline-color: #2c3b50;
                border: none;
            }
            QHeaderView::section { background: #243447; color: #8ea0b5; border: none; padding: 6px; }
            QGraphicsView { background: #1a2533; border: none; }
            QScrollBar::handle:vertical { background: #3a4d66; border-radius: 4px; }
        """)

        def make_card(title):
            card = QFrame()
            card.setObjectName("card")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(10)
            heading = QLabel(title.upper())
            heading.setObjectName("cardTitle")
            layout.addWidget(heading)
            return card, layout

        header = QFrame()
        header.setObjectName("headerBar")
        header.setFixedHeight(64)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 8, 18, 8)
        brand = QLabel("SMART CRACK DETECTION")
        brand.setObjectName("brand")
        self.navButtons = {}
        for key, title in (
            ("dashboard", "Dashboard"),
            ("inspection", "Inspection"),
            ("results", "Results"),
            ("reports", "Reports"),
        ):
            button = QPushButton(title)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setObjectName("navActive" if key == "dashboard" else "navItem")
            button.clicked.connect(lambda _checked=False, page=key: self.showPage(page))
            self.navButtons[key] = button
        self.labelDetector = QLabel("YOLOv8-crack-seg")
        self.labelDetector.setObjectName("muted")
        self.labelDetectorState = QLabel("Online")
        self.labelDetectorState.setObjectName("pill")
        header_layout.addWidget(brand)
        header_layout.addSpacing(24)
        header_layout.addWidget(self.navButtons["dashboard"])
        header_layout.addWidget(self.navButtons["inspection"])
        header_layout.addWidget(self.navButtons["results"])
        header_layout.addWidget(self.navButtons["reports"])
        header_layout.addStretch()
        header_layout.addWidget(self.labelDetector)
        header_layout.addWidget(self.labelDetectorState)

        self.labelHelp = QLabel("No folder selected")
        self.labelHelp.setObjectName("muted")
        self.labelHelp.setWordWrap(True)
        self.labelThresholdLower = QLabel("Lower confidence")
        self.labelThresholdUpper = QLabel("Upper confidence")
        self.labelThresholdLowerValue = QLabel(str(self.thresholdLower) + "%")
        self.labelThresholdUpperValue = QLabel(str(self.thresholdUpper) + "%")
        self.btnSelectFolder = QPushButton("Select image folder")
        self.btnRunAnalysis = QPushButton("Run crack analysis")
        self.btnRunAnalysis.setObjectName("primaryButton")
        self.btnVerifyConfidentResults = QPushButton("Verify confident")
        self.btnVerifyPossibleResults = QPushButton("Verify possible")
        self.btnOpenResults = QPushButton("Generate report")
        self.thresholdSliderUpper = QSlider(Qt.Orientation.Horizontal)
        self.thresholdSliderUpper.setMinimum(0)
        self.thresholdSliderUpper.setMaximum(100)
        self.thresholdSliderUpper.setValue(self.thresholdUpper)
        self.thresholdSliderLower = QSlider(Qt.Orientation.Horizontal)
        self.thresholdSliderLower.setMinimum(0)
        self.thresholdSliderLower.setMaximum(100)
        self.thresholdSliderLower.setValue(self.thresholdLower)
        self.chkShowBoundingBoxes = QCheckBox("Show bounding boxes")
        self.chkShowBoundingBoxes.setChecked(self.showBoundingBoxes)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.Console = QLabel("Ready")
        self.Console.setObjectName("muted")
        self.Console.setWordWrap(True)
        self.labelActivity = self.Console
        self.btnPrevImage = QPushButton("Previous")
        self.btnRemoveImage = QPushButton("Remove")
        self.btnZoomImage = QPushButton("Fit")
        self.btnNextImage = QPushButton("Next")
        self.imageHolder = InspectionCanvas()

        self.labelSourceCount = QLabel("0")
        self.labelSourceCount.setObjectName("statValue")
        self.labelPossibleCount = QLabel("0")
        self.labelPossibleCount.setObjectName("statValueWarn")
        self.labelConfidentCount = QLabel("0")
        self.labelConfidentCount.setObjectName("statValueGood")
        self.labelSkippedCount = QLabel("0")
        self.labelSkippedCount.setObjectName("statValueBad")
        self.labelInspectionMeta = QLabel("Inspection idle")
        self.labelInspectionMeta.setObjectName("muted")
        self.labelCrackDetails = QLabel(
            "No inspection selected.\nRun analysis, then browse Possible or Confident results.")
        self.labelCrackDetails.setWordWrap(True)
        self.labelLegend = QLabel(
            "Diagonal  ·  deep red\nHorizontal  ·  burnt orange\nVertical  ·  forest green")
        self.labelLegend.setObjectName("muted")

        self.resultsTable = QTableWidget(0, 6)
        self.resultsTable.setHorizontalHeaderLabels(
            ["File", "Confidence", "Types", "Score", "Coverage %", "Length px"])
        self.resultsTable.horizontalHeader().setStretchLastSection(True)
        self.resultsTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.resultsTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.resultsTable.setAlternatingRowColors(True)
        self.resultsTable.verticalHeader().setVisible(False)

        self.labelReportsHelp = QLabel(
            "Generate a CSV of every detected crack: filename, confidence type, "
            "capture time, crack types, score, coverage, and length in pixels.\n\n"
            "The report is saved in the selected image folder and opened in File Explorer."
        )
        self.labelReportsHelp.setWordWrap(True)
        self.labelReportsHelp.setObjectName("muted")
        self.btnOpenResults.setText("Generate CSV report")

        setup_card, setup_layout = make_card("Inspection setup")
        setup_layout.addWidget(QLabel("Source folder"))
        setup_layout.addWidget(self.labelHelp)
        setup_layout.addWidget(self.btnSelectFolder)
        lower_row = QHBoxLayout()
        lower_row.addWidget(self.labelThresholdLower)
        lower_row.addStretch()
        lower_row.addWidget(self.labelThresholdLowerValue)
        setup_layout.addLayout(lower_row)
        setup_layout.addWidget(self.thresholdSliderLower)
        upper_row = QHBoxLayout()
        upper_row.addWidget(self.labelThresholdUpper)
        upper_row.addStretch()
        upper_row.addWidget(self.labelThresholdUpperValue)
        setup_layout.addLayout(upper_row)
        setup_layout.addWidget(self.thresholdSliderUpper)
        setup_layout.addWidget(self.chkShowBoundingBoxes)
        setup_layout.addWidget(self.btnRunAnalysis)
        setup_layout.addWidget(self.progress)
        setup_layout.addStretch()

        stats_card, stats_layout = make_card("Detection status")
        tiles = QHBoxLayout()
        def add_tile(caption, value_label):
            tile = QFrame()
            tile.setObjectName("statTile")
            tile_layout = QVBoxLayout(tile)
            cap = QLabel(caption)
            cap.setObjectName("statCaption")
            tile_layout.addWidget(cap)
            tile_layout.addWidget(value_label)
            tiles.addWidget(tile)
        add_tile("SOURCE", self.labelSourceCount)
        add_tile("POSSIBLE", self.labelPossibleCount)
        add_tile("CONFIDENT", self.labelConfidentCount)
        add_tile("BELOW CUTOFF", self.labelSkippedCount)
        stats_layout.addLayout(tiles)

        inspect_card, inspect_layout = make_card("Live inspection")
        inspect_layout.addWidget(self.labelInspectionMeta)
        nav_row = QHBoxLayout()
        nav_row.addWidget(self.btnPrevImage)
        nav_row.addWidget(self.btnRemoveImage)
        nav_row.addWidget(self.btnZoomImage)
        nav_row.addWidget(self.btnNextImage)
        inspect_layout.addLayout(nav_row)
        inspect_layout.addWidget(self.imageHolder, 1)

        detail_card, detail_layout = make_card("Selected crack")
        detail_layout.addWidget(self.labelCrackDetails)
        detail_layout.addWidget(QLabel("Overlay legend"))
        detail_layout.addWidget(self.labelLegend)
        detail_layout.addWidget(self.btnVerifyConfidentResults)
        detail_layout.addWidget(self.btnVerifyPossibleResults)
        detail_layout.addStretch()

        table_card, table_layout = make_card("Aggregated detections")
        table_layout.addWidget(self.resultsTable, 1)
        table_hint = QLabel("Double-click a row to open that image in Inspection.")
        table_hint.setObjectName("muted")
        table_layout.addWidget(table_hint)

        reports_card, reports_layout = make_card("Reports")
        reports_layout.addWidget(self.labelReportsHelp)
        reports_layout.addWidget(self.btnOpenResults)
        reports_layout.addWidget(self.Console)
        reports_layout.addStretch()

        dashboard_page = QWidget()
        dash_layout = QHBoxLayout(dashboard_page)
        dash_layout.setContentsMargins(0, 0, 0, 0)
        dash_left = QVBoxLayout()
        dash_left.addWidget(setup_card, 1)
        dash_right = QVBoxLayout()
        dash_right.addWidget(stats_card)
        dash_right.addStretch()
        dash_layout.addLayout(dash_left, 2)
        dash_layout.addLayout(dash_right, 5)

        inspection_page = QWidget()
        inspect_page_layout = QHBoxLayout(inspection_page)
        inspect_page_layout.setContentsMargins(0, 0, 0, 0)
        inspect_page_layout.addWidget(inspect_card, 5)
        inspect_page_layout.addWidget(detail_card, 2)

        results_page = QWidget()
        results_page_layout = QVBoxLayout(results_page)
        results_page_layout.setContentsMargins(0, 0, 0, 0)
        results_page_layout.addWidget(table_card)

        reports_page = QWidget()
        reports_page_layout = QVBoxLayout(reports_page)
        reports_page_layout.setContentsMargins(0, 0, 0, 0)
        reports_page_layout.addWidget(reports_card)

        self.pageStack = QStackedWidget()
        self.pageStack.addWidget(dashboard_page)
        self.pageStack.addWidget(inspection_page)
        self.pageStack.addWidget(results_page)
        self.pageStack.addWidget(reports_page)

        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(header)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.addWidget(self.pageStack, 1)
        root.addWidget(content, 1)
        self.setLayout(root)

        self.thresholdSliderUpper.valueChanged.connect(self.changeThresholdUpper)
        self.thresholdSliderLower.valueChanged.connect(self.changeThresholdLower)
        self.btnSelectFolder.clicked.connect(self.openFolder)
        self.btnRunAnalysis.clicked.connect(self.runAnalysis)
        self.btnVerifyPossibleResults.clicked.connect(self.VerifyPossible)
        self.btnVerifyConfidentResults.clicked.connect(self.VerifyConfident)
        self.btnOpenResults.clicked.connect(self.generateReport)
        self.btnPrevImage.clicked.connect(self.PrevImage)
        self.btnRemoveImage.clicked.connect(self.RemoveImage)
        self.btnZoomImage.clicked.connect(self.ZoomImage)
        self.btnNextImage.clicked.connect(self.NextImage)
        self.resultsTable.itemDoubleClicked.connect(self.openResultFromTable)
        self.setInspectionButtons(False)
        self.showPage("dashboard")
        self.show()
        self.downloadPretrainedModel()


if __name__ == "__main__":
    qApp = QApplication(sys.argv)
    qApp.setStyle("Fusion")
    w = MainWindow()
    sys.exit(qApp.exec())

