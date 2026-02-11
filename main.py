#!/usr/bin/env python3

import sys
import os
import shutil
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

try:
    import face_recognition
except ImportError:
    face_recognition = None

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import cv2

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    insightface = None
    FaceAnalysis = None

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QGridLayout,
    QLineEdit, QMessageBox, QProgressBar, QComboBox, QCheckBox,
    QGroupBox, QInputDialog, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QEvent, QTimer
from PyQt6.QtGui import QPixmap, QIcon


FACE_CROP_SIZE = 128


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sorted_clusters(clusters: dict):
    """Return (cluster_id, photos) sorted by unique photo count, descending."""
    items = [(cid, photos) for cid, photos in clusters.items()
             if not str(cid).startswith('unknown_')]
    return sorted(items,
                  key=lambda x: len(set(p for p, _ in x[1])),
                  reverse=True)


def _unique_dest(dest_folder: str, filename: str) -> str:
    """Return a collision-free destination path."""
    dest = os.path.join(dest_folder, filename)
    if not os.path.exists(dest):
        return dest
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(dest):
        dest = os.path.join(dest_folder, f"{base}_{counter}{ext}")
        counter += 1
    return dest


def _safe_transfer(src: str, dst: str, is_move: bool):
    """
    Move or copy src to dst.
    Returns None on success, error string on failure.
    Missing source files are silently skipped (return None).
    """
    if not os.path.exists(src):
        return None   # already moved/deleted – skip silently
    try:
        if is_move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
        return None
    except Exception as e:
        return f"{os.path.basename(src)}: {e}"


def _do_transfer(unique_photos, dest_folder, is_move):
    """Transfer a list of photos; returns (moved_count, errors_list)."""
    os.makedirs(dest_folder, exist_ok=True)
    moved, errors = 0, []
    for photo in unique_photos:
        if not os.path.exists(photo):
            continue   # silently skip missing files
        if os.path.abspath(os.path.dirname(photo)) == os.path.abspath(dest_folder):
            continue   # already in destination
        dst = _unique_dest(dest_folder, os.path.basename(photo))
        err = _safe_transfer(photo, dst, is_move)
        if err:
            errors.append(err)
        else:
            moved += 1
    return moved, errors


# ─────────────────────────────────────────────────────────────────────────────
# Background analysis thread
# ─────────────────────────────────────────────────────────────────────────────
class FaceAnalyzer(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)

    def __init__(self, photo_dir, max_dimension=800,
                 recursive=True, use_insightface=True, confidence_threshold=0.5):
        super().__init__()
        self.photo_dir            = photo_dir
        self.image_extensions     = {'.jpg', '.jpeg', '.png', '.bmp',
                                      '.gif', '.tiff', '.webp'}
        self.max_dimension        = max_dimension
        self.recursive            = recursive
        self.use_insightface      = use_insightface
        self.confidence_threshold = confidence_threshold
        self._is_running          = True
        self._is_paused           = False

        self.face_app = None
        if use_insightface:
            if insightface is not None:
                try:
                    print("Initialising InsightFace buffalo_l …")
                    self.face_app = FaceAnalysis(name='buffalo_l')
                    try:
                        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                        print("InsightFace: GPU")
                    except Exception:
                        self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
                        print("InsightFace: CPU")
                except Exception as e:
                    print(f"InsightFace init failed: {e}")
                    self.use_insightface = False
            else:
                print("InsightFace not installed – falling back to face_recognition")
                self.use_insightface = False

    def pause(self):  self._is_paused = True
    def resume(self): self._is_paused = False
    def stop(self):   self._is_running = False

    def _wait_if_paused(self, pct, msg):
        while self._is_paused and self._is_running:
            self.progress.emit(pct, f"PAUSED – {msg}")
            self.msleep(100)

    def _find_photos(self):
        files = []
        if self.recursive:
            for root, _, fnames in os.walk(self.photo_dir):
                for f in fnames:
                    if Path(f).suffix.lower() in self.image_extensions:
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(self.photo_dir):
                fp = os.path.join(self.photo_dir, f)
                if os.path.isfile(fp) and Path(f).suffix.lower() in self.image_extensions:
                    files.append(fp)
        return files

    def _dbscan(self, encodings, eps, min_samples=2):
        arr = np.array(encodings)
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='euclidean').fit_predict(arr)
        n_clust = sum(1 for l in labels if l != -1)
        n_noise = sum(1 for l in labels if l == -1)
        print(f"DBSCAN eps={eps}: {n_clust} clustered, {n_noise} noise")
        return labels

    def _build_clusters(self, labels, results):
        clusters = {}
        for face_idx, label in enumerate(labels):
            if label == -1:
                label = f"unknown_{face_idx}"
            clusters.setdefault(label, [])
            for photo_path, face_list in results['photo_faces'].items():
                if face_idx in face_list:
                    clusters[label].append((photo_path, face_idx))
                    break
        results['clusters'] = clusters


    def _resize_if_needed(self, img):
        h, w = img.shape[:2]
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img

    # ── main run ──────────────────────────────────────────────────────────
    def run(self):
        results = {'photos': [], 'face_encodings': [],
                   'photo_faces': {}, 'clusters': {}}

        photo_files = self._find_photos()
        total = len(photo_files)
        self.progress.emit(0, f"Found {total} photos")



        # ── face recognition path ─────────────────────────────────────────
        for idx, photo_path in enumerate(photo_files):
            if not self._is_running:
                return
            self._wait_if_paused(int(idx / total * 100), f"{idx}/{total}")
            if not self._is_running:
                return
            self.progress.emit(int(idx / total * 100),
                f"Analysing {os.path.basename(photo_path)} ({idx+1}/{total})")
            try:
                img = cv2.imread(photo_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                if max(h, w) > self.max_dimension:
                    from PIL import Image as PILImage
                    scale = self.max_dimension / max(h, w)
                    pil = PILImage.fromarray(img_rgb).resize(
                        (int(w * scale), int(h * scale)),
                        PILImage.Resampling.LANCZOS)
                    img_rgb = np.array(pil)

                encodings = []
                if self.use_insightface and self.face_app is not None:
                    for face in self.face_app.get(img_rgb):
                        if face.det_score >= self.confidence_threshold:
                            encodings.append(face.embedding)
                elif face_recognition is not None:
                    locs = face_recognition.face_locations(img_rgb)
                    if locs:
                        encodings = face_recognition.face_encodings(img_rgb, locs)

                if encodings:
                    results['photos'].append(photo_path)
                    idxs = []
                    for enc in encodings:
                        fi = len(results['face_encodings'])
                        results['face_encodings'].append(enc)
                        idxs.append(fi)
                    results['photo_faces'][photo_path] = idxs

            except Exception as e:
                print(f"Face recog error {photo_path}: {e}")

        if results['face_encodings'] and self._is_running:
            self.progress.emit(100, "Clustering faces…")
            eps = 18.0 if (self.use_insightface and self.face_app) else 0.5
            labels = self._dbscan(results['face_encodings'], eps=eps)
            self._build_clusters(labels, results)

        self.progress.emit(100, "Analysis complete!")
        self.finished.emit(results)


# ─────────────────────────────────────────────────────────────────────────────
# Person thumbnail widget
# ─────────────────────────────────────────────────────────────────────────────
class PersonThumbnailWidget(QWidget):
    def __init__(self, cluster_id, photos, person_name=None, size=120):
        super().__init__()
        self.cluster_id  = cluster_id
        self.photos      = photos
        self.person_name = person_name
        self.size        = size
        self.is_selected = False

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)

        self.thumbnail_container = QWidget()
        self._apply_style(selected=False)

        thumb_layout = QVBoxLayout(self.thumbnail_container)
        thumb_layout.setContentsMargins(3, 3, 3, 3)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: none;")
        thumb_layout.addWidget(self.image_label)

        display_name = person_name if person_name else f"Person {cluster_id}"
        self.name_label = QLabel(display_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setMaximumWidth(size)

        unique = list(set(p for p, _ in photos))
        count_label = QLabel(f"{len(unique)} photos")
        count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_label.setStyleSheet("color: #666; font-size: 9px;")

        layout.addWidget(self.thumbnail_container)
        layout.addWidget(self.name_label)
        layout.addWidget(count_label)
        self.setLayout(layout)
        self._load_thumbnail()

        self.thumbnail_container.mousePressEvent       = self.on_mouse_press
        self.thumbnail_container.mouseDoubleClickEvent = self.on_double_click
        self.thumbnail_container.setMinimumWidth(0)
        self.thumbnail_container.setMinimumHeight(0)

    def _apply_style(self, selected):
        if selected:
            self.thumbnail_container.setStyleSheet("""
                QWidget { border: 3px solid #0078d4; border-radius: 6px;
                          background-color: #e8f4fd; }
                QWidget:hover { border-color: #005a9e;
                                background-color: #d4e9fc; }
            """)
        else:
            self.thumbnail_container.setStyleSheet("""
                QWidget { border: 2px solid #ccc; border-radius: 6px;
                          background-color: #f5f5f5; }
                QWidget:hover { border-color: #0078d4;
                                background-color: #e8f4fd; }
            """)

    def _load_thumbnail(self):
        if not self.photos:
            self.image_label.setText("No photos")
            return
        # Try each photo until we find one that exists
        for photo_path, _ in self.photos:
            if not os.path.exists(photo_path):
                continue
            try:
                px = QPixmap(photo_path)
                if not px.isNull():
                    self.image_label.setPixmap(px.scaled(
                        self.size - 16, self.size - 16,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                    return
            except Exception:
                continue
        self.image_label.setText("No preview")

    def on_mouse_press(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier:
                self.is_selected = not self.is_selected
                self._apply_style(self.is_selected)
                parent = self.parent()
                while parent and not hasattr(parent, 'on_thumbnail_selection_changed'):
                    parent = parent.parent()
                if parent:
                    parent.on_thumbnail_selection_changed(
                        self.cluster_id, self.is_selected)

    def on_double_click(self, event):
        parent = self.parent()
        while parent and not hasattr(parent, 'on_person_double_clicked'):
            parent = parent.parent()
        if parent:
            parent.on_person_double_clicked(self.cluster_id)

    def update_name(self, new_name):
        self.person_name = new_name
        try:
            self.name_label.setText(
                new_name if new_name else f"Person {self.cluster_id}")
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Photo thumbnail widget
# ─────────────────────────────────────────────────────────────────────────────
class ThumbnailWidget(QWidget):
    def __init__(self, photo_path, size=150):
        super().__init__()
        self.photo_path = photo_path
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.setCursor(Qt.CursorShape.PointingHandCursor)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(size, size)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setCursor(Qt.CursorShape.PointingHandCursor)

        if os.path.exists(photo_path):
            try:
                px = QPixmap(photo_path)
                self.image_label.setPixmap(px.scaled(
                    size, size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            except Exception:
                self.image_label.setText("Error")
        else:
            self.image_label.setText("⚠ Missing")
            self.image_label.setStyleSheet("color:red;")
            self.checkbox.setEnabled(False)

        name_label = QLabel(os.path.basename(photo_path))
        name_label.setWordWrap(True)
        name_label.setMaximumWidth(size)

        layout.addWidget(self.checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        layout.addWidget(name_label)
        self.setLayout(layout)

        # Make clicking the image toggle the checkbox
        self.image_label.mousePressEvent = self.on_image_click

    def on_image_click(self, event):
        """Toggle checkbox when clicking on the image."""
        if self.checkbox.isEnabled():
            self.checkbox.setChecked(not self.checkbox.isChecked())
            # Notify parent widget about selection change
            parent = self.parent()
            while parent and not hasattr(parent, 'on_thumbnail_selection_changed'):
                parent = parent.parent()
            if parent:
                parent.on_thumbnail_selection_changed(self.photo_path, self.checkbox.isChecked())


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class PhotoOrganizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.photo_dir              = None
        self.results                = None
        self.person_labels          = {}
        self.current_cluster        = None
        self.person_widgets         = {}
        self.view_mode              = "people"
        self.selected_ids_for_merge = set()
        self._resize_timer          = QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._handle_resize)
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("MantaScan")
        self.setGeometry(100, 100, 1200, 800)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        # ── top bar ───────────────────────────────────────────────────────
        top = QHBoxLayout()
        self.select_folder_btn = QPushButton("Select Photo Folder")
        self.select_folder_btn.clicked.connect(self.select_folder)
        top.addWidget(self.select_folder_btn)

        self.folder_label = QLabel("No folder selected")
        top.addWidget(self.folder_label, 1)

        self.include_subfolders = QCheckBox("Include subfolders")
        self.include_subfolders.setChecked(True)
        top.addWidget(self.include_subfolders)

        # mode group
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout()




        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Face confidence:"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 90)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_value_label = QLabel("0.50")
        self.confidence_slider.valueChanged.connect(
            lambda v: self.confidence_value_label.setText(f"{v/100:.2f}"))
        conf_row.addWidget(self.confidence_slider)
        conf_row.addWidget(self.confidence_value_label)
        mode_layout.addLayout(conf_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Max image size:"))
        self.resize_combo = QComboBox()
        self.resize_combo.addItems(["400px (fastest)", "800px (balanced)",
                                    "1600px (accurate)", "No resize (slowest)"])
        self.resize_combo.setCurrentIndex(1)
        size_row.addWidget(self.resize_combo)
        mode_layout.addLayout(size_row)

        mode_group.setLayout(mode_layout)
        top.addWidget(mode_group)

        self.analyze_btn = QPushButton("Analyse Photos")
        self.analyze_btn.clicked.connect(self.analyze_photos)
        self.analyze_btn.setEnabled(False)
        top.addWidget(self.analyze_btn)
        root_layout.addLayout(top)

        # progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        root_layout.addWidget(self.progress_bar)

        pc_row = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setVisible(False)
        pc_row.addWidget(self.pause_btn)

        self.cancel_btn = QPushButton("Stop & Process")
        self.cancel_btn.clicked.connect(self.cancel_analysis)
        self.cancel_btn.setVisible(False)
        pc_row.addWidget(self.cancel_btn)
        pc_row.addStretch()
        root_layout.addLayout(pc_row)

        self.status_label = QLabel("")
        root_layout.addWidget(self.status_label)

        # ── main content ──────────────────────────────────────────────────
        content = QWidget()
        content_layout = QHBoxLayout(content)

        # sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(300)
        sl = QVBoxLayout(sidebar)

        # label group
        name_grp = QGroupBox("Label Person")
        name_lay = QVBoxLayout()
        self.person_name_input = QLineEdit()
        self.person_name_input.setPlaceholderText("Enter person's name…")
        name_lay.addWidget(self.person_name_input)
        save_btn = QPushButton("Save Name")
        save_btn.clicked.connect(self.save_person_name)
        name_lay.addWidget(save_btn)
        name_grp.setLayout(name_lay)
        sl.addWidget(name_grp)

        # merge group
        merge_grp = QGroupBox("Merge People")
        merge_lay = QVBoxLayout()
        merge_lay.addWidget(QLabel("Click or Ctrl+Click to select, then:"))
        self.merge_btn = QPushButton("Merge / Rename Selected People")
        self.merge_btn.clicked.connect(self.merge_clusters)
        merge_lay.addWidget(self.merge_btn)
        merge_grp.setLayout(merge_lay)
        sl.addWidget(merge_grp)

        # clustering
        clust_grp = QGroupBox("Clustering Sensitivity")
        clust_lay = QVBoxLayout()
        clust_lay.addWidget(QLabel("Adjust if people are split/merged incorrectly:"))

        strict_row = QHBoxLayout()
        strict_row.addWidget(QLabel("Strict (17)"))
        strict_row.addStretch()
        strict_row.addWidget(QLabel("Loose (22)"))
        clust_lay.addLayout(strict_row)
        
        lbl_row = QHBoxLayout()
        lbl_row.addWidget(QLabel("Strictness:"))
        self.cluster_slider = QSlider(Qt.Orientation.Horizontal)
        self.cluster_slider.setRange(170, 220)
        self.cluster_slider.setValue(200)
        self.cluster_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cluster_slider.setTickInterval(10)
        self.cluster_value_label = QLabel("20.0")
        self.cluster_slider.valueChanged.connect(
            lambda v: self.cluster_value_label.setText(f"{v/10:.1f}"))
        lbl_row.addWidget(self.cluster_slider)
        lbl_row.addWidget(self.cluster_value_label)
        clust_lay.addLayout(lbl_row)

        self.mode_label = QLabel("Mode: Unknown")
        self.mode_label.setStyleSheet("color:#666;font-size:11px;")
        clust_lay.addWidget(self.mode_label)

        ms_row = QHBoxLayout()
        ms_row.addWidget(QLabel("Min faces per person:"))
        self.min_samples_combo = QComboBox()
        self.min_samples_combo.addItems(["1 (most sensitive)", "2 (default)",
                                         "3 (stricter)", "4 (very strict)"])
        self.min_samples_combo.setCurrentIndex(1)
        ms_row.addWidget(self.min_samples_combo)
        clust_lay.addLayout(ms_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        for lbl, val in [("Very Loose", 205), ("Medium", 190), ("Very Strict", 180)]:
            btn = QPushButton(lbl)
            btn.clicked.connect(lambda _, v=val: self.cluster_slider.setValue(v))
            preset_row.addWidget(btn)
        clust_lay.addLayout(preset_row)

        self.recluster_btn = QPushButton("Re-cluster with New Setting")
        self.recluster_btn.clicked.connect(self.recluster)
        self.recluster_btn.setEnabled(False)
        clust_lay.addWidget(self.recluster_btn)
        clust_grp.setLayout(clust_lay)
        sl.addWidget(clust_grp)



        # bulk organise
        bulk_grp = QGroupBox("Bulk Organise")
        bulk_lay = QVBoxLayout()
        self.bulk_org_btn = QPushButton("Organise All People to Subfolders")
        self.bulk_org_btn.clicked.connect(self.bulk_organize)
        self.bulk_org_btn.setEnabled(False)
        bulk_lay.addWidget(self.bulk_org_btn)
        bulk_grp.setLayout(bulk_lay)
        sl.addWidget(bulk_grp)

        sl.addStretch()
        content_layout.addWidget(sidebar)

        # main area
        main_area = QWidget()
        mal = QVBoxLayout(main_area)

        nav_row = QHBoxLayout()
        self.back_btn = QPushButton("← Back to People")
        self.back_btn.clicked.connect(self.show_people_overview)
        self.back_btn.setVisible(False)
        nav_row.addWidget(self.back_btn)
        self.people_title = QLabel("Detected People:")
        self.people_title.setStyleSheet("font-size:16px;font-weight:bold;")
        nav_row.addWidget(self.people_title)
        nav_row.addStretch()
        mal.addLayout(nav_row)

        self.org_controls_widget = QWidget()
        self.org_controls_widget.setVisible(False)
        org_row = QHBoxLayout(self.org_controls_widget)
        org_row.addWidget(QLabel("Organise selected photos:"))
        self.org_action = QComboBox()
        self.org_action.addItems(["Move to subfolder", "Copy to subfolder"])
        org_row.addWidget(self.org_action)
        self.organize_btn = QPushButton("Organise Photos")
        self.organize_btn.clicked.connect(self.organize_photos)
        self.organize_btn.setEnabled(False)
        org_row.addWidget(self.organize_btn)
        org_row.addStretch()
        mal.addWidget(self.org_controls_widget)

        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.main_widget = QWidget()
        self.main_widget.setMinimumWidth(0)
        self.main_layout = QGridLayout(self.main_widget)
        self.main_layout.setHorizontalSpacing(10)
        self.main_layout.setVerticalSpacing(10)
        self.main_scroll.setWidget(self.main_widget)
        mal.addWidget(self.main_scroll)

        content_layout.addWidget(main_area)
        root_layout.addWidget(content, 1)

    # ── folder / analysis ─────────────────────────────────────────────────
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Photo Folder")
        if folder:
            self.photo_dir = folder
            self.folder_label.setText(f"Selected: {folder}")
            self.analyze_btn.setEnabled(True)

    def analyze_photos(self):
        if not self.photo_dir:
            return
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.pause_btn.setVisible(True)
        self.pause_btn.setText("Pause")
        self.cancel_btn.setVisible(True)
        self._clear_grid()

        max_dimension   = {0: 400, 1: 800, 2: 1600}.get(
            self.resize_combo.currentIndex(), 10000)

        self.analyzer = FaceAnalyzer(
            self.photo_dir,
            max_dimension=max_dimension,
            recursive=self.include_subfolders.isChecked(),
            use_insightface=True,
            confidence_threshold=self.confidence_slider.value() / 100.0,
        )
        self.analyzer.progress.connect(self.on_progress)
        self.analyzer.finished.connect(self.on_analysis_complete)
        self.analyzer.start()

    def toggle_pause(self):
        if hasattr(self, 'analyzer') and self.analyzer.isRunning():
            if self.analyzer._is_paused:
                self.analyzer.resume(); self.pause_btn.setText("Pause")
            else:
                self.analyzer.pause();  self.pause_btn.setText("Resume")

    def cancel_analysis(self):
        if hasattr(self, 'analyzer') and self.analyzer.isRunning():
            reply = QMessageBox.question(self, "Stop Analysis",
                "Stop and process photos analysed so far?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.status_label.setText("Stopping…")
                self.analyzer.stop()

    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_analysis_complete(self, results):
        self.results = results
        self.progress_bar.setVisible(False)
        self.pause_btn.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.bulk_org_btn.setEnabled(True)
        self.recluster_btn.setEnabled(True)

        if results['face_encodings']:
            dim = len(results['face_encodings'][0])
            if dim == 512:
                self.mode_label.setText("Mode: InsightFace")
            else:
                self.mode_label.setText(f"Mode: Legacy ({dim}d)")

        self._clear_grid()
        nc = sum(1 for c in results['clusters'] if not str(c).startswith('unknown_'))
        nu = sum(1 for c in results['clusters'] if     str(c).startswith('unknown_'))
        self.status_label.setText(
            f"Found {nc} groups of people, {nu} single/unknown faces")

        self.show_people_overview()

    # ── grid helpers ──────────────────────────────────────────────────────
    def _clear_grid(self):
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    # ── selection / navigation ────────────────────────────────────────────
    def on_thumbnail_selection_changed(self, cluster_id, is_selected):
        if is_selected:
            self.selected_ids_for_merge.add(cluster_id)
        else:
            self.selected_ids_for_merge.discard(cluster_id)

    def on_person_double_clicked(self, cluster_id):
        self.show_person_photos(cluster_id)

    def show_person_photos(self, cluster_id):
        self.current_cluster = cluster_id
        self.view_mode = "photos"
        self.back_btn.setVisible(True)
        self.people_title.setText(
            f"Photos for "
            f"{self.person_labels.get(cluster_id, f'Person {cluster_id}')}:")
        self.org_controls_widget.setVisible(True)
        self.person_name_input.setText(self.person_labels.get(cluster_id, ""))
        self._clear_grid()

        # preserve order, deduplicate
        seen = set()
        unique = []
        for p, _ in self.results['clusters'][cluster_id]:
            if p not in seen:
                seen.add(p); unique.append(p)

        # Calculate optimal columns for photo view
        width = self.main_scroll.viewport().width()
        thumbnail_width = 150
        spacing = self.main_layout.horizontalSpacing()
        base_size = thumbnail_width + spacing * 2
        cols = max(2, int(width / base_size))

        for idx, photo_path in enumerate(unique):
            self.main_layout.addWidget(
                ThumbnailWidget(photo_path), idx // cols, idx % cols)
        self.organize_btn.setEnabled(True)

    def show_people_overview(self):
        self.view_mode = "people"
        self.current_cluster = None
        self.back_btn.setVisible(False)
        self.people_title.setText("Detected People:")
        self.org_controls_widget.setVisible(False)
        self._clear_grid()
        self.person_widgets.clear()

        if not self.results:
            return

        # Calculate optimal columns based on current width
        width = self.main_scroll.viewport().width()
        thumbnail_width = 150  # Size used in PersonThumbnailWidget
        spacing = self.main_layout.horizontalSpacing()
        base_size = thumbnail_width + spacing * 2
        cols = max(2, int(width / base_size))

        # sorted largest → smallest
        row = col = 0
        for cluster_id, photos in _sorted_clusters(self.results['clusters']):
            w = PersonThumbnailWidget(
                cluster_id, photos, self.person_labels.get(cluster_id), size=150)
            self.person_widgets[cluster_id] = w
            self.main_layout.addWidget(w, row, col)
            col += 1
            if col >= cols:
                col = 0; row += 1

        self.organize_btn.setEnabled(False)

    # ── re-clustering ─────────────────────────────────────────────────────
    def recluster(self):
        if not self.results or not self.results['face_encodings']:
            return

        min_samples = [1, 2, 3, 4][self.min_samples_combo.currentIndex()]
        enc = self.results['face_encodings']
        dim = len(enc[0])

        if dim == 512:
            eps = max(10.0, self.cluster_slider.value() / 10.0)
            self.mode_label.setText(f"Mode: InsightFace (eps={eps:.1f})")
        else:
            eps = self.cluster_slider.value() / 10.0 / 50.0
            self.mode_label.setText(f"Mode: Legacy (eps={eps:.2f})")

        self.status_label.setText(f"Re-clustering with eps={eps:.3f}…")
        arr = np.array(enc)
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='euclidean').fit_predict(arr)

        new_clusters = {}
        for face_idx, label in enumerate(labels):
            if label == -1:
                label = f"unknown_{face_idx}"
            new_clusters.setdefault(label, [])
            for photo_path, face_list in self.results['photo_faces'].items():
                if face_idx in face_list:
                    new_clusters[label].append((photo_path, face_idx))
                    break

        self.results['clusters'] = new_clusters
        self.person_labels.clear()
        self.person_widgets.clear()
        self.show_people_overview()

        nc = sum(1 for c in new_clusters if not str(c).startswith('unknown_'))
        nu = sum(1 for c in new_clusters if     str(c).startswith('unknown_'))
        self.status_label.setText(
            f"Re-clustered! {nc} groups, {nu} singles (eps={eps:.3f})")

    # ── people management ─────────────────────────────────────────────────
    def merge_clusters(self):
        sel = list(self.selected_ids_for_merge)
        if len(sel) == 0:
            QMessageBox.warning(self, "No Selection",
                "Click to select a person to rename.")
            return
        name, ok = QInputDialog.getText(
            self, "Rename Person",
            f"Enter new name for {len(sel)} group(s):",
            text=self.person_labels.get(sel[0] if sel else sel[0], ""))
        if not ok or not name.strip():
            return
        name = name.strip()
        if len(sel) == 1:
            # Just rename the single person
            cid = sel[0]
            self.person_labels[cid] = name
            w = self.person_widgets.get(cid)
            if w:
                w.update_name(name)
            self.selected_ids_for_merge.clear()
            self.show_people_overview()
            QMessageBox.information(self, "Renamed",
                f"Renamed to '{name}'")
        else:
            # Merge multiple people
            primary = sel[0]
            for cid in sel[1:]:
                self.results['clusters'][primary].extend(
                    self.results['clusters'].pop(cid))
                self.person_labels.pop(cid, None)
            self.person_labels[primary] = name
            self.selected_ids_for_merge.clear()
            self.person_widgets.clear()
            self.show_people_overview()
            QMessageBox.information(self, "Merged",
                f"Merged {len(sel)} groups into '{name}'")

    def save_person_name(self):
        if self.current_cluster is None:
            return
        name = self.person_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "No Name", "Please enter a name.")
            return
        self.person_labels[self.current_cluster] = name
        w = self.person_widgets.get(self.current_cluster)
        if w:
            w.update_name(name)
        QMessageBox.information(self, "Saved", f"Named as '{name}'")

    # ── organise ──────────────────────────────────────────────────────────
    def bulk_organize(self):
        if not self.results or not self.results['clusters']:
            return
        if not self.person_labels:
            reply = QMessageBox.question(self, "No Names",
                "No people labelled. Use generic names like 'Person_0'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return

        reply = QMessageBox.question(self, "Move or Copy?",
            "Move (removes from original) or Copy?",
            QMessageBox.StandardButton.Yes  |
            QMessageBox.StandardButton.No   |
            QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Cancel:
            return
        is_move = (reply == QMessageBox.StandardButton.Yes)
        word    = "moved" if is_move else "copied"

        total_moved, all_errors = 0, []
        for cluster_id, photos in self.results['clusters'].items():
            if str(cluster_id).startswith('unknown_'):
                continue
            person_name = self.person_labels.get(cluster_id, f"Person_{cluster_id}")
            safe_name = "".join(c for c in person_name
                                if c.isalnum() or c in (' ', '_', '-')).strip()
            dest = os.path.join(self.photo_dir, safe_name)

            seen = set(); unique = []
            for p, _ in photos:
                if p not in seen:
                    seen.add(p); unique.append(p)

            moved, errs = _do_transfer(unique, dest, is_move)
            total_moved += moved
            all_errors  += errs

        msg = f"Successfully {word} {total_moved} photos to subfolders!"
        if all_errors:
            msg += f"\n\n{len(all_errors)} errors:\n" + "\n".join(all_errors[:5])
            if len(all_errors) > 5:
                msg += f"\n… and {len(all_errors)-5} more"
        QMessageBox.information(self, "Bulk Organise Complete", msg)

    def organize_photos(self):
        if self.current_cluster is None:
            return
        person_name = self.person_labels.get(
            self.current_cluster, f"Person_{self.current_cluster}")
        safe_name = "".join(c for c in person_name
                            if c.isalnum() or c in (' ', '_', '-')).strip()
        dest_folder = os.path.join(self.photo_dir, safe_name)

        selected = []
        for i in range(self.main_layout.count()):
            w = self.main_layout.itemAt(i).widget()
            if isinstance(w, ThumbnailWidget) and w.checkbox.isChecked():
                selected.append(w.photo_path)
        if not selected:
            QMessageBox.warning(self, "No Selection", "No photos selected.")
            return

        is_move = self.org_action.currentText().startswith("Move")
        word    = "moved" if is_move else "copied"
        moved, errors = _do_transfer(selected, dest_folder, is_move)

        msg = f"{moved} photos {word} to '{dest_folder}'"
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        QMessageBox.information(self, "Done", msg)

    # ── resize handling ────────────────────────────────────────────────────
    def resizeEvent(self, event):
        """Debounced resize handler to avoid excessive recalculations."""
        super().resizeEvent(event)
        # Restart the timer - it will only fire after resizing stops
        self._resize_timer.start(150)  # 150ms delay

    def _handle_resize(self):
        """Actually handle the resize after debounce period."""
        if self.view_mode == "people":
            self._update_people_grid()
        elif self.view_mode == "photos":
            self._update_photos_grid()

    def _update_people_grid(self):
        """Recalculate and rebuild people grid with optimal column count."""
        if not self.person_widgets or not self.results:
            return

        # Get actual available width (accounting for scrollbar)
        width = self.main_scroll.viewport().width()
        thumbnail_width = 150  # Match size in show_people_overview
        spacing = self.main_layout.horizontalSpacing()
        base_size = thumbnail_width + spacing * 2

        # Calculate optimal columns
        cols = max(2, int(width / base_size))

        # Only rebuild if column count changed
        if hasattr(self, '_current_people_cols') and self._current_people_cols == cols:
            return

        self._current_people_cols = cols

        # Rebuild grid with new column count
        sorted_items = _sorted_clusters(self.results['clusters'])
        row = col = 0
        
        for cluster_id, photos in sorted_items:
            w = self.person_widgets.get(cluster_id)
            if w:
                # Remove from current position
                self.main_layout.removeWidget(w)
                # Add to new position
                self.main_layout.addWidget(w, row, col)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1

    def _update_photos_grid(self):
        """Recalculate and rebuild photos grid with optimal column count."""
        if self.current_cluster is None or not self.results:
            return

        # Get actual available width
        width = self.main_scroll.viewport().width()
        thumbnail_width = 150
        spacing = self.main_layout.horizontalSpacing()
        base_size = thumbnail_width + spacing * 2

        # Calculate optimal columns
        cols = max(2, int(width / base_size))

        # Only rebuild if column count changed
        if hasattr(self, '_current_photo_cols') and self._current_photo_cols == cols:
            return

        self._current_photo_cols = cols

        # Clear and rebuild
        self._clear_grid()

        # Get unique photos
        seen = set()
        unique = []
        for p, _ in self.results['clusters'][self.current_cluster]:
            if p not in seen:
                seen.add(p)
                unique.append(p)

        # Rebuild with new column count
        for idx, photo_path in enumerate(unique):
            self.main_layout.addWidget(
                ThumbnailWidget(photo_path), idx // cols, idx % cols)


def main():
    app = QApplication(sys.argv)
    window = PhotoOrganizerApp()
    window.setWindowIcon(QIcon('icon.ico'))
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
