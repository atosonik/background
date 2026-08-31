"""PyQt5 face-swap desktop app. Fully offline once models/ is populated."""
from __future__ import annotations

import os
import sys

# swapper MUST be imported before PyQt5: it loads onnxruntime, whose DLL fails
# to initialise on Windows if Qt has already been loaded. Do not reorder.
import swapper
import segment

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

ACCENT = "#4c8dff"
BG = "#16181d"
PANEL = "#1e2128"


class Task(QtCore.QThread):
    """Runs any callable off the GUI thread so the window never freezes."""

    ok = QtCore.pyqtSignal(object)
    err = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.ok.emit(self._fn())
        except Exception as exc:
            self.err.emit(str(exc))


class Canvas(QtWidgets.QWidget):
    """Displays a BGR image scaled to fit, and reports clicks in image pixels."""

    clicked = QtCore.pyqtSignal(int, int)

    def __init__(self, placeholder, clickable=False):
        super().__init__()
        self._qimg = None
        self._shown = QtCore.QRect()
        self._placeholder = placeholder
        self._clickable = clickable
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        if clickable:
            self.setCursor(QtCore.Qt.PointingHandCursor)

    def set_image(self, bgr):
        if bgr is None:
            self._qimg = None
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            # .copy() because QImage does not take ownership of the numpy buffer.
            self._qimg = QtGui.QImage(
                rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888
            ).copy()
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setPen(QtGui.QPen(QtGui.QColor("#2c313a")))
        p.setBrush(QtGui.QColor("#101216"))
        p.drawRoundedRect(r, 10, 10)

        if self._qimg is None:
            p.setPen(QtGui.QColor("#5a6270"))
            p.drawText(r, QtCore.Qt.AlignCenter, self._placeholder)
            return

        scaled = self._qimg.scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._shown = QtCore.QRect(x, y, scaled.width(), scaled.height())
        p.drawImage(x, y, scaled)

    def mousePressEvent(self, ev):
        if not self._clickable or self._qimg is None:
            return
        if not self._shown.contains(ev.pos()):
            return
        # Map widget coords back through the fit-scale into original pixel coords.
        fx = self._qimg.width() / self._shown.width()
        fy = self._qimg.height() / self._shown.height()
        self.clicked.emit(
            int((ev.x() - self._shown.x()) * fx), int((ev.y() - self._shown.y()) * fy)
        )


HELP_STEPS = [
    ("Open the photo",
     "Click <b>Open photo…</b>, or drag a picture straight onto the window. "
     "Every face that is found gets a numbered bracket drawn around it."),
    ("Click the face you want replaced",
     "Click it directly in the left pane. Its brackets turn blue so you can see "
     "which one is selected. Clicking another face moves the selection."),
    ("Choose the new face",
     "Click <b>Open new face…</b> and pick a photo of the person going in. "
     "If that photo contains several people, the largest face is used."),
    ("Press Swap face",
     "About a second on this machine. The result appears on the right. Your "
     "original file is never modified."),
    ("Fine-tune, then save",
     "<b>Skin tone match</b> and <b>Swap strength</b> update the preview instantly "
     "— they re-blend the finished swap rather than running it again. When it "
     "looks right, press <b>Save result…</b>."),
]

HELP_BG_STEPS = [
    ("Open the photo",
     "The same photo pane as the other tab. Nothing needs clicking here — "
     "<i>everyone</i> in the picture is kept."),
    ("Choose a new background",
     "Click <b>Open background…</b> and pick any image. It is scaled to cover "
     "the frame and centre-cropped, never stretched, so a portrait backdrop "
     "behind a landscape photo will not lean."),
    ("Press Replace background",
     "The people are separated from the scene once, then held. That first pass "
     "takes about half a second per person."),
    ("Tune the edge, then save",
     "<b>Edge softness</b> feathers the cut-out. It re-composites from the "
     "stored separation rather than running the model again, so it stays quick "
     "on a large photo. Raise it if the outline looks cut with scissors; lower "
     "it if hair looks smeared."),
]

HELP_TIPS = [
    "<b>Match the pose.</b> A front-on source photo onto a face that is turned or "
    "tilted is where this fails hardest. It matters more than any slider here.",
    "<b>Keep Skin tone match high (70–100%).</b> It pulls the new face into the "
    "photo's own lighting so it doesn't read lighter than everyone else.",
    "The face is rebuilt at 128×128 before being placed back, so on a large "
    "photo it can look slightly softer than the rest of the frame.",
    "Everything runs on this computer. No photo is ever uploaded anywhere.",
]


class HelpDialog(QtWidgets.QDialog):
    """Numbered walkthrough shown by the Help button (or F1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How to use Face Swap")
        self.setModal(True)
        self.resize(580, 700)

        inner = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(inner)
        col.setContentsMargins(26, 24, 26, 24)
        col.setSpacing(0)

        title = QtWidgets.QLabel("How to use Face Swap")
        title.setObjectName("helpTitle")
        col.addWidget(title)
        lede = QtWidgets.QLabel(
            "Two things this does: put someone else's face on a person in the "
            "photo, or keep everyone and change what is behind them."
        )
        lede.setObjectName("helpLede")
        lede.setWordWrap(True)
        col.addWidget(lede)
        col.addSpacing(18)

        for heading, steps in (("SWAP A FACE", HELP_STEPS),
                               ("REPLACE THE BACKGROUND", HELP_BG_STEPS)):
            lb = QtWidgets.QLabel(heading)
            lb.setObjectName("section")
            col.addWidget(lb)
            col.addSpacing(12)
            for n, (head, body) in enumerate(steps, 1):
                col.addLayout(self._step(n, head, body))
                col.addSpacing(16)
            col.addSpacing(6)

        rule = QtWidgets.QFrame()
        rule.setFrameShape(QtWidgets.QFrame.HLine)
        rule.setObjectName("rule")
        col.addSpacing(4)
        col.addWidget(rule)
        col.addSpacing(16)

        tips = QtWidgets.QLabel("GETTING A GOOD RESULT")
        tips.setObjectName("section")
        col.addWidget(tips)
        col.addSpacing(10)
        for tip in HELP_TIPS:
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(10)
            dot = QtWidgets.QLabel("•")
            dot.setObjectName("bullet")
            dot.setAlignment(QtCore.Qt.AlignTop)
            body = QtWidgets.QLabel(tip)
            body.setObjectName("helpBody")
            body.setWordWrap(True)
            row.addWidget(dot)
            row.addWidget(body, 1)
            col.addLayout(row)
            col.addSpacing(10)
        col.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        close = QtWidgets.QPushButton("Got it")
        close.setObjectName("primary")
        close.setFixedHeight(40)
        close.clicked.connect(self.accept)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)
        outer.addWidget(scroll, 1)
        outer.addWidget(close)
        self.setStyleSheet(STYLE)

    @staticmethod
    def _step(n, head, body):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(14)
        badge = QtWidgets.QLabel(str(n))
        badge.setObjectName("badge")
        badge.setFixedSize(28, 28)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        # Keep the badge pinned to the first line of a wrapped paragraph.
        holder = QtWidgets.QVBoxLayout()
        holder.setContentsMargins(0, 2, 0, 0)
        holder.addWidget(badge)
        holder.addStretch(1)

        text = QtWidgets.QVBoxLayout()
        text.setSpacing(3)
        h = QtWidgets.QLabel(head)
        h.setObjectName("helpHead")
        b = QtWidgets.QLabel(body)
        b.setObjectName("helpBody")
        b.setWordWrap(True)
        text.addWidget(h)
        text.addWidget(b)

        row.addLayout(holder)
        row.addLayout(text, 1)
        return row


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Swap")
        self.resize(1360, 840)
        self.setAcceptDrops(True)

        self.target = None      # original photo, BGR
        self.source = None      # replacement face photo, BGR
        self.faces = []
        self.selected = -1
        self.raw = None         # cached network output
        self.mask = None
        self.region = None      # touched box, so slider redraws stay local
        self.background = None  # replacement backdrop, BGR
        self.fg_alpha = None    # cached people-matte, so the edge slider is cheap

        # Coalesces a burst of slider ticks into one recomposite.
        self._slider_settle = QtCore.QTimer(self)
        self._slider_settle.setSingleShot(True)
        self._slider_settle.setInterval(120)
        self._slider_settle.timeout.connect(self.refresh_active)
        self.result = None
        self._tasks = set()

        self.view_in = Canvas("Open a photo, or drop one here", clickable=True)
        self.view_in.clicked.connect(self.on_click_face)
        self.view_out = Canvas("The swapped photo appears here")

        views = QtWidgets.QSplitter()
        for view, title in (
            (self.view_in, "PHOTO  ·  click the face to replace"),
            (self.view_out, "RESULT"),
        ):
            box = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(box)
            lay.setContentsMargins(0, 0, 0, 0)
            cap = QtWidgets.QLabel(title)
            cap.setObjectName("caption")
            cap.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                              QtWidgets.QSizePolicy.Fixed)
            if view is self.view_in:
                self.cap_in = cap
            lay.addWidget(cap)
            lay.addWidget(view, 1)
            views.addWidget(box)
        views.setSizes([680, 680])

        root = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(root)
        row.setContentsMargins(14, 14, 14, 14)
        row.setSpacing(14)
        row.addWidget(views, 1)
        row.addWidget(self._sidebar())
        self.setCentralWidget(root)

        self.status = self.statusBar()
        self.setStyleSheet(STYLE)
        QtWidgets.QShortcut(QtGui.QKeySequence("F1"), self, self.show_help)
        self._boot()

    def _sidebar(self):
        panel = QtWidgets.QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(300)
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(10)

        def header(text):
            lb = QtWidgets.QLabel(text)
            lb.setObjectName("section")
            return lb

        # Sits above step 1, outlined in the accent colour so it reads as the
        # obvious starting point for anyone opening the app for the first time.
        self.btn_help = QtWidgets.QPushButton("  ?     How to use this app")
        self.btn_help.setObjectName("help")
        self.btn_help.setFixedHeight(40)
        self.btn_help.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_help.setToolTip("Step-by-step guide  (F1)")
        self.btn_help.clicked.connect(self.show_help)
        lay.addWidget(self.btn_help)
        lay.addSpacing(10)

        lay.addWidget(header("1  ·  PHOTO"))
        self.btn_target = QtWidgets.QPushButton("Open photo…")
        self.btn_target.clicked.connect(self.open_target)
        lay.addWidget(self.btn_target)
        self.lbl_faces = QtWidgets.QLabel("No photo loaded")
        self.lbl_faces.setObjectName("hint")
        self.lbl_faces.setWordWrap(True)
        lay.addWidget(self.lbl_faces)

        lay.addSpacing(8)
        # The photo, the result pane and Save are shared; only the middle of the
        # sidebar changes with the mode, so it is a tab strip rather than two
        # separate screens the user would have to reload their photo into.
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._tab_face(header), "Face")
        self.tabs.addTab(self._tab_background(header), "Background")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self.tabs, 1)

        self.btn_action = QtWidgets.QPushButton("Swap face")
        self.btn_action.setObjectName("primary")
        self.btn_action.setFixedHeight(44)
        self.btn_action.clicked.connect(self.do_action)
        self.btn_save = QtWidgets.QPushButton("Save result…")
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setEnabled(False)
        lay.addWidget(self.btn_action)
        lay.addWidget(self.btn_save)
        return panel

    def _tab_face(self, header):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 14, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(header("2  ·  NEW FACE"))
        self.thumb = Canvas("no face chosen")
        self.thumb.setMinimumSize(0, 0)
        self.thumb.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                 QtWidgets.QSizePolicy.Fixed)
        self.thumb.setFixedHeight(140)
        lay.addWidget(self.thumb)
        self.btn_source = QtWidgets.QPushButton("Open new face…")
        self.btn_source.clicked.connect(self.open_source)
        lay.addWidget(self.btn_source)

        lay.addSpacing(4)
        lay.addWidget(header("3  ·  ADJUST"))
        self.sl_tone, tone_row = self._slider("Skin tone match", 70)
        self.sl_blend, blend_row = self._slider("Swap strength", 100)
        lay.addLayout(tone_row)
        lay.addLayout(blend_row)
        tip = QtWidgets.QLabel(
            "Tone match pulls the new face toward the photo's own lighting. Keep "
            "it high (70–100) so the swapped face doesn't read lighter than "
            "everyone else."
        )
        tip.setObjectName("hint")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        lay.addStretch(1)
        return page

    def _tab_background(self, header):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 14, 0, 0)
        lay.setSpacing(10)

        lay.addWidget(header("2  ·  NEW BACKGROUND"))
        self.bg_thumb = Canvas("no background chosen")
        self.bg_thumb.setMinimumSize(0, 0)
        self.bg_thumb.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                    QtWidgets.QSizePolicy.Fixed)
        self.bg_thumb.setFixedHeight(140)
        lay.addWidget(self.bg_thumb)
        self.btn_bg = QtWidgets.QPushButton("Open background…")
        self.btn_bg.clicked.connect(self.open_background)
        lay.addWidget(self.btn_bg)

        lay.addSpacing(4)
        lay.addWidget(header("3  ·  ADJUST"))
        self.cmb_quality = QtWidgets.QComboBox()
        # (label, segment model key). Fast is the default because the accurate
        # model is ~17x slower on CPU; on a CUDA build that gap mostly closes.
        self._quality = [("Fast", "human"), ("Best hair detail (slower)", "birefnet_lite")]
        for label, key in self._quality:
            self.cmb_quality.addItem(label if segment.available().get(key)
                                     else f"{label} - not downloaded")
        self.cmb_quality.currentIndexChanged.connect(self._on_quality_changed)
        lay.addWidget(self.cmb_quality)
        self.sl_feather, feather_row = self._slider("Edge softness", 3, 0, 25, "px")
        lay.addLayout(feather_row)
        self.lbl_bg = QtWidgets.QLabel(
            "Everyone in the photo is kept and everything behind them is "
            "replaced. The new background is scaled to cover the frame, never "
            "stretched."
        )
        self.lbl_bg.setObjectName("hint")
        self.lbl_bg.setWordWrap(True)
        lay.addWidget(self.lbl_bg)
        lay.addStretch(1)
        return page

    def _slider(self, name, value, lo=0, hi=100, suffix="%"):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        val = QtWidgets.QLabel(f"{value}{suffix}")
        val.setObjectName("hint")
        head.addWidget(QtWidgets.QLabel(name))
        head.addStretch(1)
        head.addWidget(val)
        sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(value)
        sl.valueChanged.connect(lambda v: val.setText(f"{v}{suffix}"))
        # The number updates on every tick, but the recomposite waits until the
        # drag settles. Even restricted to the face region that costs ~200 ms on
        # a 10 MP photo, and running it per tick makes the slider feel stuck.
        sl.valueChanged.connect(self._slider_settle.start)
        col.addLayout(head)
        col.addWidget(sl)
        return sl, col

    def show_help(self):
        HelpDialog(self).exec_()

    # ---------- mode dispatch ----------

    def _on_tab_changed(self, index):
        face = index == 0
        self.btn_action.setText("Swap face" if face else "Replace background")
        # "click the face to replace" is wrong in background mode -- nothing is
        # clicked there, and leaving it up makes the app look broken.
        self.cap_in.setText("PHOTO  ·  click the face to replace" if face
                            else "PHOTO  ·  everyone in it is kept")
        self.refresh_active()

    def do_action(self):
        (self.do_swap if self.tabs.currentIndex() == 0 else self.do_background)()

    def refresh_active(self):
        """Recompose whichever mode is showing, from cached model output."""
        if self.tabs.currentIndex() == 0:
            self.refresh_result()
        else:
            self.refresh_background()

    # ---------- background replacement ----------

    def _on_quality_changed(self, _index):
        # The cached matte belongs to the previous model, so drop it and make
        # the user press the button again rather than showing a stale cut-out.
        self.fg_alpha = None
        self.status.showMessage("Press Replace background to re-cut at this quality")

    def matte_model(self):
        return self._quality[self.cmb_quality.currentIndex()][1]

    def open_background(self):
        path = self._pick("Open background")
        if not path:
            return
        img = swapper.imread(path)
        if img is None:
            self._fail(f"Could not read {path}")
            return
        self.background = img
        self.bg_thumb.set_image(img)
        self.status.showMessage(f"Background loaded ({img.shape[1]}x{img.shape[0]})")
        # If the people-matte is already computed, show the new backdrop at once.
        self.refresh_background()

    def do_background(self):
        if self.target is None:
            self.status.showMessage("Open a photo first")
            return
        if self.background is None:
            self.status.showMessage("Open a background image")
            return
        if self.fg_alpha is not None:          # matte already cached for this photo
            self.refresh_background()
            return

        self.btn_action.setEnabled(False)
        slow = self.matte_model() != "human"
        self.status.showMessage("Separating the people from the background…"
                                + (" (accurate model, this takes longer)" if slow else ""))
        tgt, faces = self.target, self.faces
        model = self.matte_model()
        self._run(lambda: segment.foreground_alpha(tgt, faces, model),
                  self._alpha_ready, self._bg_failed)

    def _alpha_ready(self, alpha):
        self.fg_alpha = alpha
        self.btn_action.setEnabled(True)
        self.refresh_background()
        self.status.showMessage("Done — drag Edge softness to tune the cut-out")

    def _bg_failed(self, msg):
        self.btn_action.setEnabled(True)
        self._fail(msg)

    def refresh_background(self):
        """Composite from the cached matte. No model runs, so the edge slider is
        live even on a large photo."""
        if self.fg_alpha is None or self.background is None or self.target is None:
            return
        self.result = segment.compose_over(
            self.target, self.fg_alpha, self.background,
            feather=self.sl_feather.value())
        self.view_out.set_image(self.result)
        self.btn_save.setEnabled(True)

    # ---------- model loading ----------

    def _boot(self):
        gone = swapper.missing_models()
        if gone:
            self.lbl_faces.setText("Models missing.")
            self.status.showMessage("Missing: " + ", ".join(gone))
            self.btn_action.setEnabled(False)
            QtWidgets.QMessageBox.warning(
                self, "Models missing",
                "These model files are not present:\n\n  " + "\n  ".join(gone)
                + "\n\nRun this once on a connected machine:\n\n"
                  "    python download_models.py",
            )
            return
        self.status.showMessage("Loading models…")
        self.btn_action.setEnabled(False)
        self._run(
            swapper.load, self._booted,
            lambda e: self.status.showMessage(f"Model load failed: {e}"),
        )

    def _booted(self, _):
        prov = swapper.providers()[0].replace("ExecutionProvider", "")
        self.status.showMessage(f"Ready  ·  {prov}  ·  offline")
        self.btn_action.setEnabled(True)

    def _run(self, fn, ok, err=None):
        # Every live QThread must stay referenced from Python or it can be
        # collected mid-run and take the process down. Operations do overlap --
        # picking a source face while target detection is still going -- so this
        # is a set, not a single slot, and entries are dropped only on finish.
        task = Task(fn, self)
        self._tasks.add(task)
        task.finished.connect(lambda t=task: self._tasks.discard(t))
        task.ok.connect(ok)
        task.err.connect(err or self._fail)
        task.start()

    def _fail(self, msg):
        self.status.showMessage(msg)
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    # ---------- actions ----------

    def _pick(self, caption):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, caption, "", "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
        )
        return path

    def open_target(self):
        path = self._pick("Open photo")
        if path:
            self.load_target(path)

    def load_target(self, path):
        img = swapper.imread(path)
        if img is None:
            self._fail(f"Could not read {path}")
            return
        self.target, self.faces, self.selected = img, [], -1
        self.raw = self.result = self.region = None
        self.fg_alpha = None          # matte belongs to the old photo
        self.btn_save.setEnabled(False)
        self.view_in.set_image(img)
        self.view_out.set_image(None)
        self.status.showMessage("Detecting faces…")
        self._run(lambda: swapper.detect(img), self._detected)

    def _detected(self, faces):
        self.faces = faces
        if not faces:
            self.lbl_faces.setText("No faces found. Try a larger or sharper photo.")
            self.status.showMessage("No faces detected")
            return
        self.view_in.set_image(swapper.draw_overlay(self.target, faces))
        self.lbl_faces.setText(f"{len(faces)} face(s) found — click one in the photo.")
        self.status.showMessage(f"{len(faces)} face(s) detected")

    def on_click_face(self, x, y):
        hit = swapper.face_at_point(self.faces, x, y)
        if hit is None:
            return
        self.selected = hit.index
        self.view_in.set_image(swapper.draw_overlay(self.target, self.faces, self.selected))
        self.lbl_faces.setText(f"Face {hit.index + 1} selected.")
        self.status.showMessage(f"Face {hit.index + 1} selected")

    def open_source(self):
        path = self._pick("Open new face")
        if not path:
            return
        img = swapper.imread(path)
        if img is None:
            self._fail(f"Could not read {path}")
            return
        self.source = img
        self.thumb.set_image(img)
        self.status.showMessage("Checking the new face…")
        self._run(
            lambda: swapper.pick_source_face(img),
            lambda f: self.status.showMessage("New face ready"),
            lambda e: self.status.showMessage(f"⚠ {e}"),
        )

    def do_swap(self):
        if self.target is None:
            self.status.showMessage("Open a photo first")
            return
        if self.selected < 0:
            self.status.showMessage("Click the face you want to replace")
            return
        if self.source is None:
            self.status.showMessage("Open the new person's photo")
            return

        self.btn_action.setEnabled(False)
        self.status.showMessage("Swapping…")
        face, tgt, src = self.faces[self.selected], self.target, self.source
        self._run(
            lambda: swapper.swap_identity(tgt, face, src),
            self._swapped,
            self._swap_failed,
        )

    def _swap_failed(self, msg):
        self.btn_action.setEnabled(True)
        self._fail(msg)

    def _swapped(self, out):
        self.raw, self.mask, self.region = out
        self.btn_action.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.refresh_result()
        self.status.showMessage("Done — drag the sliders to fine-tune")

    def refresh_result(self):
        """Re-composite from the cached network output. No forward pass, so this is
        fast enough to run live on every slider tick."""
        if self.raw is None:
            return
        self.result = swapper.finish(
            self.raw, self.target, self.mask, self.region,
            self.sl_tone.value() / 100.0, self.sl_blend.value() / 100.0,
        )
        self.view_out.set_image(self.result)

    def save(self):
        if self.result is None:
            return
        default = os.path.join(os.getcwd(), "outputs", "swapped.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save result", default, "PNG (*.png);;JPEG (*.jpg)"
        )
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if swapper.imwrite(path, self.result):
            self.status.showMessage(f"Saved to {path}")
        else:
            self._fail("Could not write that file.")

    def closeEvent(self, ev):
        # Tearing down the interpreter while a QThread is still inside an
        # onnxruntime call aborts the process, so let them finish first.
        for task in list(self._tasks):
            task.wait(10000)
        super().closeEvent(ev)


    # ---------- drag & drop ----------

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls()]
        if paths:
            self.load_target(paths[0])


STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: #e6e9ef;
    font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
QFrame#panel {{ background: {PANEL}; border-radius: 12px; }}
QLabel {{ background: transparent; }}
QLabel#section {{ color: #8b93a3; font-size: 11px; font-weight: 600;
    letter-spacing: 1.2px; }}
QLabel#caption {{ color: #8b93a3; font-size: 11px; font-weight: 600;
    letter-spacing: 1px; padding: 4px 2px 8px 2px; }}
QLabel#hint {{ color: #7a8292; font-size: 11px; }}
QPushButton {{ background: #2a2f39; border: none; border-radius: 8px;
    padding: 10px 14px; color: #e6e9ef; }}
QPushButton:hover {{ background: #343a46; }}
QPushButton:disabled {{ background: #23262e; color: #565c68; }}
QPushButton#primary {{ background: {ACCENT}; color: #ffffff; font-weight: 600;
    font-size: 14px; }}
QPushButton#primary:hover {{ background: #5f9bff; }}
QPushButton#primary:disabled {{ background: #2b3242; color: #666e7d; }}
QPushButton#help {{ background: rgba(76, 141, 255, 0.10);
    border: 1px solid {ACCENT}; border-radius: 8px; color: {ACCENT};
    font-weight: 600; text-align: left; padding-left: 12px; }}
QPushButton#help:hover {{ background: {ACCENT}; color: #ffffff; }}
QDialog {{ background: {PANEL}; }}
QScrollArea {{ background: {PANEL}; }}
QLabel#helpTitle {{ font-size: 21px; font-weight: 700; color: #f2f5fa; }}
QLabel#helpLede {{ font-size: 13px; color: #99a2b2; }}
QLabel#helpHead {{ font-size: 14px; font-weight: 600; color: #e6e9ef; }}
QLabel#helpBody {{ font-size: 13px; color: #99a2b2; line-height: 150%; }}
QLabel#badge {{ background: {ACCENT}; color: #ffffff; border-radius: 14px;
    font-weight: 700; font-size: 13px; }}
QLabel#bullet {{ color: {ACCENT}; font-size: 14px; font-weight: 700; }}
QFrame#rule {{ color: #2f3542; }}
QTabWidget::pane {{ border: none; border-top: 1px solid #2f3542; }}
QTabBar::tab {{ background: transparent; color: #8b93a3; padding: 8px 11px;
    border-bottom: 2px solid transparent; font-size: 12px; }}
QTabBar::tab:selected {{ color: #e6e9ef; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: #c9d0dc; }}
QComboBox {{ background: #2a2f39; border: none; border-radius: 8px;
    padding: 8px 12px; color: #e6e9ef; }}
QComboBox:hover {{ background: #343a46; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: #2a2f39; color: #e6e9ef;
    selection-background-color: {ACCENT}; border: 1px solid #3a4150;
    outline: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3a4150; border-radius: 5px;
    min-height: 36px; }}
QScrollBar::handle:vertical:hover {{ background: #4c5464; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: #333947; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px; }}
QStatusBar {{ color: #8b93a3; }}
QSplitter::handle {{ background: transparent; width: 14px; }}
"""


def main():
    # Lets the packaged exe check its own install on a machine with no Python:
    #   FaceSwap.exe --selftest
    if "--selftest" in sys.argv:
        import selftest

        raise SystemExit(selftest.main([a for a in sys.argv[1:] if a != "--selftest"]))

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
