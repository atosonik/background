# PyInstaller spec for the Face Swap desktop app.
#   .venv\Scripts\python.exe -m PyInstaller --noconfirm --clean FaceSwap.spec
#
# Two things here are load-bearing; see the notes on each.
import os

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("insightface")

# onnxruntime's own hook collects its DLLs. Do NOT --collect-all it: that drags
# in onnxruntime.transformers/quantization/tools, which pull sympy, onnx.reference
# and pip, and PyInstaller's isolated analysis subprocess crashes importing
# onnx.reference.
# matplotlib is NOT excludable, however unused it looks: insightface/app/__init__
# imports mask_renderer -> thirdparty.face3d.mesh.vis -> matplotlib, at import
# time and unconditionally. Dropping it builds fine and then dies at first use
# with ModuleNotFoundError. Costs ~40 MB in the bundle.
# torch, diffusers and friends are installed in this venv from the abandoned
# clothing feature. They are NOT used by the app any more and would add several
# gigabytes to a onefile build, so they are excluded explicitly -- PyInstaller
# would otherwise pull them in through any stray transitive import.
EXCLUDES = [
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "safetensors", "huggingface_hub", "tokenizers", "sympy", "mpmath",
    "onnx.reference",
    "onnxruntime.transformers", "onnxruntime.quantization",
    "onnxruntime.tools", "onnxruntime.training", "onnxruntime.backend",
    "sympy", "mpmath", "pip", "tkinter", "PyQt6",
    "IPython", "pytest", "pyximport",
]

# PyQt5 and scikit-learn each ship their own copy of the MSVC redistributable
# under the SAME filename (590 KB vs 643 KB). PyInstaller flattens every binary
# into one folder, so one silently wins -- and if PyQt5's wins, onnxruntime's
# pybind11 extension fails with "DLL load failed ... initialization routine
# failed" no matter what order things are imported in. Dropping all of them makes
# the process bind to the system runtime in C:\Windows\System32, which is the
# same runtime onnxruntime binds to successfully outside PyInstaller.
#
# Requires the Microsoft Visual C++ 2015-2022 Redistributable on the target
# machine. Windows 10/11 normally has it; onnxruntime requires it regardless.
STRIP_RUNTIME_DLLS = {
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll",
}

a = Analysis(
    ["ui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# numpy/ml_dtypes ship hash-suffixed copies (msvcp140-<hash>.dll) that cannot
# collide and are loaded by their exact name, so only the canonical names go.
a.binaries = [b for b in a.binaries
              if os.path.basename(b[0]).lower() not in STRIP_RUNTIME_DLLS]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FaceSwap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    # Console stays on: a onefile GUI that dies during import has no window to
    # show an error in, and a silent failure is far worse than a console flash.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
