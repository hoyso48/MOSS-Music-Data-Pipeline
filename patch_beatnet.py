"""Patch BeatNet to make pyaudio import optional."""
import os
path = os.path.join(
    "/inspire/ssd/project/embodied-multimodality/public/hfchen/miniconda",
    "miniconda3/envs/music_cpu/lib/python3.9/site-packages/BeatNet/BeatNet.py",
)
with open(path) as f:
    content = f.read()
old = "import pyaudio
"
new = "try:
    import pyaudio
except ImportError:
    pyaudio = None
"
if old in content:
    content = content.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(content)
    print("OK: patched")
else:
    print("SKIP: already patched?")
