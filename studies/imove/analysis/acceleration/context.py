"""
Import module context to make local package mhealth available.
"""
import sys
from pathlib import Path

dir_path = Path(__file__).parent
print("dir_path is at: ", dir_path)

# adapt src_path so that it points back to the folder "mhealth"
src_path = (dir_path / ".." / ".." / ".." / ".." / "src").resolve()
sys.path.insert(0, str(src_path))
