import sys
from pathlib import Path

dir_path = Path(__file__).parent
src_path = (dir_path / ".." / ".." / "src").resolve()
sys.path.insert(0, str(src_path))
