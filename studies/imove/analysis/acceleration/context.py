import sys
from pathlib import Path

dir_path = Path(__file__).parent
print("dir_path is at: ", dir_path)

# adapt src_path so that it points back to the folder "mhealth"
src_path = (dir_path / ".." / ".." / ".." / ".." / "src").resolve()
sys.path.insert(0, str(src_path))


# PATHS ----------------------------------------------------------------------------
path_src = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth/src'
path_output = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/OUTPUT'
path_output_plots = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 MODULE/TM/OUTPUT/plots/'
path_data = '/Users/julien/GD/ACLS/TM/DATA/'