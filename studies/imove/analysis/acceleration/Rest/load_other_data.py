"""Load other data """


## 2 ##  store/001.h5
filepath = '/Users/julien/GD/ACLS/TM/DATA/extracted/quality50_clipped/store/001.h5'
store = pd.HDFStore(filepath, mode='r')
store.keys()
h5_001 = store["/exercises"]
h5_001 = store["/vital/left"]
h5_001 = store["/vital/right"]
h5_001 = store["/raw/left"]
h5_001 = store["/raw/right"]


# PATHS ----------------------------------------------------------------------------

#wd = os.getcwd()
#path_src = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/mhealth/src'
#path_output = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 Module/TM/OUTPUT'
#plots_path = '/Users/julien/My Drive/20_STUDIUM/ACLS/05 MODULE/TM/OUTPUT/plots/'
#path_data = '/Users/julien/GD/ACLS/TM/DATA/'