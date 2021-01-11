# =============================================================================
# Unit tests
# =============================================================================
# from patient.imove_label_loader import ImoveLabelLoader
# ll = ImoveLabelLoader()
# ll.load_labels(dir_name="labels", filename="001-1.xlsx", tz_to_zurich=True)




# from patient.imove_label_loader import ImoveLabelLoader

# from patient_tests.imove_label_loader_test import ImoveLabelLoaderTest
# import patient_tests.imove_label_loader_test

# test = ImoveLabelLoaderTest()
# test.test_load_labels()

# =============================================================================
# CHANGES OF de_morton values
# =============================================================================
# col 'match': check whether previous row has same de_morton value
df['match'] = df.de_morton.eq(df.de_morton.shift())
# df['match']

# compare matches and changes
sum(df['match'])
~df['match'] # invert Series of Boolean
sum(~df['match'])


# =============================================================================
# groupby
# =============================================================================
# min & max date_time of each date
acc.groupby(['date'])['date_time'].min()
acc.groupby(['date'])['date_time'].max()
# acc['Data4'] = acc['Data3'].groupby(acc['Date']).transform('sum')
# acc["surveillance_start"] = acc['date_time'].groupby(acc['date']).transform('min')
# acc["surveillance_end"] = acc['date_time'].groupby(acc['date']).transform('max')
# acc.loc[:,"surveillance_end"] = acc['date_time'].groupby(acc['date']).transform('max')

# 
acc["surveillance_start"]
acc["surveillance_end"]

