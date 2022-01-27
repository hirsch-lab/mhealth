


acc = acc.reset_index() # create new index. timestamp becomes a normal col
A = acc['timestamp'].groupby([ acc['DeMortonDay'], acc['Patient'], acc['Side'] ]).min()
# mask = () & ()

# SQL ----------------------------------------------------------------------------

# Omit need to pass in a global variables every time an object is used:
pysqldf = lambda q: sqldf(q, globals()) # 
# Now, whenever you pass a query into pysqldf, the global variables will be passed 
# along in the lambda so that you don’t have to do that over and over again for each 
# object that’s used.
       
q = """SELECT Patient, DeMortonDay, Side, MIN(timestamp), MAX(timestamp), MAX(timestamp) + 0000-0-00 01
       FROM acc 
       GROUP BY Patient, DeMortonDay, Side
       ORDER BY Patient, DeMortonDay, Side
       LIMIT 30;"""
#  WHERE  BETWEEN AND
# MIN(timestamp), MAX(timestamp),
       

bla = pysqldf(q) # execute SQL-query on df
bla








