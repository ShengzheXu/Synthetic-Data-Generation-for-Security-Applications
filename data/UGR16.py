import pandas as pd
import numpy as np
from datetime import datetime
import os

def do_write(df, filename):
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names')
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False)

datafile = 'D:\\research_local_data\\april_week3_csv\\april_week3_csv\\uniq\\april.week3.csv.uniqblacklistremoved'
# outputfile = 'output\\selected_data%s.csv'
outputfile = 'output\\day_1_internal_ip_only.csv'
# datafile = 'D:\\research_local_data\\spam_april_week3_csv\\april\\week3\\spam_flows_cut.csv'
columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
theDate = '2016-04-11'
theLocalIp = '42.219'
userFilter = {}
filteredList = []
chunkNum = 0
totalLines = 0

starttime = datetime.now()
chunksize = 10 ** 6
for chunk in pd.read_csv(datafile, chunksize=chunksize, header=None, names = columnName):
    first_rows = chunk.head(n=5)
    # print(first_rows)
    # print(chunk[0:1, 'te'].str.split(' ')[0])
    
    chunk = chunk[(chunk['te'].str.startswith(theDate)) & (chunk['sa'].str.startswith(theLocalIp))
                    & (chunk['da'].str.startswith(theLocalIp))]
    # print(chunk.head(n=5))
    if (len(chunk.index) == 0):
        break
    else:
        chunkNum += 1
        totalLines += len(chunk.index)
        print(chunkNum, totalLines)
        # chunk.to_csv(outputfile % chunkNum)
        do_write(chunk, outputfile)

endtime = datetime.now()
print('process time', (endtime-starttime).seconds)