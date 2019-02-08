import pandas as pd
import numpy as np
from datetime import datetime
import os

def do_write(df, filename):
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names', index=False)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)

a_occur = {}
target_colname = 'da'
def cal_stats(df):
    for r in zip(df[target_colname]):
        if r[0] in a_occur:
            a_occur[r[0]] += 1
        else:
            a_occur[r[0]] = 1


normal_datafile = 'D:\\research_local_data\\april_week3_csv\\april_week3_csv\\uniq\\april.week3.csv.uniqblacklistremoved'
spam_datafile = 'D:\\research_local_data\\spam_april_week3_csv\\april\\week3\\spam_flows_cut.csv'

outputfile = 'output\\day_1_allinternalsada.csv'
outputfile = 'output\\day_1_42.219.153.89.csv'

columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
theDate = '2016-04-11'
theLocalIp = '42.219.153.89'
theLocalIp = '42.219'

filteredList = []
chunkNum = 0
totalLines = 0

starttime = datetime.now()
chunksize = 10 ** 6

for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
    block_time1 = datetime.now()
    # chunk = chunk[(chunk['te'].str.startswith(theDate)) & 
    #             ((chunk['sa'].str.startswith(theLocalIp)) | (chunk['da'].str.startswith(theLocalIp)))]
    # chunk = chunk[(chunk['te'].str.startswith(theDate)) & (chunk['sa'].str.startswith(theLocalIp))]
    chunk = chunk[chunk['te'].str.startswith(theDate) & chunk['da'].str.startswith(theLocalIp)]

    if (len(chunk.index) == 0):
        continue
        # break
    else:
        chunkNum += 1
        totalLines += len(chunk.index)
        cal_stats(chunk)
        print(len(chunk.index), "to write")
        # do_write(chunk, outputfile)
        block_time2 = datetime.now()
        print("blockNum", chunkNum, "cur_total_record", totalLines, "time:", (block_time2-block_time1).seconds)
    if chunkNum == 154:
        break

most_a = sorted( ((v,k) for k,v in a_occur.items()), reverse=True)
print("most "+target_colname, most_a[:10])

endtime = datetime.now()
print('process time', (endtime-starttime).seconds)
