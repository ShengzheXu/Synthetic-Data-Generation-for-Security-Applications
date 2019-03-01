import pandas as pd
import numpy as np
from datetime import datetime
import os

columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
theDate = '2016-04-11'

normal_datafile = 'D:\\research_local_data\\april_week3_csv\\april_week3_csv\\uniq\\april.week3.csv.uniqblacklistremoved'
spam_datafile = 'D:\\research_local_data\\spam_april_week3_csv\\april\\week3\\spam_flows_cut.csv'

outputfile = 'baseline1&2/raw_data/day_1_%s.csv'

def do_write(df, filename):
    filename = outputfile % filename
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names', index=False)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)

def extract(theLocalIp):
    chunkNum = 0
    totalLines = 0
    # targetLines = 91342588
    gen_flag = True
    chunksize = 10 ** 6

    for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
        block_time1 = datetime.now()
        if isinstance(theLocalIp, list):
            chunk = chunk[(chunk['te'].str.startswith(theDate)) & (chunk['sa'].isin(theLocalIp))]
        else:
            chunk = chunk[(chunk['te'].str.startswith(theDate)) & (chunk['sa'].str.endswith(theLocalIp))]
    
        if (len(chunk.index) == 0):
            continue
        else:
            chunkNum += 1
            chunk = chunk.sample(n=int(len(chunk.index)/10),random_state=131,axis=0)
            print(len(chunk.index), "to write")
            if gen_flag:
                # if totalLines + len(chunk.index) > targetLines:
                #     overlines = totalLines + len(chunk.index) - targetLines
                #     chunk = chunk.head(len(chunk.index) - overlines )
                if isinstance(theLocalIp, list):
                    do_write(chunk, 'merged_10IPs')
                else:
                    do_write(chunk, theLocalIp)
                
            totalLines += len(chunk.index)
            block_time2 = datetime.now()
            print("blockNum", chunkNum, "cur_total_record", totalLines, "time:", (block_time2-block_time1).seconds)
        if chunkNum == 154: # totalLines == targetLines 
            break

def sample_choice(filename, num_of_row):
    all_record = pd.read_csv(filename)
    all_record = all_record.sample(n=num_of_row,random_state=131,axis=0)
    do_write(all_record, 'sampled_10IPs')

occur_dict = {}
def cal_stats(df, target_colname):
    for r in zip(df[target_colname]):
        if r[0] in occur_dict:
            occur_dict[r[0]] += 1
        else:
            occur_dict[r[0]] = 1

def probe(theLocalIp, target_colname, num_of_selected):
    starttime = datetime.now()
    chunksize = 10 ** 6
    chunkNum = 0
    import gc

    for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
        chunk = chunk[(chunk['te'].str.startswith(theDate)) & 
                    ((chunk['sa'].str.startswith(theLocalIp)) | (chunk['da'].str.startswith(theLocalIp)))]
        # chunk = chunk[(chunk['te'].str.startswith(theDate)) & (chunk['sa'].str.endswith(theLocalIp))]
        # chunk = chunk[chunk['te'].str.startswith(theDate) & chunk['da'].str.startswith(theLocalIp)]

        if (len(chunk.index) == 0):
            break
        else:
            chunkNum += 1
            cal_stats(chunk, target_colname)
            print("blockNum", chunkNum, "with", len(chunk.index))
        del chunk
        gc.collect()

    most_occure = sorted( ((v,k) for k,v in occur_dict.items()), reverse=True)
    print("most "+ target_colname, most_occure[:num_of_selected])

    endtime = datetime.now()
    print('process time', (endtime-starttime).seconds)


if __name__ == "__main__":
    # probe('42.219', 'sa', 20)

    ten_ips = ['42.219.153.7', '42.219.153.89', '42.219.155.56', '42.219.155.26', '42.219.159.194',
                '42.219.152.249', '42.219.159.82', '42.219.159.92', '42.219.159.94', '42.219.158.226']


    # extract('42.219.158.226')

    # extract(ten_ips)
    sample_choice(outputfile % 'merged_10IPs', 637608) # 637608, '42.219.158.226'
    
    # for ip in ten_ips:
    #     print('now extracting: %s' % ip)
    #     extract(ip)


# 10667863, '42.219.156.211'
# 10664359, '42.219.156.231'
# 5982859, '42.219.159.95'
# 3995830, '42.219.153.191'
# 2760867, '42.219.155.28'
# 2126619, '42.219.153.62'
# 2031099, '42.219.159.85'
# 1740711, '42.219.158.156'
# below 10 IPs to build gen_data_version1
# 1366940, '42.219.153.7'      
# 1342589, '42.219.153.89'     
# 1210025, '42.219.155.56'
# 1175793, '42.219.155.26'
# 1081080, '42.219.159.194'
# 1046866, '42.219.152.249'
# 944492, '42.219.159.82'
# 888781, '42.219.159.92'
# 771349, '42.219.159.94'
# 637608, '42.219.158.226'           <======== standard