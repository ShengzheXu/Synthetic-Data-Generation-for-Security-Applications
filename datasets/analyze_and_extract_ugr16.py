import pandas as pd
import numpy as np
from datetime import datetime
import configparser
import os
import sys

columnName = ['te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'fwd', 'stos', 'pkt', 'byt', 'lable']
theDate = '2016-04-11'
internal_ip = '42.219.'

normal_datafile = 'D:\\research_local_data\\april_week3_csv\\april_week3_csv\\uniq\\april.week3.csv.uniqblacklistremoved'
spam_datafile = 'D:\\research_local_data\\spam_april_week3_csv\\april\\week3\\spam_flows_cut.csv'

working_folder = './../data/'
outputfile = working_folder+'raw_data/day_1_%s.csv'

def prepare_folders():
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)
    sub_folders = ['raw_data', 'cleaned_data', 'gen_data']
    for i in sub_folders:
        if not os.path.exists(working_folder+i+'/'):
            os.makedirs(working_folder+i+'/')

def do_write(df, filename):
    filename = outputfile % filename
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names', index=False)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)

def extract(theUserIP):
    chunkNum = 0
    gen_flag = True
    chunksize = 10 ** 6
    import gc

    for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
        block_time1 = datetime.now()
        chunk = chunk[chunk['te'].str.startswith(theDate)]
        if (len(chunk.index) == 0):
            break

        chunkNum += 1
        # chunk = chunk.sample(n=int(len(chunk.index)/10),random_state=131,axis=0)
        if isinstance(theUserIP, list):
            for one_ip in theUserIP:
                chunk2 = chunk[chunk['sa'] == one_ip]
                if gen_flag:
                    print(len(chunk2.index), "to write for", one_ip)
                    do_write(chunk2, one_ip)
                del chunk2
        else:
            chunk2 = chunk[chunk['sa'] == theUserIP]
            print(len(chunk2.index), "to write")
            do_write(chunk2, theUserIP)
            
        block_time2 = datetime.now()
        print("blockNum", chunkNum, ",time:", (block_time2-block_time1).seconds)
        del chunk
        gc.collect()

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

def analyze(theUserIP, target_colname, num_of_selected):
    starttime = datetime.now()
    chunksize = 10 ** 6
    chunkNum = 0
    import gc

    for chunk in pd.read_csv(normal_datafile, chunksize=chunksize, header=None, names = columnName):    
        chunk = chunk[chunk['te'].str.startswith(theDate)]
        if (len(chunk.index) == 0):
            break
        chunk = chunk[chunk['sa'].str.startswith(theUserIP) | chunk['da'].str.startswith(theUserIP)]
        # chunk = chunk[chunk['sa'].str.startswith(theUserIP)]
        chunkNum += 1
        cal_stats(chunk, target_colname)
        print("blockNum", chunkNum, "with", len(chunk.index))
        del chunk
        gc.collect()

    most_occure = sorted( ((v,k) for k,v in occur_dict.items()), reverse=True)
    print("most "+ target_colname, most_occure[:num_of_selected])

    with open('data_stats.csv', 'w') as f:
        outstring = 'occurrence,ip\n'
        for case in most_occure:
            outstring += str(case[0]) + ',' + case[1] + '\n'
        
        f.write(outstring)

    endtime = datetime.now()
    print('process time', (endtime-starttime).seconds)

def plot_refer(stats_file):
    a = pd.read_csv(stats_file)
    a = a[a['ip'].str.startswith(internal_ip)]
    a.to_csv('internal_only_'+stats_file, header='column_names', index=False)
    sys.path.append('../')
    print("current sys path:", sys.path)
    from utils.plot_utils import plot_source_distribution

    num_of_connection = a['occurrence'].values.tolist()
    plot_source_distribution(np.log(num_of_connection))

    user_addresses = a['ip'].values.tolist()
    print(len(num_of_connection), sum(num_of_connection))
    median_index = int(len(num_of_connection)/2)
    for i in range(median_index-5, median_index+5):
        print(user_addresses[i], num_of_connection[i])

# call this function with python3 UGR16.py [arg], where [arg] is '-p' or '-e' (for probe and extract seperately).
if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
    
    if '-a' in sys.argv:
        print('reach analyze')
        analyze('42.219.', 'sa', 20)

    if '-p' in sys.argv:
        print('reach plot_stats')
        stats_file = 'data_stats_3traffic.csv'
        plot_refer(stats_file)
    
    if '-e' in sys.argv:
        print('reach extract')

        # ten_ips = ['42.219.153.7', '42.219.153.89', '42.219.155.56', '42.219.155.26', '42.219.159.194',
        #         '42.219.152.249', '42.219.159.82', '42.219.159.92', '42.219.159.94', '42.219.158.226']

        # ten_ips = ['42.219.159.118', '42.219.159.76', '42.219.159.195', '42.219.159.171', '42.219.159.199',
        #         '42.219.170.246', '42.219.159.186', '42.219.159.182', '42.219.159.179', '42.219.159.221']

        # ten_ips = ['42.219.159.170','42.219.159.95']

        config = configparser.ConfigParser()
        config.read('./../config.ini')
        user_list = config['DEFAULT']['userlist'].split(',')
        print('extracting:', user_list)
        prepare_folders()
        extract(user_list)

        # for ip in ten_ips:
        #     print('now extracting: %s' % ip)
        #     extract(ip)
        # sample_choice(outputfile % 'merged_10IPs', 637608) # 637608, '42.219.158.226'
        
        


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