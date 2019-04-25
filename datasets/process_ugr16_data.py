import pandas as pd
import os
import sys

working_folder = './../data/'

def do_write(df, filename):
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names', index=False)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)

def date_to_seconds(end_date_time_str):
    import datetime
    start_date_time_str = '2016-04-11 00:00:00'  
    start_date_time_obj = datetime.datetime.strptime(start_date_time_str, '%Y-%m-%d %H:%M:%S')
    end_date_time_obj = datetime.datetime.strptime(end_date_time_str, '%Y-%m-%d %H:%M:%S')
    return (end_date_time_obj-start_date_time_obj).total_seconds()

def date_to_T(date_time_str):
    hh = date_time_str[11:13]
    if hh.startswith('0'):
        hh = hh[-1]
    return hh

def filter(source_data, name_str):
    all_record = pd.read_csv(source_data)
    # print(all_record.tail())

    # add byt-1 column
    all_record['byt-1'] = all_record['byt'].shift(1).fillna(2).astype(int)

    # add teT & teS & teDelta column
    all_record['teT'] = all_record['te'].apply(date_to_T).astype(int)
    all_record['teS'] = all_record['te'].apply(date_to_seconds).astype(int)
    all_record['teDelta'] = all_record['teS'].diff().fillna(0).astype(int)
    
    # print(all_record.head())

    do_write(all_record, working_folder+'cleaned_data/expanded_day_1_%s.csv' % name_str)

def split_file(source_data, name_str):
    all_record = pd.read_csv(source_data)
    theDate = '2016-04-11 '
    for T in range(24):
        hour_str = '00'
        if T<10:
            hour_str = hour_str[:-1] + str(T)[0]
        else:
            hour_str = hour_str[:-2] + str(T)[:1]
        if not os.path.exists(source_data+hour_str):
            os.makedirs(source_data+hour_str)
        
        theDate = theDate + hour_str
        chunk = all_record[all_record['te'].str.startswith(theDate)]
        
        # do_write(chunk, source_data+hour_str+'/'+name_str)

if __name__ == "__main__":
    # ten_ips = ['42.219.159.118', '42.219.159.76', '42.219.159.195', '42.219.159.171', '42.219.159.199',
    #             '42.219.170.246', '42.219.159.186', '42.219.159.182', '42.219.159.179', '42.219.159.221']
    ten_ips = ['42.219.159.170','42.219.159.95']

    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
    
    if '-f' in sys.argv:
        print('reach filter, to generate the expanded_csv files')
        for ip_str in ten_ips:
            source_data = working_folder + 'raw_data/day_1_%s.csv' % ip_str
            filter(source_data, ip_str)
    
    if '-split_t' in sys.argv:
        print('reach split_t')
        target_dir = '../data/baseline1&2/gen_data/'
        target_name_pattern = ''

        for ip_str in ten_ips:
            source_data = 'data/baseline1&2/raw_data/'
            split_file(source_data, 'day_1_%s.csv' % ip_str)
        
