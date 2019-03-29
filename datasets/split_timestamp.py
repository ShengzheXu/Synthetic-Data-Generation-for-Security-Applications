import pandas as pd
import os

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
    print(all_record.tail())

    # add byt-1 column
    all_record['byt-1'] = all_record['byt'].shift(1).fillna(2).astype(int)

    # add teT & teS & teDelta column
    all_record['teT'] = all_record['te'].apply(date_to_T).astype(int)
    all_record['teS'] = all_record['te'].apply(date_to_seconds).astype(int)
    all_record['teDelta'] = all_record['teS'].diff().fillna(0).astype(int)
    
    # print(all_record.head())

    do_write(all_record, 'baseline1&2/cleaned_data/expanded_day_1_%s.csv' % name_str)
    

if __name__ == "__main__":
    # ip_str = '42.219.158.226'
    ten_ips = ['42.219.153.7', '42.219.153.89', '42.219.155.56', '42.219.155.26', '42.219.159.194',
            '42.219.152.249', '42.219.159.82', '42.219.159.92', '42.219.159.94', '42.219.158.226']
    for ip_str in ten_ips:
        source_data = 'baseline1&2/raw_data/day_1_%s.csv' % ip_str
        filter(source_data, ip_str)

#     rectify_byt()