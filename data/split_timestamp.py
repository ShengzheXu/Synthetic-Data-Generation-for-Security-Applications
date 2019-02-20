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

def filter(source_data):
    all_record = pd.read_csv(source_data)
    print(all_record.tail())

    # add teT column
    all_record['teT'] = all_record['te'].apply(date_to_T).astype(int)
    
    # print(all_record.head())
    # add teS & teDelta column
    all_record['teS'] = all_record['te'].apply(date_to_seconds).astype(int)
    all_record['teDelta'] = all_record['teS'].diff().fillna(0).astype(int)
    
    print(all_record.head())

    do_write(all_record, './../data/output/expanded_day_1_42.219.153.89.csv')
    
def rectify_byt():
    source_data = './../data/day_1_42.219.153.89/original_hour_division/T%s_day_1_42.219.153.89.csv'
    for t in range(0, 24):
        t_str = str(int(t/10)) + str(t%10)
        df = pd.read_csv(source_data % t_str)
        # print(df.head())
        df['byt'] = df['byt'].map(lambda x: int((x+50)/100) * 100)
        # print(df.head())
        print(len(df.index))
        # do_write(df, './../data/day_1_42.219.153.89/rectified_hour_division/T%s_day_1_42.219.153.89.csv' % t_str)

if __name__ == "__main__":
    source_data = './../data/day_1_42.219.153.89/day_1_42.219.153.89.csv'
    filter(source_data)

#     rectify_byt()