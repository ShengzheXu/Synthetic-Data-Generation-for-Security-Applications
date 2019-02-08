import pandas as pd
import os

def do_write(df, filename):
    # if file does not exist write header 
    if not os.path.isfile(filename):
        df.to_csv(filename, header='column_names', index=False)
    else: # else it exists so append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)

def filter(source_data):
    all_record = pd.read_csv(source_data)
    print(all_record.tail())
    for t in range(0, 24):
        t_str = str(int(t/10)) + str(t%10)

        print('2016-04-11 %s' % t_str),
        df = all_record[all_record['te'].str.startswith('2016-04-11 %s' % t_str)]
        print('=======')
        print(df.head())
        print(df.tail())
        do_write(df, './../data/output/T%s_day_1_42.219.153.89.csv' % t_str)
    
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
    # source_data = './../data/output/day_1_42.219.153.89.csv'
    # filter(source_data)

    rectify_byt()