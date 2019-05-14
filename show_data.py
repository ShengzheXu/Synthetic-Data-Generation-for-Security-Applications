import sys
import configparser
import pandas as pd
import utils.plot_utils as plot_utils

if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
    
    if '-raw' in sys.argv:
        print('reach show raw_data')
        
        config = configparser.ConfigParser()
        config.read('config.ini')
        user_list = config['DEFAULT']['userlist'].split(',')
        
        x_data = []
        y_data = []
        for ip in user_list:
            x_data.append(ip)
            y_data_i = []
            source_data = './data/raw_data/day_1_%s.csv' % ip
            all_record = pd.read_csv(source_data)

            for t_hour in range(24):
                str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
                chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
                y_data_i.append(len(chunk.index))
            
            y_data.append(y_data_i)
        
        plot_utils.temporal_lineplot(x_data, y_data)

    if '-spe' in sys.argv:
        testing_name = input("Input the testing file name:")
        
        x_data = [testing_name]
        y_data = []
        
        y_data_i = []
        source_data = './data/gen_data/%s.csv' % testing_name
        all_record = pd.read_csv(source_data)

        for t_hour in range(24):
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data_i.append(len(chunk.index))
        
        y_data.append(y_data_i)
        
        plot_utils.temporal_lineplot(x_data, y_data)
