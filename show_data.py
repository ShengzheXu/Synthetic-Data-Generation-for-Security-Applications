import sys
import configparser
import pandas as pd
import glob
import utils.plot_utils as plot_utils
import numpy as np

if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) < 2:
        print('no instruction input.')
        sys.exit()
    
    if '-raw' in sys.argv:
        print('reach show raw_data')
        
        source_folder = './data/midium100users/raw_data/'
        # source_folder = './data/gen_data/baseline2_1days_folder/'

        all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)

        x_data = [source_folder]
        y_data = [[]]
        for t_hour in range(24):   
            y_data_i = []
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data[0].append(len(chunk.index))
        
        plot_utils.temporal_lineplot(x_data, y_data, x_label="hour", y_label="#flow", title="#flow distribution over 100 users")

    if '-spe' in sys.argv:
        # testing_name = input("Input the testing file name:")
        # testing_name = 'day_1_42.219.144.193'
        # source_data = './data/gen_data/%s.csv' % testing_name
        # source_data = './data/raw_data/%s.csv' % testing_name
        # all_record = pd.read_csv(source_data)

        source_folder = './data/midium100users/raw_data/'
        # source_folder = './data/gen_data/baseline2_1days_folder/'

        all_record = pd.concat([pd.read_csv(f) for f in glob.glob(source_folder+'*.csv')], ignore_index = True)

        x_data = []
        y_data = []

        for t_hour in range(24):   
            x_data.append(str(t_hour))
            y_data_i = []
            str_hour = str(t_hour) if t_hour > 9 else '0'+ str(t_hour)
            chunk = all_record[all_record['te'].str.contains(' '+str_hour+':')]
            y_data_i = np.log(chunk['byt'].values.tolist())
            # y_data_i = chunk['byt'].values.tolist()
            print(t_hour, len(y_data_i), y_data_i)
            y_data.append(y_data_i)
        
        # plot_utils.temporal_lineplot(x_data, y_data)
        plot_utils.boxplot(x_data, y_data, x_label='hour', y_label='byt (log)', title='hour distribution of all '+source_folder)

    
