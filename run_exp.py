from models.baselines import baseline1
from models.baselines import baseline2
from datetime import datetime
from datetime import timedelta
from utils.config_utils import recieve_cmd_config
import numpy as np
import pandas as pd
import argparse
import configparser
import gc

np.random.seed(131)
user_list = []
baseline_choice = 'baseline1'
day_number = 1
real_gen = 'no'

def data_prepare(ip_str):
    source_data = './data/baseline1&2/cleaned_data/expanded_day_1_%s.csv' % ip_str
    all_record = pd.read_csv(source_data)
    print('pre', all_record.shape)
    all_record = all_record.sample(frac=0.1, random_state=1)
    print('after',all_record.shape)

    byt_train = np.reshape(all_record['byt'].values, (-1, 1))
    byt_log_train = np.log(byt_train)
    byt1_train = np.reshape(all_record['byt-1'].values, (-1, 1))
    byt1_log_train = np.log(byt1_train)

    time_delta_train = np.reshape(all_record['teDelta'].values, (-1, 1))
    sip = all_record['sa'].values[0]
    dip_train = np.ravel(all_record['da'].values)

    teT_df_col = all_record#['teT']


    return byt_log_train, time_delta_train, sip, dip_train, byt1_log_train, teT_df_col

def model_prepare(sip, byt_log_train, time_delta_train, dip_train, byt1_log_train=None, teT_df_col=None):
    if baseline_choice == 'baseline1':
        meta_model = baseline1(sip, byt_log_train, time_delta_train, dip_train)
        return meta_model
    elif baseline_choice == 'baseline2':
        meta_model = baseline2(sip, byt_log_train, time_delta_train, dip_train, byt1_log_train, teT_df_col)
        return meta_model
    else:
        pass

def flush(gen_data):
    # write to a csv file
    import csv
    import os
    gen_file = "./data/baseline1&2/gen_data/%s_%sdays.csv" % (baseline_choice, day_number)
    label = True
    if os.path.isfile(gen_file):
        label = False
    with open(gen_file, "a", newline="") as f:
        fieldnames = ['te', 'sa', 'da', 'byt', 'teDelta']
        writer = csv.writer(f)
        if label:
            writer.writerow(fieldnames)
        writer.writerows(gen_data)

def do_one():
    starttime = datetime.now()
    final_byt_log_train = np.reshape(np.array([]), (-1, 1))
    final_time_delta_train = np.reshape(np.array([]), (-1, 1))
    final_sip = []
    final_dip_train = np.ravel(np.array([]))
    final_byt1_log_train = np.reshape(np.array([]), (-1, 1))
    final_teT_df_col = None


    for deal_str in user_list:
        print(deal_str)
        byt_log_train, time_delta_train, sip, dip_train, byt1_log_train, teT_df_col = data_prepare(deal_str)

        final_byt_log_train = np.concatenate((final_byt_log_train, byt_log_train))
        final_time_delta_train = np.concatenate((final_time_delta_train, time_delta_train))
        final_sip.append(sip)
        final_dip_train = np.concatenate((final_dip_train, dip_train))

        final_byt1_log_train = np.concatenate((final_byt1_log_train, byt1_log_train))
        if final_teT_df_col is None:
            final_teT_df_col = teT_df_col
        else:
            final_teT_df_col = final_teT_df_col.append(teT_df_col) #np.concatenate((final_teT_df_col, teT_df_col)) #
        # print(len(final_byt_log_train))
        del byt_log_train, time_delta_train, sip, dip_train, byt1_log_train, teT_df_col
        gc.collect()

    print(final_teT_df_col.shape)
    model1 = model_prepare(final_sip, final_byt_log_train, final_time_delta_train, final_dip_train, final_byt1_log_train, final_teT_df_col)

    gen_data = []
    now_t = 0
    last_b = 1
    cnt = 0
    current_date = 11
    start_date = -1
    while True:
        dep_info = [now_t, last_b] if baseline_choice == 'baseline2' else []
        gen_te, gen_dip, gen_byt, gen_te_delta = model1.generate_one(dep_info)
        gen_data.append([gen_te, final_sip[0], gen_dip, gen_byt, gen_te_delta])
        now_t = int(str(gen_te)[11:13])
        last_b = gen_byt
        cnt += 1
        print(cnt, gen_data[-1])

        if start_date == -1:
            start_date = int(gen_te[8:10])
        if (int(gen_te[8:10]) - start_date) >= day_number:
            flush(gen_data[:-1])
            break

        if int(gen_te[8:10]) > current_date:
            current_date += 1
            if real_gen == 'yes':
                flush(gen_data[:-1])
                gen_data = [gen_data[-1]]
    
    endtime = datetime.now()
    with open("exp_record.txt", "a") as myfile:
        myfile.write(str(len(user_list)) + 'users,' + str(day_number) +'days,'+ baseline_choice
            + ' ==> time:' + str((endtime-starttime).seconds/60) + 'mins\n')


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config-bl2.ini')
    user_list = config['DEFAULT']['userlist'].split(',')
    baseline_choice = config['DEFAULT']['baseline']
    day_number = int(config['DEFAULT']['daynumber'])
    real_gen = config['DEFAULT']['do_gen']
    
    # override the config with the command line
    recieve_cmd_config(config['DEFAULT'])
    
    # run the experiment
    do_one()