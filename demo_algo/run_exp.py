from baselines import baseline1
from baselines import baseline2
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd

np.random.seed(131)
ten_ips = ['42.219.153.7', '42.219.153.89', '42.219.155.56', '42.219.155.26', '42.219.159.194',
        '42.219.152.249', '42.219.159.82', '42.219.159.92', '42.219.159.94', '42.219.158.226']

baseline_choice = 'baseline2'
nol = 637608

def data_prepare(ip_str):
    source_data = './../data/baseline1&2/cleaned_data/expanded_day_1_%s.csv' % ip_str
    all_record = pd.read_csv(source_data)

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

def flush(gen_data, out_str):
    # write to a csv file
    import csv

    with open("./../data/baseline1&2/gen_data/%s_%s.csv" % (baseline_choice, out_str), "w", newline="") as f:
        fieldnames = ['te', 'sa', 'da', 'byt', 'teDelta']
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(gen_data)

def do_one(deal_str=None):
    starttime = datetime.now()
    if deal_str == 'ten_ips':
        # ten ips experiment
        models = []
        for ip_str in ten_ips:
            byt_log_train, time_delta_train, sip, dip_train, byt1_log_train, teT_df_col = data_prepare(ip_str)
            model = model_prepare(sip, byt_log_train, time_delta_train, dip_train, byt1_log_train, teT_df_col)
            models.append(model)
            print(ip_str, 'model ready')

        gen_data = []
        now_t = 0
        last_b = 1
        start_date_time_str = '2016-04-11 00:00:00'  
        start_date_time_obj = datetime.strptime(start_date_time_str, '%Y-%m-%d %H:%M:%S')
        for i in range(nol):
            step_gen = []
            dep_info = [now_t, last_b] if baseline_choice == 'baseline2' else []
            for model in models:
                step_gen.append(list(model.generate_one(dep_info)))
            # gen_te = step_gen[np.random.randint(len(models))][0]
            gen_te_delta = step_gen[np.random.randint(len(models))][3]
            start_date_time_obj = start_date_time_obj + timedelta(seconds=gen_te_delta)
            gen_te = start_date_time_obj.strftime("%Y-%m-%d %H:%M:%S")

            gen_sip = models[np.random.randint(len(models))].sip
            gen_dip = step_gen[np.random.randint(len(models))][1]
            gen_byt = [x[2] for x in step_gen]
            gen_byt = int(np.average(gen_byt))
            # record data
            gen_data.append([gen_te, gen_sip, gen_dip, gen_byt, gen_te_delta])
            now_t = int(str(gen_te)[11:13])
            last_b = gen_byt
            print(i+1, gen_data[-1])
        flush(gen_data, deal_str)
    else:
        # one ip experiment
        byt_log_train, time_delta_train, sip, dip_train, byt1_log_train, teT_df_col = data_prepare(deal_str)
        model1 = model_prepare(sip, byt_log_train, time_delta_train, dip_train, byt1_log_train, teT_df_col)

        gen_data = []
        now_t = 0
        last_b = 1
        for i in range(nol):
            dep_info = [now_t, last_b] if baseline_choice == 'baseline2' else []
            gen_te, gen_dip, gen_byt, gen_te_delta = model1.generate_one(dep_info)
            gen_data.append([gen_te, sip, gen_dip, gen_byt, gen_te_delta])
            now_t = int(str(gen_te)[11:13])
            last_b = gen_byt
            print(i+1, gen_data[-1])
        flush(gen_data, deal_str)
    
    
    endtime = datetime.now()
    with open("exp_record.txt", "a") as myfile:
        myfile.write(deal_str +','+ baseline_choice + ' ==> time:' + str((endtime-starttime).seconds/60) + 'mins\n')

if __name__ == "__main__":
    for i in range(5, 6):
        do_one(ten_ips[i])
    # do_one("ten_ips")