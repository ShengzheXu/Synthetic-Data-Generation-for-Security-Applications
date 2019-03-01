from GaussianMixtureModel import SourceData
from GaussianMixtureModel import GMMs
from datetime import timedelta
from datetime import datetime
from sklearn import mixture 
import numpy as np
import pandas as pd
import operator

ip_str = '42.219.158.226'
ip_str = 'sampled_10IPs'
output_str = '10IP'

pr = []
eps = 1e-5
bins = [0-eps, 3.7, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.5, 20]
bins_name = []

def find_bin(byt_number):
    return pd.cut([byt_number], bins)[0]

def integrate(f, a, b, N):
    x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
    x = np.reshape(x, (-1, 1))
    fx = f(x)
    area = np.sum(fx)*(b-a)/N
    return area

def build_bayesian_dep(all_record, log_byt_col, log_byt_1_col, clf):
    cats = pd.cut(log_byt_col, bins)
    # print(cats)
    all_record['bytBins'] = cats

    cats = pd.cut(log_byt_1_col, bins)
    all_record['byt-1Bins'] = cats

    # print(all_record.head())
    pr = all_record['bytBins'].value_counts()/all_record['bytBins'].size
    bins_name = pr.keys()
    # print(bins_name)
    p_B = {}
    p_T_B = {}
    p_B1_B = {}
    for t in range(24):
        p_T_B[t] = {}

    for interval in bins_name:
        p_B1_B[interval] = {}

    for interval in bins_name:
        p_B[interval] = integrate(clf.score_samples, interval.left, interval.right, 500)
        df = all_record[all_record['bytBins'] == interval]

        for t in range(24):
            df_t = df[df['teT'] == t]
            p_T_B[t][interval] = df_t.size / df.size

        for interval_1 in bins_name:
            df_b_1 = df[df['byt-1Bins'] == interval_1]
            p_B1_B[interval_1][interval] = df_b_1.size / df.size
    
    learnt_p_B_gv_T_B1 = {}
    for t in range(24):
        learnt_p_B_gv_T_B1[t] = {}
        for interval_1 in bins_name:
            learnt_p_B_gv_T_B1[t][interval_1] = {}
            for interval in bins_name:
                learnt_p_B_gv_T_B1[t][interval_1][interval] = p_T_B[t][interval] * p_B[interval] * p_B1_B[interval_1][interval]

    return bins_name, learnt_p_B_gv_T_B1

def select_bayesian_output(t, b_1, learnt_p_B_gv_T_B1, bins_name):
    # p_B_gv_T_B1 = {}
    maxp = -1
    maxB = -1
    # print(bins_name)
    b1 = find_bin(b_1)
    # for interval in bins_name:
    #     # print(interval, b_1, find_bin(b_1)pr)
    #     # p_B1_B[interval] = pr[find_bin(b_1)] # / pr[interval]
    #     # print('b1|b', pr[find_bin(b_1)] , pr[interval] )
    #     # p_T_B[interval] = all_record['bytBins'].value_counts()[interval] / all_record['bytBins'].size
        
    #     # print(df_t.size, df.size)
    #     # print('t|b', all_record['bytBins'].value_counts()[interval] , all_record['bytBins'].size)
    #     # print(interval, p_B1_B[interval], p_T_B[interval])
    #     # print(b_1, find_bin(b_1))
    #     # print(interval)
    #     # print(p_B1_B)
    #     # print("===========")
    #     # print(p_B1_B[find_bin(b_1)].keys)
    #     p_B_gv_T_B1 = p_T_B[t][interval] * p_B[interval] * p_B1_B[b1][interval]
    #     if maxp < p_B_gv_T_B1:
    #         maxp = p_B_gv_T_B1
    #         maxB = interval
    maxB = max(learnt_p_B_gv_T_B1[t][b1].items(), key=operator.itemgetter(1))[0]
    return np.random.uniform(maxB.left, maxB.right)

# init data & settings
np.random.seed(131)
source_data = './../data/baseline1&2/cleaned_data/expanded_day_1_%s.csv' % ip_str
all_record = pd.read_csv(source_data)#[:100]

byt_train = np.reshape(all_record['byt'].values, (-1, 1))
byt_log_train = np.log(byt_train)
byt1_train = np.reshape(all_record['byt-1'].values, (-1, 1))
byt1_log_train = np.log(byt1_train)

time_delta_train = np.reshape(all_record['teDelta'].values, (-1, 1))
sip = all_record['sa'].values[0]
dip_train = np.ravel(all_record['da'].values)

# init models
byt_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(byt_log_train)
bins_name, learnt_p_B_gv_T_B1 = build_bayesian_dep(all_record, np.ravel(byt_log_train), np.ravel(byt1_log_train), byt_model)

teDelta_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(time_delta_train)
start_date_time_str = '2016-04-11 00:00:00'  
start_date_time_obj = datetime.strptime(start_date_time_str, '%Y-%m-%d %H:%M:%S')

def choice_from_dip_pool():
    return np.random.choice(dip_train, 1)

dip_model = choice_from_dip_pool

# gen data

starttime = datetime.now()
row_number = 637608
byt_col = []
last_b = 1
now_t = 0
te_col = []
dip_col = []
gen_data = []
for i in range(row_number):
    _, gen_te_delta = teDelta_model.sample()
#     print(gen_te_delta, int(gen_te_delta[0]))
    start_date_time_obj = start_date_time_obj + timedelta(seconds=int(gen_te_delta[0]))
    gen_te = start_date_time_obj.strftime("%Y-%m-%d %H:%M:%S")
#     print(gen_te)
    # print(str(gen_te)[11:13])
    now_t = int(str(gen_te)[11:13])
    # print('t:::', now_t)
    gen_b = select_bayesian_output(now_t, last_b, learnt_p_B_gv_T_B1, bins_name)
    byt_col.append(gen_b)
    te_col.append(gen_te)
    dip_col.append(dip_model())
    gen_data.append([te_col[i], sip, dip_col[i][0], int(np.exp(gen_b)), gen_te_delta[0]])
    last_b = gen_b
    print(i, '===>', gen_data[-1])

# write to a csv file
import csv

with open("./../data/baseline1&2/gen_data/baseline2_%s.csv" % output_str, "w", newline="") as f:
    fieldnames = ['te', 'sa', 'da', 'byt', 'teDelta']
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(gen_data)


endtime = datetime.now()
print('process time', (endtime-starttime).seconds/60, 'mins')