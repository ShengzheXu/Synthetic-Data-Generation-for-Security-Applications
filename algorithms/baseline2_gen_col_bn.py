from GaussianMixtureModel import SourceData
from GaussianMixtureModel import GMMs
from datetime import timedelta
from datetime import datetime
from sklearn import mixture 
import numpy as np
import pandas as pd

p_T_B = {}
p_B1_B = {}
pr = []
bins = [-0.001, 3.7, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.5, 20]
bins_name = []

def find_bin(byt_number):
    return pd.cut([byt_number], bins)[0]

def build_bayesian_dep(all_record, log_byt_col):
    cats = pd.cut(log_byt_col, bins)
    # print(cats)
    all_record['bytBins'] = cats
    # print(all_record.head())
    pr = all_record['bytBins'].value_counts()/all_record['bytBins'].size
    bins_name = pr.keys()
    # print(bins_name)
    return pr, bins_name

def integrate(f, a, b, N):
    x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
    x = np.reshape(x, (-1, 1))
    fx = f(x)
    area = np.sum(fx)*(b-a)/N
    return area

def select_bayesian_output(t, b_1, pr, bins_name, clf):
    # p_B_gv_T_B1 = {}
    maxp = -1
    maxB = -1
    # print(bins_name)
    for interval in bins_name:
        # print(interval, b_1, find_bin(b_1)pr)
        p_B1_B[interval] = pr[find_bin(b_1)] # / pr[interval]
        # print('b1|b', pr[find_bin(b_1)] , pr[interval] )
        # p_T_B[interval] = all_record['bytBins'].value_counts()[interval] / all_record['bytBins'].size
        df = all_record[all_record['bytBins'] == interval]
        df_t = df[df['teT'] == t]
        p_T_B[interval] = df_t.size / df.size
        # print(df_t.size, df.size)
        # print('t|b', all_record['bytBins'].value_counts()[interval] , all_record['bytBins'].size)
        # print(interval, p_B1_B[interval], p_T_B[interval])
        p_B_gv_T_B1 = p_T_B[interval] * integrate(clf.score_samples, interval.left, interval.right, 500) 
        if maxp < p_B_gv_T_B1:
            maxp = p_B_gv_T_B1
            maxB = interval
    return np.random.uniform(maxB.left, maxB.right)


# init data & settings
np.random.seed(131)
source_data = './../data/output/expanded_day_1_42.219.153.89.csv'
all_record = pd.read_csv(source_data)#[:100]

byt_train = np.reshape(all_record['byt'].values, (-1, 1))
byt_log_train = np.log(byt_train)

time_delta_train = np.reshape(all_record['teDelta'].values, (-1, 1))
sip = all_record['sa'].values[0]
dip_train = np.ravel(all_record['da'].values)

pr, bins_name = build_bayesian_dep(all_record, np.ravel(byt_log_train))

# init models
byt_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(byt_log_train)

teDelta_model = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(time_delta_train)
start_date_time_str = '2016-04-11 00:00:00'  
start_date_time_obj = datetime.strptime(start_date_time_str, '%Y-%m-%d %H:%M:%S')

def choice_from_dip_pool():
    return np.random.choice(dip_train, 1)

dip_model = choice_from_dip_pool

# gen data
row_number = 500000
byt_col = []
last_b = 0
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
    gen_b = select_bayesian_output(now_t, last_b, pr, bins_name, byt_model)
    byt_col.append(gen_b)
    te_col.append(gen_te)
    dip_col.append(dip_model())
    gen_data.append([te_col[i], sip, dip_col[i][0], int(np.exp(gen_b)), gen_te_delta[0]])
    last_b = gen_b
    print('===>', gen_data[-1])

# write to a csv file
import csv

with open("baseline2.csv", "w", newline="") as f:
    fieldnames = ['te', 'sa', 'da', 'byt', 'teDelta']
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(gen_data)
