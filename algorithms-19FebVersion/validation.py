import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score

def KLc(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def vali_one(baseline_choice, raw_ip_str, gen_ip_str):
    rawdata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % raw_ip_str)
    # gendata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % raw_ip_str)
    # gendata = pd.read_csv('./../data/baseline1&2/gen_data/%s_%s.csv' % (baseline_choice, gen_ip_str))
    gendata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % gen_ip_str)

    values1 = np.log(list(rawdata.byt))
    values2 = np.log(list(gendata.byt))

    cats1, bins = pd.qcut(values1, 10, retbins=True, duplicates='drop')
    pr1 = cats1.value_counts()/cats1.size
    # print(pr1)
    # print("======>")
    # print(cats1.value_counts())
    # print("======>")
    # print(bins)
    # print(type(bins))
    pr1 = list(pr1)

    cats2 = pd.cut(values2, bins)
    pr2 = (cats2.value_counts()+1)/(cats2.size+1)
    # print(pr2)
    # pr2 = list(pr2)

    # print(pr1+pr2)

    KL = scipy.stats.entropy(pr1, pr2)
    # NMI = normalized_mutual_info_score(pr2,pr2)
    print('raw_data/day_1_%s' % raw_ip_str + ',raw_data/day_1_%s' % gen_ip_str + ','+ str(KL)) #, ",NMI", NMI)

def add_two_list(first, second):
    return [x + y for x, y in zip(first, second)]

def vali_avg(baseline_choice, raw_ip_str, ten_ips):
    rawdata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % raw_ip_str)
    # rawdata = pd.read_csv('./../data/baseline1&2/gen_data/%s_%s.csv' % (baseline_choice, raw_ip_str))
    values1 = np.log(list(rawdata.byt))
    cats1, bins = pd.qcut(values1, 10, retbins=True, duplicates='drop')
    pr1 = cats1.value_counts()/cats1.size
    pr1 = list(pr1)

    molecule = [0] * 10

    for ip in ten_ips:
        # gendata = pd.read_csv('./../data/baseline1&2/gen_data/%s_%s.csv' % (baseline_choice, ip))
        gendata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % ip)
        values2 = np.log(list(gendata.byt))
        cats2 = pd.cut(values2, bins)
        pr2 = (cats2.value_counts()+1)/(cats2.size+1)
        pr2 = list(pr2)
        # print(molecule)
        molecule = add_two_list(molecule, pr2)
    molecule[:] = [x / 10 for x in molecule]
    KL = scipy.stats.entropy(pr1, molecule)
    # NMI = normalized_mutual_info_score(pr2,pr2)
    
    print('raw_data/day_1_%s.csv' % ip + 'raw_data/day_1_%s.csv' % ip + ','+ str(KL)) #, ",NMI", NMI)
    # print('raw_data/day_1_%s' % raw_ip_str + ',%s_stat_10csvs' % (baseline_choice) + ','+ str(KL)) #, ",NMI", NMI)

ten_ips = ['42.219.153.7', '42.219.153.89', '42.219.155.56', '42.219.155.26', '42.219.159.194',
        '42.219.152.249', '42.219.159.82', '42.219.159.92', '42.219.159.94', '42.219.158.226']
# ten_ips = ['42.219.153.7', '42.219.153.89', '42.219.158.226']
baseline_choice = 'baseline1'

# vali_one(baseline_choice, ten_ips[9], ten_ips[9])
# for ips in ten_ips:
#     vali_avg(baseline_choice, ips, ten_ips)

# input_raw_ip_str = '42.219.158.226'
# input_gen_ip_str = '42.219.158.226'

for ips in ten_ips:
    for ip2 in ten_ips:
        vali_one(baseline_choice, ips, ip2)

# for ips in ten_ips:
#     vali_one(baseline_choice, ips, 'ten_ips')