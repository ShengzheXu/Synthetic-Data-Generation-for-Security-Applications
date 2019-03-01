import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score

def KLc(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

raw_ip_str = '42.219.158.226'
gen_ip_str = '1IP'

raw_ip_str = 'sampled_10IPs'
gen_ip_str = '10IP'

rawdata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % raw_ip_str)
# gendata = pd.read_csv('./../data/baseline1&2/raw_data/day_1_%s.csv' % raw_ip_str)
gendata = pd.read_csv('./../data/baseline1&2/gen_data/baseline1_%s.csv' % gen_ip_str)

values1 = np.log(list(rawdata.byt))
values2 = np.log(list(gendata.byt))

cats1, bins = pd.qcut(values1, 10, retbins=True)
pr1 = cats1.value_counts()/cats1.size
print(pr1)
print("======>")
# print(bins)
# print(type(bins))
pr1 = list(pr1)

cats2 = pd.cut(values2, bins)
pr2 = cats2.value_counts()/cats2.size
print(pr2)
pr2 = list(pr2)

KL = scipy.stats.entropy(pr1, pr2)
NMI = normalized_mutual_info_score(pr2,pr2)
print("KL", KL, ",NMI", NMI)