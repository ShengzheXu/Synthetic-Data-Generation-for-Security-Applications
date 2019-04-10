import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score
import configparser
from utils.config_utils import recieve_cmd_config

def KLc(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def vali_one(raw_ip_str, gen_ip_str):
    rawdata = pd.read_csv('./data/baseline1&2/%s' % raw_ip_str)
    gendata = pd.read_csv('./data/baseline1&2/%s' % gen_ip_str)
    
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
    print(raw_ip_str + ',' + gen_ip_str + ','+ str(KL)) #, ",NMI", NMI)


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    user_list = config['DEFAULT']['userlist'].split(',')
    baseline_choice = config['DEFAULT']['baseline']
    # user_list = ['42.219.153.7', '42.219.153.89', '42.219.158.226']
    test_list = config['VALIDATE']['test_set'].split(',')
    
    # override the config with the command line
    recieve_cmd_config(config['DEFAULT'])
    print(test_list)

    if config['VALIDATE']['raw_compare'] == 'True':
        for ip1 in user_list:
            for ip2 in user_list:
                if ip1 == ip2:
                    continue
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'raw_data/day_1_%s.csv' % ip2
                vali_one(src_file, des_file)

    if config['VALIDATE']['gen_compare'] == 'True':
        for ip1 in user_list:
            for ip2 in test_list: 
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'gen_data/%s.csv' % ip2
                vali_one(src_file, des_file)