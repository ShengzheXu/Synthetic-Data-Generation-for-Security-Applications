import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score
import configparser
from utils.config_utils import recieve_cmd_config
from utils.plot_utils import boxplot
from models.baselines import baseline1

def KLc(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def loglikelihood(p):
    meta_model = baseline1()
    meta_model.load_params(meta_model.byt_model)

    logprob, responsibilities = meta_model.byt_model.score_samples(p)
    pdf = np.exp(logprob)
    return sum(logprob)

def cross_validation_preparation(data):
    k_fold = 5
    csvfile = open('import_1458922827.csv', 'r').readlines()
    filename = 1
    len_rows = int(len(csvfile) / float(k_fold) * k_fold)
    block_rows = len_rows / k_fold
    for i in range(k_fold):
        if i % block_rows == 0:
            open(str(filename) + 'import_1458922827.csv', 'w').writelines(csvfile[i:i+block_rows])
            filename += 1
    # build 5 model
    for i in range(k_fold):
        # open test data and test loglikelihood
        test_data = None
        loglikelihood(test_data)

working_folder = ''

def vali_one(raw_ip_str, gen_ip_str):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)

    # values1 = np.asarray(list(rawdata.byt))
    # values2 = np.asarray(list(gendata.byt))
    values1 = list(rawdata.byt)
    values2 = list(gendata.byt)

    max_value_1 = max(values1)
    max_value_2 = max(values2)
    maxn = max(max_value_1, max_value_2)
    
    eps = 0.01
    bins = np.linspace(0, maxn+eps, 10)
    print(bins)
    # if max_value_1 > max_value_2:
    #     _, bins = pd.qcut(values1, 10, retbins=True, duplicates='drop')
    # else:
    #     _, bins = pd.qcut(values2, 10, retbins=True, duplicates='drop')
    # print(count_unique_value_raw, count_unique_value_gen, len(bins))

    cats1 = pd.cut(values1, bins)
    pr1 = cats1.value_counts()/cats1.size
    # print(pr1)
    print("======>")
    print(cats1.value_counts())
    print("======>")
    # print(bins)
    # print(type(bins))
    pr1 = list(pr1)

    cats2 = pd.cut(values2, bins)
    pr2 = (cats2.value_counts()+1)/(cats2.size+1)
    # print(pr2)
    print("<======")
    print(cats2.value_counts())
    print("<======")
    pr2 = list(pr2)

    # print(pr1+pr2)

    KL = scipy.stats.entropy(pr1, pr2)
    # NMI = normalized_mutual_info_score(pr2,pr2)
    print(raw_ip_str + ',' + gen_ip_str + ','+ str(KL)) #, ",NMI", NMI)
    return KL


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    # override the config with the command line
    recieve_cmd_config(config['DEFAULT'])
    
    user_list = config['DEFAULT']['userlist'].split(',')
    test_list = config['VALIDATE']['test_set'].split(',')
    working_folder = config['DEFAULT']['working_folder']

    # user_list.reverse()
    if config['VALIDATE']['raw_compare'] == 'True':
        x_data = []
        y_data = []
        for ip2 in user_list:
            x_data.append(ip2)
            y_data_i = []
            for ip1 in user_list:
                if ip1 == ip2:
                    continue
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'raw_data/day_1_%s.csv' % ip2
                y_data_i.append(vali_one(src_file, des_file))
            y_data.append(y_data_i)
            
        boxplot(x_data, y_data, title='KL of other 9 raw ips || 1 raw_ip')

    if config['VALIDATE']['gen_compare'] == 'True':
        x_data = []
        y_data = []
        for ip2 in test_list:
            x_data.append(ip2)
            y_data_i = []
            for ip1 in user_list: 
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'gen_data/%s.csv' % ip2
                y_data_i.append(vali_one(src_file, des_file))
            y_data.append(y_data_i)
        
        # x_data.append('10 users 1 day to their own raw')
        # y_data_i = []
        # for ip3 in user_list:
        #     src_file = 'raw_data/day_1_%s.csv' % ip3
        #     des_file = 'gen_data/multi_users/baseline2_1days_%s.csv' % ip3
        #     y_data_i.append(vali_one(src_file, des_file))
        # y_data.append(y_data_i)
        
        boxplot(x_data, y_data, title='KL of 10 raw ips || gen baselines')
