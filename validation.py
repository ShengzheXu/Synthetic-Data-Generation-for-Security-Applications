import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics.cluster import normalized_mutual_info_score
import configparser
from utils.config_utils import recieve_cmd_config
from utils.plot_utils import boxplot
from models.baselines import baseline1
from utils.distribution_utils import get_distribution_with_laplace_smoothing

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

def real_vali_KL(rawdata, gendata, bins):
    values1 = list(np.log(rawdata.byt))
    values2 = list(np.log(gendata.byt))

    cats1 = pd.cut(values1, bins)
    pr1 = list(cats1.value_counts())
    cats2 = pd.cut(values2, bins)
    pr2 = list(cats2.value_counts())

    pk = get_distribution_with_laplace_smoothing(pr1)
    qk = get_distribution_with_laplace_smoothing(pr2)

    # If only probabilities pk are given, the entropy is calculated as S = -sum(pk * log(pk), axis=0).
    # If qk is not None, then compute the Kullback-Leibler divergence S = sum(pk * log(pk / qk), axis=0).
    KL = scipy.stats.entropy(pk, qk)

    # myKL = KLc(pk, qk)
    # NMI = normalized_mutual_info_score(pr2,pr2)

    return KL

def vali_one(raw_ip_str, gen_ip_str, bins):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)

    KL = real_vali_KL(rawdata, gendata, bins)

    print(','.join([raw_ip_str, gen_ip_str, str(KL)]))
    return KL
    

def vali_hourly(raw_ip_str, gen_ip_str, bins):
    rawdata = pd.read_csv(working_folder + raw_ip_str)
    gendata = pd.read_csv(working_folder + gen_ip_str)
    
    rtn = []
    for T in range(24):
        hour_str = '00'
        if T<10:
            hour_str = hour_str[:-1] + str(T)[0]
        else:
            hour_str = hour_str[:-2] + str(T)[:1]
        
        raw_chunk = rawdata[rawdata['te'].str.contains(' '+hour_str+':')]
        gen_chunk = gendata[gendata['te'].str.contains(' '+hour_str+':')]
        KL = real_vali_KL(raw_chunk, gen_chunk, bins)
        print(','.join(['For %d hour' % T, raw_ip_str, gen_ip_str, str(KL)]))
        rtn.append(KL)

    return rtn
        


if __name__ == "__main__":
    # load in the configs
    config = configparser.ConfigParser()
    config.read('config.ini')
    # override the config with the command line
    recieve_cmd_config(config['DEFAULT'])
    
    user_list = config['DEFAULT']['userlist'].split(',')
    working_folder = config['DEFAULT']['working_folder']
    bins = list(map(float, config['DEFAULT']['bins'].split(',')))
    
    test_list = config['VALIDATE']['test_set'].split(',')

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
                y_data_i.append(vali_one(src_file, des_file, bins))
            y_data.append(y_data_i)
            
        boxplot(x_data, y_data, title='KL(99 other raw ips || 1 raw_ip)')

    if config['VALIDATE']['gen_compare'] == 'True':
        x_data = []
        y_data = []
        for ip2 in test_list:
            x_data.append(ip2)
            y_data_i = []
            for ip1 in user_list: 
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'gen_data/%s.csv' % ip2
                y_data_i.append(vali_one(src_file, des_file, bins))
            y_data.append(y_data_i)
        
        # x_data.append('10 users 1 day to their own raw')
        # y_data_i = []
        # for ip3 in user_list:
        #     src_file = 'raw_data/day_1_%s.csv' % ip3
        #     des_file = 'gen_data/multi_users/baseline2_1days_%s.csv' % ip3
        #     y_data_i.append(vali_one(src_file, des_file))
        # y_data.append(y_data_i)
        
        boxplot(x_data, y_data, title='KL(10 raw ips || gen baseline)')
    
    if config['VALIDATE']['hour_compare'] == 'True':
        for ip2 in test_list:
            x_data = []
            y_data = []
            for h in range(24):
                x_data.append(h)
                y_data.append([])
            
            for ip1 in user_list: 
                src_file = 'raw_data/day_1_%s.csv' % ip1
                des_file = 'gen_data/%s.csv' % ip2
                y_data_i = vali_hourly(src_file, des_file, bins)
                for h in range(24):
                    y_data[h].append(y_data_i[h])
            print(len(y_data), len(y_data[0]))
        
            boxplot(x_data, y_data, title='KL(10 raw ips || %s)' % ip2)
