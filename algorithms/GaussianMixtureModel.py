import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture 

class SourceData(object):
    def __init__(self, source_data):
        self.all_record = pd.read_csv(source_data)
        
    def plot_source_distribution(self):
        plt.style.use('bmh')
        colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
                '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

        fig = plt.figure(figsize=(11,3))
        _ = plt.title('Frequency of records by number of bytes')
        _ = plt.xlabel('number of bytes')
        _ = plt.ylabel('Number of records')
        _ = plt.hist(self.all_record['byt'].values, 
                    range=[0, 5000], bins=100, histtype='stepfilled')

        fig.savefig('T00_data_dis')

class GMMs(object):
    def __init__(self, data_col, model='normal'):
        X_train = np.reshape(data_col.values, (-1, 1))
        if model == 'cross-valid':
            self.cross_validation(X_train)
        else:
            self.fit(X_train)
    
    def fit(self, X_train):
        self.clf = mixture.GaussianMixture(n_components=7, covariance_type='full')
        X_train = np.log(X_train)

        self.clf.fit(X_train)
        x_min = np.min(X_train)
        x_max = np.max(X_train)
        x_len = len(X_train)
        print(np.min(X_train), np.max(X_train), len(X_train))
        x = np.linspace(x_min, x_max, x_len/100)
        # x = np.linspace(0, 5000, 5000)

        # Plot the data to which the GMM is being fitted
        x = np.reshape(x, (-1, 1))
        logprob = self.clf.score_samples(x) # log-probabilities
        pdf = np.exp(logprob)

        resp = self.clf.predict_proba(x) # to predict posterior probabilities/ responsibilities
        pdf_individual = resp * pdf[:, np.newaxis]

        # fig = plt.figure()
        # _ = plt.title('estimated pdf of 42.219.153.89 day 1')
        # _ = plt.xlabel('number of bytes (log)')
        # _ = plt.ylabel('probability')
        # _ = plt.hist(data_col.values, density = 1,
        #         range=[x_min, x_max], bins=1, histtype='stepfilled')
        # _ = plt.plot(x, pdf, color='blue')
        # _ = plt.plot(x, pdf_individual, '--k')
        # fig.savefig('allT_gmm_pdf')


    def trunc(self, arr, start, end):
        # print(arr, start, end)
        sample = arr[start:end]
        remainder = np.delete(arr, np.s_[start:end], axis=0)
        return sample, remainder

    def cross_validation(self, X_train):
        X_train = np.log(X_train)
        # candis_len = np.split(X_train, 5, axis=0)
        candis_len = int(len(X_train) / 5)
        for i in range(1, 10):
            print("n_comp:", i)
            rst = []
            start = 0
            clf = mixture.GaussianMixture(n_components=i, covariance_type='full')
            for j in range(0, 5):
                print(len(X_train), j, start, start+candis_len)
                tester, trainer = self.trunc(X_train, start, start+candis_len)
                clf.fit(trainer)
                print('trainertest:', len(trainer), len(tester))
                rst.append(clf.score(tester))            
                start += candis_len
            print(rst)
            # print(','.join(rst))

if __name__ == "__main__":
    a = SourceData('./../data/output/T_day_1_42.219.153.89.csv')
    # a.plot_source_distribution()
    # b = GMMs(a.all_record['byt'], model='cross-valid')
    b = GMMs(a.all_record['byt'], model='normal')
