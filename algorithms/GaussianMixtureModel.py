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
    def __init__(self, data_col, t_str, model='normal'):
        X_train = np.reshape(data_col.values, (-1, 1))
        if model == 'AIC':
            self.build_models(X_train)
        else:
            clf = mixture.GaussianMixture(n_components=7, covariance_type='full')
            clf.fit(X_train)

            # x = np.linspace(np.min(X_train), np.max(X_train), len(X_train))
            x = np.linspace(0, 5000, 5000)

            # Plot the data to which the GMM is being fitted
            x = np.reshape(x, (-1, 1))
            logprob = clf.score_samples(x) # log-probabilities
            pdf = np.exp(logprob)

            resp = clf.predict_proba(x) # to predict posterior probabilities/ responsibilities
            pdf_individual = resp * pdf[:, np.newaxis]

            fig = plt.figure()
            _ = plt.title('estimated pdf of 42.219.153.89 day 1, T=%s' % t_str)
            _ = plt.xlabel('number of bytes')
            _ = plt.ylabel('Number of records')
            _ = plt.hist(data_col.values, density = 1,
                    range=[0, 5000], bins=100, histtype='stepfilled')
            _ = plt.plot(x, pdf, color='blue')
            _ = plt.plot(x, pdf_individual, '--k')
            fig.savefig('rect_timediv/T%s_gmm_pdf' % t_str)
    
    def build_models(self):
        n_components = np.arange(1, 21)
        models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X_train)
          for n in n_components]

if __name__ == "__main__":
    # a = SourceData('./../data/output/T00_day_1_42.219.153.89.csv')
    # a.plot_source_distribution()
    t_str = '11'
    for t in range(0, 24):
        t_str = str(int(t/10)) + str(t%10)
        a = SourceData('./../data/day_1_42.219.153.89/rectified_hour_division/T%s_day_1_42.219.153.89.csv' % t_str)
        # a = SourceData('./../data/day_1_42.219.153.89/day_1_42.219.153.89.csv')
    
        b = GMMs(a.all_record['byt'], t_str)