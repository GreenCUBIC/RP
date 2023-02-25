import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from kneed import KneeLocator
import seaborn as sns
import matplotlib.pyplot as plt

class RPAB(object):

    def __init__(self,df,elementA,elementB,score,truth=None,progress=False,baseline_method='knee',ranking_method='first'):
        self.df = df.copy()
        self.elementA = elementA
        self.elementB = elementB
        self.score = score
        self.truth = truth

        self.elementA_rank = '{}_rank'.format(elementA)
        self.elementB_rank = '{}_rank'.format(elementB)
        self.elementA_base = '{}_base'.format(elementA)
        self.elementB_base = '{}_base'.format(elementB)
        self.elementA_base_rank = '{}_base_rank'.format(elementA)
        self.elementB_base_rank = '{}_base_rank'.format(elementB)
        self.elementA_mean = '{}_mean'.format(elementA)
        self.elementB_mean = '{}_mean'.format(elementB)
        self.elementA_std = '{}_std'.format(elementA)
        self.elementB_std = '{}_std'.format(elementB)
        self.elementA_size = '{}_size'.format(elementA)
        self.elementB_size = '{}_size'.format(elementB)

        self.columns= [score,
                  '{}_in_{}_perc'.format(elementB,elementA),'{}_in_{}_perc'.format(elementA,elementB),
                  'ARRO',
                  '{}_base'.format(elementA),'{}_base_perc'.format(elementA),'{}_base'.format(elementB),'{}_base_perc'.format(elementB),
                  'fdp_{}_base'.format(elementA),'fdp_{}_base'.format(elementB),'fd_{}_base'.format(elementA),'fd_{}_base'.format(elementB),
                  'std_{}_dist'.format(elementA),'std_{}_dist'.format(elementB)]
        self.group_by_element()
        self.add_ranks(ranking_method)
        self.calculate_baselines(default_methods[baseline_method],progress)

    def group_by_element(self):
        """ Create group objects for use in other functions needing one2all scores
            RETURNS: elementA_group,elementB_group
        """
        self.elementA_group = self.df.groupby(level=self.elementA)
        self.elementB_group = self.df.groupby(level=self.elementB)
        self.df[self.elementA_mean] = self.elementA_group[self.score].transform('mean').astype('category')
        self.df[self.elementA_std] = self.elementA_group[self.score].transform('std').astype('category')
        self.df[self.elementA_size] = self.elementA_group[self.score].transform('size').astype('category')
        self.df[self.elementB_mean] = self.elementB_group[self.score].transform('mean').astype('category')
        self.df[self.elementB_std] = self.elementB_group[self.score].transform('std').astype('category')
        self.df[self.elementB_size] = self.elementB_group[self.score].transform('size').astype('category')

    def add_ranks(self,method):
        print('Ranking {} groups...'.format(self.elementA))
        elementA_values = self.elementA_group[self.score].rank(method=method,ascending=False,pct=True)
        print('Ranking {} groups...'.format(self.elementB))
        elementB_values = self.elementB_group[self.score].rank(method=method,ascending=False,pct=True)
        self.df[self.elementA_rank] = elementA_values
        self.df[self.elementB_rank] = elementB_values

    def calculate_baselines(self,baseline_func,progress):
        elementA_baseline_func = lambda A: baseline_func(A[self.elementA_rank].values,A[self.score].values)
        elementB_baseline_func = lambda B: baseline_func(B[self.elementB_rank].values,B[self.score].values)
        if progress:
            tqdm().pandas()
            print('Calculating baselines for {} groups...'.format(self.elementA))
            elementA_baselines = self.elementA_group.progress_apply(elementA_baseline_func)
            print('Calculating baselines for {} groups...'.format(self.elementB))
            elementB_baselines = self.elementB_group.progress_apply(elementB_baseline_func)
        else:
            print('Calculating baselines for {} groups...'.format(self.elementA))
            elementA_baselines = self.elementA_group.apply(elementA_baseline_func)
            print('Calculating baselines for {} groups...'.format(self.elementB))
            elementB_baselines = self.elementB_group.apply(elementB_baseline_func)
        self.df = self.df.join(pd.DataFrame.from_records(list(elementA_baselines.values),index=elementA_baselines.index,columns=[self.elementA_base,self.elementA_base_rank]),on=self.elementA)\
                         .join(pd.DataFrame.from_records(list(elementB_baselines.values),index=elementB_baselines.index,columns=[self.elementB_base,self.elementB_base_rank]),on=self.elementB)
        self.df.astype({self.elementA_base:'category',self.elementA_base_rank:'category',self.elementB_base:'category',self.elementB_base_rank:'category'},copy=False)

    def percentile_features(self,dtype='float64'):
        elementA_base = self.df[self.elementA_base].astype(dtype).values
        elementB_base = self.df[self.elementB_base].astype(dtype).values
        elementA_base_perc = self.df[self.elementA_base_rank].astype(dtype).values
        elementB_base_perc = self.df[self.elementB_base_rank].astype(dtype).values
        score = self.df[self.score].astype(dtype).values
        elementB_in_elementA_perc = self.df[self.elementA_rank].astype(dtype).values
        elementA_in_elementB_perc = self.df[self.elementB_rank].astype(dtype).values
        elementA_size = self.df[self.elementA_size].astype(dtype).values
        elementB_size = self.df[self.elementB_size].astype(dtype).values
        elementA_mean = self.df[self.elementA_mean].astype(dtype).values
        elementB_mean = self.df[self.elementB_mean].astype(dtype).values
        elementA_std = self.df[self.elementA_std].astype(dtype).values
        elementB_std = self.df[self.elementB_std].astype(dtype).values

        # Calculate Reciprocal Rank Order Metrics
        ARRO = 1. / (elementB_in_elementA_perc * elementA_in_elementB_perc)  # product of the perc order values, inverted

        # Determine the features based on baseline method
        fdp_elementA_base = elementA_base_perc - elementB_in_elementA_perc
        fdp_elementB_base = elementB_base_perc - elementA_in_elementB_perc

        # Calculate the Fold-Difference features
        fd_elementA_base = (score - elementA_base) / elementA_base
        fd_elementB_base = (score - elementB_base) / elementB_base

        # Calculate the Standard Deviations Away from baseline features: (ES-mean)/std
        std_elementA_dist = (score - elementA_mean) / (elementA_std)
        std_elementB_dist = (score - elementB_mean) / (elementB_std)

        feature_df = np.array([score,
                      elementB_in_elementA_perc,elementA_in_elementB_perc,
                      ARRO,
                      elementA_base,elementA_base_perc,elementB_base,elementB_base_perc,
                      fdp_elementA_base,fdp_elementB_base,fd_elementA_base,fd_elementB_base,
                      std_elementA_dist,std_elementB_dist]).T
        feature_df = pd.DataFrame(feature_df,self.df.index,columns=self.columns)
        if self.truth is not None:
            feature_df[self.truth] = self.df[self.truth]
        return feature_df

    def plot_pairs(self, A, B,include_baseline=True,include_truth=False,show_plot=True):
        A_group = self.elementA_group.get_group(A).copy()
        B_group = self.elementB_group.get_group(B).copy()
        A_group['highlight'] = False
        B_group['highlight'] = False
        try:
            A_group.loc[(A,B), 'highlight'] = True
        except:
            print('{} is not contained in the {} group.'.format(self.elementA,self.elementB))
            return
        B_group.loc[A, 'highlight'] = True
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 7.5))
        sns.scatterplot(x=self.elementA_rank, y=self.score, data=A_group, ax=ax1)
        sns.scatterplot(x=self.elementB_rank, y=self.score, data=B_group, ax=ax2)
        if include_truth and self.truth is not None:
            sns.scatterplot(x=self.elementA_rank, y=self.score, data=A_group, ax=ax1,hue=self.truth)
            sns.scatterplot(x=self.elementB_rank, y=self.score, data=B_group, ax=ax2,hue=self.truth)
            plt.legend()
        if include_baseline:
            ax1.axhline(self.df.loc[(A,B),self.elementA_base])
            ax2.axhline(self.df.loc[(A,B),self.elementB_base])
        sns.scatterplot(x=self.elementA_rank, y=self.score, data=A_group.loc[A_group['highlight']], ax=ax1, s=90,color='y')
        sns.scatterplot(x=self.elementB_rank, y=self.score, data=B_group.loc[B_group['highlight']], ax=ax2, s=90,color='y')
        ax1.set(title='{}'.format(self.elementA))
        ax2.set(title='{}'.format(self.elementB))
        if show_plot:
            plt.show()
            return
        else:
            return fig

#Baseline functions
global default_methods

def get_percentile(ranks,scores,percentile=0.5):
    index = np.abs(ranks-percentile).argmin()
    return scores[index],percentile

def find_knee(ranks,scores,sensitivity=5,threshold=100,fallback=0.5):
    if len(ranks)>threshold:
        index = ranks.argsort()
        ranks = ranks[index]
        scores = scores[index]
        try:
            kn = KneeLocator(ranks, scores, interp_method='polynomial',curve='convex', direction='decreasing',S=sensitivity)
        except RuntimeWarning:
            return np.nan,np.nan
        if kn.knee:
            index = np.where(kn.knee==ranks)[0]
            return scores[index][0],kn.knee
        else:
            return np.nan,np.nan
    else:
        return get_percentile(ranks,scores,fallback)

default_methods = {'percentile':get_percentile,'knee':find_knee}
