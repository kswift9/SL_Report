import os, shutil, traceback
import kagglehub
import pandas as pd
import itertools
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

class SL_Report_Data():
    def __init__(
            self,
            root='./', 
            demo_size=0.1, 
            cancer_source='zahidmughal2343/global-cancer-patients-2015-2024',
            bankruptcy_source='fedesoriano/company-bankruptcy-prediction'
            ):
        '''
        Inputs:
        - root: home directory (default ./)
        - demo_size: portion of full data to save as demo (default 0.1)
        - cancer_source: kaggle source for cancer data
        - bankruptcy_source: kaggle source for bankruptcy data
        '''
        self.root = root
        self.demo_size = demo_size
        self.cancer_source = cancer_source
        self.bankruptcy_source = bankruptcy_source
        self.latest_loaded = None  # latest data source loaded

    @staticmethod
    def copy_csvs(source_dir, dest_path):
        '''
        Copy only csvs with shutil
        '''
        for path, subpaths, filenames in os.walk(source_dir):
            for filename in filenames:
                if filename.endswith('.csv'):
                    source_path = os.path.join(path, filename)
                    shutil.copy(source_path, dest_path)
        return dest_path

    def download_kaggle_dataset(self, data_dir, save_path):
        '''
        Use the kaggle API to download a dataset
        '''
        tmp_path = kagglehub.dataset_download(data_dir)
        moved_to = self.copy_csvs(tmp_path, save_path)
        return moved_to

    def set_up_data(self):
        '''
        One-stop-shop for setting up our data from kaggle.
        
        Must have kaggle.json saved to the required directory, per kaggle's documentation: 
        - https://www.kaggle.com/docs/api
        '''

        # set up data folders
        if not os.path.exists(self.root+'data/'):
            os.mkdir(self.root+'data/')
            os.mkdir(self.root+'data/cancer/')
            os.mkdir(self.root+'data/cancer/full/')
            os.mkdir(self.root+'data/cancer/demo/')
            os.mkdir(self.root+'data/bankruptcy/')
            os.mkdir(self.root+'data/bankruptcy/full/')
            os.mkdir(self.root+'data/bankruptcy/demo/')

        # set up log folders
        if not os.path.exists(self.root+'logs/'):
            os.mkdir(self.root+'logs/')

        # download the cancer data
        if not os.path.exists(self.root+'data/cancer/full/data.csv'):
            self.download_kaggle_dataset(
                'zahidmughal2343/global-cancer-patients-2015-2024',
                self.root+'data/cancer/full/data.csv'
                )
            print('Cancer Dataset Downloaded')
        else:
            print('Already have Cancer Dataset')
        
        # # set up cancer demo
        if not os.path.exists(self.root+'data/cancer/demo/data.csv'):
            sample_df = pd.read_csv(self.root+'data/cancer/full/data.csv')
            sample_idx = sample_df.sample(round(sample_df.shape[0]*self.demo_size)).index
            sample_df.loc[sample_idx].reset_index().to_csv(self.root+'data/cancer/demo/data.csv')
        print('Cancer Demo Saved')

        # download the bankruptcy data
        if not os.path.exists(self.root+'data/bankruptcy/full/data.csv'):
            self.download_kaggle_dataset(
                'fedesoriano/company-bankruptcy-prediction',
                self.root+'data/bankruptcy/full/data.csv'
                )
            print('Bankruptcy Dataset Downloaded')
        else:
            print('Already Have Bankruptcy Dataset')

        # set up bankruptcy demo
        # samle each label evenly 
        if not os.path.exists(self.root+'data/bankruptcy/demo/data.csv'):
            sample_df = pd.read_csv(self.root+'data/bankruptcy/full/data.csv')
            sample_1s = sample_df[sample_df['Bankrupt?']==1].reset_index()
            sample_1s_idx = sample_1s.sample(round(sample_1s.shape[0]*self.demo_size)).index
            sample_1s = sample_1s.loc[sample_1s_idx]
            sample_0s = sample_df[sample_df['Bankrupt?']==0].reset_index()
            sample_0s_idx = sample_0s.sample(round(sample_0s.shape[0]*self.demo_size)).index
            sample_0s = sample_0s.loc[sample_0s_idx]
            sample_df = pd.concat([sample_1s, sample_0s]).reset_index()
            sample_df.to_csv(self.root+'data/bankruptcy/demo/data.csv')
        print('Bankruptcy Demo Saved')

    @staticmethod
    def remove_bad_cols(data):
        try:
            data = data.drop(['Unnamed: 0', 'index'], axis=1)
            return data
        except:
            return data

    def get_cancer_full(self):
        self.latest_loaded = 'full_cancer'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/cancer/full/data.csv'))

    def get_cancer_demo(self):
        self.latest_loaded = 'demo_cancer'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/cancer/demo/data.csv'))

    def get_bankruptcy_full(self):
        self.latest_loaded = 'full_bankruptcy'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/bankruptcy/full/data.csv'))

    def get_bankruptcy_demo(self):
        self.latest_loaded = 'demo_bankruptcy'
        return self.remove_bad_cols(pd.read_csv(self.root+'data/bankruptcy/demo/data.csv'))


class Gridsearch_Optimization():

    def __init__(self, rangedict, estimator, metric, X, y, holdout_X, holdout_y, holdout_metric, want_low, invert_metric=False):
        self.rangedict = rangedict
        self.estimator = estimator
        self.metric = metric
        self.holdout_metric = holdout_metric
        self.metric_mult = -1 if invert_metric else 1
        self.results = None
        self.best_result = None
        self.X = X
        self.y = y
        self.holdout_X = holdout_X
        self.holdout_y = holdout_y
        self.want_low = want_low

    @staticmethod
    def rangedict_to_tups(kv_map):
        # put non-iterable or string in list
        keys = sorted(list(kv_map.keys()))
        for key in keys:
            if not hasattr(kv_map[key], '__iter__') or isinstance(kv_map[key], str):
                kv_map.update({key: [kv_map[key]]})
        # itertools o'clock
        return keys, itertools.product(*[kv_map[key] for key in keys])

    @staticmethod
    def tup_to_entitydict(keys, tup):
        result = {}
        for i, key in enumerate(keys):
            result.update({key: tup[i]})
        return result

    def convert_rangedict_to_paramgrid(self):
        keys, tups = self.rangedict_to_tups(self.rangedict)
        return [self.tup_to_entitydict(keys, tup) for tup in tups]

    def find_best_params(self, log_as=None):
        self.paramgrid = self.convert_rangedict_to_paramgrid()
        self.gridlen = len(self.paramgrid)
        self.results = [None] * self.gridlen

        for i in tqdm(range(self.gridlen), desc='Optimizing', total=self.gridlen):
            cv_results = cross_validate(
                self.estimator(**self.paramgrid[i]),
                self.X,
                self.y,
                cv=3,  # hard coded, I know it's bad practice
                return_train_score=True,
                return_estimator=True,
                scoring=self.metric
            )
            cv_results.update({
                'index': i,
                'params': self.paramgrid[i],
                'holdout_score': np.mean([self.holdout_metric(self.holdout_y, estimator.predict(self.holdout_X)) for estimator in cv_results['estimator']]),
                'test_score': self.metric_mult * np.mean(cv_results['test_score']),
                'train_score': self.metric_mult * np.mean(cv_results['train_score']),
                'fit_time': np.mean(cv_results['fit_time']),
                'score_time': np.mean(cv_results['score_time'])
            })
            cv_results.pop('estimator')  # storage cost
            self.results[i] = cv_results
        if self.want_low:
            self.best_result = sorted(self.results, key=lambda cv_results: cv_results['test_score'])[0]
        else:
            self.best_result = sorted(self.results, key=lambda cv_results: cv_results['test_score'])[-1]
        if log_as:
            with open(log_as, 'w+') as file:
                json.dump(self.results, file)
        return

    def plot_validation_and_complexity_curves(self, title):
        params = [p for p in self.best_result['params'].keys() if len(list(self.rangedict[p]))>1 and not isinstance(self.rangedict[p], str)]  # don't check singular params
        for p in params:
            # gather viz data
            viz_data = {
                p: [],
                'train_score': [],
                'test_score': [],
                'holdout_score': [],
                'fit_time': [],
                'score_time': []
            }
            for result in self.results:
                if all([result['params'][pp]==self.best_result['params'][pp] for pp in result['params'].keys() if pp!=p]):
                    viz_data[p].append(result['params'][p])
                    for kk in [k for k in viz_data.keys() if k!=p]: 
                        viz_data[kk].append(result[kk])

            # tuple parameters - nn layer shapes
            if p == 'hidden_layer_sizes':
                    viz_data[p] = [str(seq) for seq in viz_data[p]]
                    self.plot_str_bars(viz_data, p, title)
                # viz_data.update({
                #     'depth': [len(seq) for seq in viz_data[p]],
                #     'next_layer_multiplier': [round(seq[1]/seq[0], 2) if len(seq)>1 else 1 for seq in viz_data[p]]
                # })
                # viz_data.pop(p)  # drop the offending tuple

                # ##########################
                # # 3D Plotting - fancy!
                # ##########################

                # # setup
                # xy_keys = ['depth', 'next_layer_multiplier']
                # markers = list(Line2D.markers.keys())[2:]

                # # validation curve
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # for i, key in enumerate([k for k in viz_data.keys() if (k not in xy_keys and 'time' not in k)]):
                #     ax.scatter(
                #         viz_data['depth'], 
                #         viz_data['next_layer_multiplier'], 
                #         viz_data[key], 
                #         label=key, 
                #         alpha=0.7, 
                #         marker=markers[i]
                #     )
                # ax.set_xlabel('Depth')
                # ax.set_ylabel('Next Layer Multiplier')
                # ax.set_zlabel(self.holdout_metric.__name__)
                # ax.set_xticks(viz_data['depth'])
                # ax.set_yticks(viz_data['next_layer_multiplier'])
                # ax.set_box_aspect((3,4,3), zoom=0.9)  # z-label cut off
                # plt.legend()
                # plt.title('Performance by Network Shape')
                # plt.show()

                # # complexity curve
                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                # for i, key in enumerate([k for k in viz_data.keys() if (k not in xy_keys and 'time' in k)]):
                #     ax.scatter(
                #         viz_data['depth'], 
                #         viz_data['next_layer_multiplier'], 
                #         viz_data[key], 
                #         label=key, 
                #         alpha=0.7, 
                #         marker=markers[i]
                #     )
                # ax.set_xlabel('Depth')
                # ax.set_ylabel('Next Layer Multiplier')
                # ax.set_zlabel(self.holdout_metric.__name__)
                # ax.set_xticks(viz_data['depth'])
                # ax.set_yticks(viz_data['next_layer_multiplier'])
                # ax.set_box_aspect((3,4,3), zoom=0.9)  # z-label cut off
                # plt.legend()
                # plt.title('Time Complexity by Network Shape')
                # plt.show()

                # ##########################
                # # Done being fancy
                # ##########################
            
            else:
                # ensure viz data sorted by p
                for k in viz_data.keys():
                    if k != p:
                        viz_data[k] = [x for _, x in sorted(zip(viz_data[p], viz_data[k]), key=lambda pair: pair[0])]
                viz_data[p] = sorted(viz_data[p])
                # numeric parameter
                if not isinstance(viz_data[p][0], str):  
                   self.plot_numeric_curves(viz_data, p, title)
                # string parameter
                else:
                    self.plot_str_bars(viz_data, p, title)
        return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        del self  # don't bloat local memory - this is why we log!
        return
    
    def plot_numeric_curves(self, viz_data, p, title):
        for key in [k for k in viz_data.keys() if (k!=p and 'time' not in k)]:
            plt.plot(viz_data[p], viz_data[key], 'o-', label=key, alpha=0.7)
        plt.legend()
        plt.xlabel(f'{p} Value')
        plt.xscale('log')
        plt.ylabel(self.holdout_metric.__name__)
        plt.title(f'{title} Performance by "{p}"')
        plt.show()


        # complexity curve
        for key in [k for k in viz_data.keys() if (k!=p and 'time' in k)]:
            plt.plot(viz_data[p], viz_data[key], 'o-', label=key, alpha=0.7)
        plt.legend()
        plt.xlabel(f'{p} Value')
        plt.xscale('log')
        plt.ylabel('Time (sec)')
        plt.title(f'{title} Time Complexity by "{p}"')
        plt.show()

    def plot_str_bars(self, viz_data, p, title):
        # validation curve
        w, pivot = 0.15, np.arange(len(viz_data[p]))
        n_cats = len([k for k in viz_data.keys() if k!=p and 'time' not in k])
        plot_at = [pivot + i*w for i in range(n_cats)]
        for i, key in enumerate([k for k in viz_data.keys() if k!=p and 'time' not in k]):
            plt.bar(plot_at[i], viz_data[key], width=w, label=key, align='edge')
        plt.xticks(pivot+(w*n_cats/2), labels=viz_data[p], rotation=45, ha='right')
        plt.legend()
        plt.xlabel(f'{p} Value')
        plt.ylabel(self.holdout_metric.__name__)
        plt.title(f'{title} Performance by "{p}"')
        plt.show()

        # complexity curve
        w, pivot = 0.15, np.arange(len(viz_data[p]))
        n_cats = len([k for k in viz_data.keys() if k!=p and 'time' in k])
        plot_at = [pivot + i*w for i in range(n_cats)]
        for i, key in enumerate([k for k in viz_data.keys() if k!=p and 'time' in k]):
            plt.bar(plot_at[i], viz_data[key], width=w, label=key, align='edge')
        plt.xticks(pivot+(w*n_cats/2), labels=viz_data[p], rotation=45, ha='right')
        plt.legend()
        plt.xlabel(f'{p} Value')
        plt.ylabel('Time (sec)')
        plt.title(f'{title} Time Complexity by "{p}"')
        plt.show()


if __name__=='__main__':
    datasource = SL_Report_Data()
    datasource.set_up_data()