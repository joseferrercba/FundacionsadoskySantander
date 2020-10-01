from re import error
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from enum import Enum, unique

@unique
class ResamplerEnum(Enum):
    SMOTE = 1
    AllKNN = 2,
    SMOTETomek = 3,
    RandomUnderSampler = 4,
    SMOTEENN = 5,    

class Resample(object):
    """
    return resample object
    """    
    def __init__(self, n_jobs=-1, random_state=42, verbose=0):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def __Print(self, text):
        print('')
        print('--------------------------------------------------------')
        print('-- {} --'.format(text))
        print('--------------------------------------------------------')

    def get_resampler(self, 
                        resampler_type=ResamplerEnum.SMOTE.name, 
                        sampling_strategy='minority',
                        k_neighbors = 3,
                        allow_minority = True,
                        tomek_sampling_strategy = 'majority'):
        smote = SMOTE(random_state=self.random_state, 
                         n_jobs=self.n_jobs, 
                         sampling_strategy=sampling_strategy, 
                         k_neighbors=k_neighbors)
        if resampler_type.name == ResamplerEnum.SMOTE.name:
            return smote
        if resampler_type.name == ResamplerEnum.AllKNN.name:
            return AllKNN(allow_minority=allow_minority, 
                          n_jobs=self.n_jobs) 
        if resampler_type.name == ResamplerEnum.SMOTETomek.name:
            tomekLinks = TomekLinks(n_jobs=self.n_jobs, 
                                    sampling_strategy=tomek_sampling_strategy)
            return SMOTETomek(random_state=self.random_state, 
                              n_jobs=self.n_jobs, 
                              smote=smote, 
                              tomek=tomekLinks) 
        if resampler_type.name == ResamplerEnum.RandomUnderSampler.name:
            return RandomUnderSampler(random_state=self.random_state, 
                                      sampling_strategy=sampling_strategy)
        if resampler_type.name == ResamplerEnum.SMOTEENN.name:
            return SMOTEENN(random_state=self.random_state, 
                            n_jobs=self.n_jobs, 
                            smote=smote)        
        raise error('Theres is no resampler configured')

    def add_mock_data(self, df, mask, multiplier=10):        
        row = df[mask]
        return df.append([row]*multiplier,ignore_index=True)

    def apply_resample(self, df, column, num_cases=5, multiplier=10):
        self.__Print('Summary Resample')        
        print('Before Resample: Train Shape {}'.format(str(df.shape)))                   
        #add another row to minority class with few values
        grouped = df.groupby('Intencion_cat_label').count().sort_values(by='Intencion_cat_label', ascending=True)
        poor_cases = grouped[grouped[column] <= num_cases]
        print('Poor cases: {}'.format(len(poor_cases)))
        for index in poor_cases.index.unique():
            mask = (df['Intencion_cat_label'] == index)
            df = self.add_mock_data(df, mask, multiplier)        
        print('After Resample: Train Shape {}'.format(str(df.shape)))
        print('--------------------------------------------------------')
        return df