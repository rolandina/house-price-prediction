columns_to_save =  [ 
    'LotArea', 
    #'LotShape', ## change reg/irreg
    #'Neighborhood', 
    'OverallQual', 
    'YearBuilt', 
    'YearRemodAdd', 
    #'Exterior1st', 
    #'ExterQual', ## good/not good
    #'Foundation', ## pcon/notpcon 
    'GrLivArea', 
    'TotRmsAbvGrd', 
    'Fireplaces', 
    # 'GarageType', ## attached/detached
    #'GarageCars', 
    'GarageArea',  
    'SalePrice'
    ]

columns_to_save_test =   [ 
    'LotArea', 
    #'LotShape', ## change reg/irreg
    #'Neighborhood', 
    'OverallQual']



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class Data:
    # change parameters here
    target = 'SalePrice'

    __chosen_columns = columns_to_save
    __data_file = 'data.csv'
    __max_NaN_in_columns = 0.2 # if more than 20% nan column will be removed
    __threshold_first =  0.2  #threshold_first (5%-min or max-95%) for abnormal filter
    __threshold_second =  1.    #threshold_second (second diff., times) for abnormal filter

    def __init__(self):
        ## initialise with Data object with data frame
        self.__df_train, self.__df_test = self.__read_df()

    # read data file
    def __read_df(self):
        ## return pd.read_csv(self.__data_file)
        df_train = pd.read_csv('data/train.csv')
        df_test = pd.read_csv('data/test.csv')
        return (df_train, df_test)
  
    def get_data_frame(self):
        return self.__df_train

    def get_columns_name(self):
        return self.__df_train.columns.tolist()
        
    def set_file_name(self):
        new_file = input("Write the file name you want to look at :") 
        self.__data_file = new_file

    def set_columns_to_save(self):
        ## ADD INTERACT here
        cols = self.get_columns_name()
        ## chose columns
        cols = columns_to_save_test
        self.__chosen_columns = cols
        print(self.__chosen_columns)

    def get_prepared_train_data(self):
        return self.__prepare_data(self.__df_train, self.__chosen_columns)
    
    def get_prepared_test_data(self):
        cols_to_save = self.__chosen_columns.copy()
        cols_to_save.remove(self.target)
        return self.__prepare_data(self.__df_test, cols_to_save)

### change this function for each data frame
    def __prepare_data(self, df, cols_to_save):  

    # drop columns
        new_df = drop_columns(df, cols_to_save)
    # replace nan manually - function for manual correction for different data frame
        #replace_nan_manually(new_df)     
    # drop nan rows
        #drop_nan_rows(new_df)
        
        new_df = remove_nan_column(new_df, self.__max_NaN_in_columns)
        
    #remove anomaly
        new_df = abnormal_filter(new_df, self.__threshold_first, self.__threshold_second)
        #print(f"Dropped {num_col - len(df.columns)} columns.")
        return new_df



    def plot_numeric_features(self,df):
        numerical_features = create_list_numeric_columns(df)
        sns.set()  # Setting seaborn as default style even if use only matplotlib
        sns.set_palette("Paired")  # set color palette
        fig, axes = plt.subplots(nrows=len(numerical_features),
                                ncols=2,
                                figsize=(10, 3* len(numerical_features)))
        for i, feature in enumerate(numerical_features):
            sns.histplot(data=df, x=feature, kde=True, ax=axes[i, 0])
            sns.boxplot(data=df, x=feature, ax=axes[i, 1])
        plt.tight_layout()
        plt.show()


    def plot_categorical_features(self,df):  
        categorical_features = create_list_categoric_columns(df)

        sns.set()  # Setting seaborn as default style even if use only matplotlib
        sns.set_palette("Paired")  # set color palette
        fig, axes = plt.subplots(nrows=len(categorical_features),
                                ncols=1,
                                figsize=(14, 4* len(categorical_features)))
        for i, feature in enumerate(categorical_features):
            
            df_group = df[[feature, self.target]].groupby(feature).count()
            #print(df_group)
            sns.boxplot(data=df, x=feature, y= self.target, ax=axes[i])
            sns.swarmplot(data=df, x=feature, y= self.target, ax=axes[i], color=".25", size = 2)
            if len(df[feature].unique())>20:
                plt.xticks(rotation=45)
            axes[i].set_title(f"{self.target} by {feature}")
        plt.tight_layout()
        plt.show()
    




    def plot_res_corr(self): 
        numerical_features = create_list_numeric_columns()
        n = len(numerical_features)-1

        fig, axes = plt.subplots(nrows=n,
                                ncols=2,
                                figsize=(10, 4*n))
        i = 0
        for f in numerical_features: 
            if f  != target:
                    sns.regplot(data=df, x=f, y=target, color='blue', ax=axes[i, 0])
                    sns.residplot(data=df, x=f, y=target, color='red', ax = axes[i, 1])
                    i+=1





### help functions
def drop_columns(df, columns_to_save):
    df1 = df[columns_to_save]
    #print(f"List of dropped columns:{drop_cols_names}")
    return df1
 
def drop_nan_rows(df):    
    df = df.dropna(axis = 0)

# df.count() does not include NaN values
def remove_nan_column(df_old, percent_nan_in_column):    
    
    df_new = df_old[[column for column in df_old if df_old[column].count()/len(df_old) >= 1.- percent_nan_in_column]]
    #print("List of dropped columns:", end=" ")
    if 'Id' in df_old.columns:
        del df_new['Id']
    # for c in df_old.columns:
    #     if c not in df_new.columns:
    #         print(c, end=", ")
    return df_new


def abnormal_filter(df, threshold_first, threshold_second):
    # Abnormal values filter for DataFrame df:
    # threshold_first (5%-min or max-95%)
    # threshold_second (second diff., times)
    ## 5% 10% 90% 95%
    df_describe = df.describe([.05, .1, .9, .95])
    cols = df_describe.columns.tolist()
    i = 0
    abnorm = 0
    for col in cols:
        i += 1
        # abnormal smallest
        P10_5 = df_describe.loc['10%',col]-df_describe.loc['5%',col]
        P_max_min = df_describe.loc['max',col]-df_describe.loc['min',col]
        if P10_5 != 0:
            if (df_describe.loc['5%',col]-df_describe.loc['min',col])/P10_5 > threshold_second:
                #abnormal smallest filter
                df = df[(df[col] >= df_describe.loc['5%',col])]
                #print('1: ', i, col, df_describe.loc['min',col],df_describe.loc['5%',col],df_describe.loc['10%',col])
                abnorm += 1
        else:
            if P_max_min > 0:
                if (df_describe.loc['5%',col]-df_describe.loc['min',col])/P_max_min > threshold_first:
                    # abnormal smallest filter
                    df = df[(df[col] >= df_describe.loc['5%',col])]
                    #print('2: ', i, col, df_describe.loc['min',col],df_describe.loc['5%',col],df_describe.loc['max',col])
                    abnorm += 1

        
        # abnormal biggest
        P95_90 = df_describe.loc['95%',col]-df_describe.loc['90%',col]
        if P95_90 != 0:
            if (df_describe.loc['max',col]-df_describe.loc['95%',col])/P95_90 > threshold_second:
                #abnormal biggest filter
                df = df[(df[col] <= df_describe.loc['95%',col])]
                #print('3: ', i, col, df_describe.loc['90%',col],df_describe.loc['95%',col],df_describe.loc['max',col])
                abnorm += 1
        else:
            if P_max_min > 0:
                if ((df_describe.loc['max',col]-df_describe.loc['95%',col])/P_max_min > threshold_first) & (df_describe.loc['95%',col] > 0):
                    # abnormal biggest filter
                    df = df[(df[col] <= df_describe.loc['95%',col])]
                    #print('4: ', i, col, df_describe.loc['min',col],df_describe.loc['95%',col],df_describe.loc['max',col])
                    abnorm += 1
    #print('Number of abnormal values removed =', abnorm)
    return df


def replace_nan_manually(df):
    df['LotFrontage'] = df['LotFrontage'].replace(np.nan, 'No Street connected')
    df['Alley'] = df['Alley'].replace(np.nan, 'No Alley')
    df['MasVnrType'] = df['MasVnrType'].replace(np.nan, 'No MasVnr')
    df['MasVnrArea'] = df['MasVnrArea'].replace(np.nan, 0).astype(float)
    df['BsmtQual'] = df['BsmtQual'].replace(np.nan, 'No Bsmt')
    df['BsmtCond'] = df['BsmtCond'].replace(np.nan, 'No Bsmt')
    df['BsmtExposure'] = df['BsmtExposure'].replace(np.nan, 'No Bsmt')
    df['BsmtFinType1'] = df['BsmtFinType1'].replace(np.nan, 'No Bsmt')
    df['BsmtFinType2'] = df['BsmtFinType2'].replace(np.nan, 'No 2ndBsmt')
    df['FireplaceQu'] = df['FireplaceQu'].replace(np.nan, 'No Fireplace')
    df['GarageType'] = df['GarageType'].replace(np.nan, 'No Garage')
    df['GarageYrBlt'] = df['GarageYrBlt'].replace(np.nan, 'No Garage')
    df['GarageFinish'] = df['GarageFinish'].replace(np.nan, 'No Garage')
    df['GarageQual'] = df['GarageQual'].replace(np.nan, 'No Garage')
    df['GarageCond'] = df['GarageCond'].replace(np.nan, 'No Garage')
    df['PoolQC'] = df['PoolQC'].replace(np.nan, 'No Pool')
    df['Fence'] = df['Fence'].replace(np.nan, 'No Fence')
    df['MiscFeature'] = df['MiscFeature'].replace(np.nan, 'No Feature')



def create_list_numeric_columns(df):
    return [column for column in df.columns if df[column].dtypes == "float" or (len(df[column].unique())>=15 and df[column].dtypes == "int")]

def create_list_categoric_columns(df):
    return [column for column in df.columns if df[column].dtypes != "float" and df[column].dtypes != "int"] + [column for column in df.columns if df[column].dtypes == "int" and len(df[column].unique())<15]




    