col_save =  [ 
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

import pandas as pd

def prepare_data():   
#read data
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    df = pd.concat([df_test, df_train])
    num_col = len(df.columns)
# drop columns
    df = drop_columns(df)
#replace nan
    #replace_nan(df)     
# drop nan rows
    #drop_nan_rows(df)
    
#remove columns if number of nan is bigger than 50% of len table
    df = remove_nan_column(df, 0.5)
    
#remove anomaly
    df = abnormal_filter(df, 0.2, 1)
    #print(f"Dropped {num_col - len(df.columns)} columns.")
    return df

def drop_columns(df):
    drop_cols_names = list(set(df.columns).difference(set(col_save)))
    df1 = df[col_save]
    #print(f"List of dropped columns:{drop_cols_names}")
    return df1
 
def drop_nan_rows(df):    
    df = df.dropna(axis = 0)

# df.count() does not include NaN values
def remove_nan_column(df_old, percent_not_nan_in_column):    
    df_new = df_old[[column for column in df_old if df_old[column].count() / len(df_old) >= percent_not_nan_in_column]]
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


def replace_nan(df):
    import numpy as np
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

#test = prepare_train_data()

    