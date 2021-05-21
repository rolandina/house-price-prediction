from data import *
from model import *

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt
import pandas as pd

def show_data_analysis():
    
    df = prepare_data() # prepare function
    # plot
    target = 'SalePrice'
    numerical_features = create_list_numeric_columns(df)
    categorical_features = create_list_categoric_columns(df)


    plot_numeric_features(df, numerical_features)
    plot_categorical_features(df, categorical_features, target)
    plot_res_corr(df, numerical_features, target)


def create_list_numeric_columns(df):
    return [column for column in df.columns if df[column].dtypes == "float" or (len(df[column].unique())>=15 and df[column].dtypes == "int")]

def create_list_categoric_columns(df):
    return [column for column in df.columns if df[column].dtypes != "float" and df[column].dtypes != "int"] + [column for column in df.columns if df[column].dtypes == "int" and len(df[column].unique())<15]


def plot_numeric_features(df, numerical_features_list):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()  # Setting seaborn as default style even if use only matplotlib
    sns.set_palette("Paired")  # set color palette
    fig, axes = plt.subplots(nrows=len(numerical_features_list),
                             ncols=2,
                             figsize=(10, 3* len(numerical_features_list)))
    for i, feature in enumerate(numerical_features_list):
        sns.histplot(data=df, x=feature, kde=True, ax=axes[i, 0])
        sns.boxplot(data=df, x=feature, ax=axes[i, 1])
    plt.tight_layout()
    plt.show()


def plot_categorical_features(df, features_list, label_feature):  
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set()  # Setting seaborn as default style even if use only matplotlib
    sns.set_palette("Paired")  # set color palette
    fig, axes = plt.subplots(nrows=len(features_list),
                             ncols=1,
                             figsize=(14, 4* len(features_list)))
    for i, feature in enumerate(features_list):
        
        df_group = df[[feature, label_feature]].groupby(feature).count()
        #print(df_group)
        sns.boxplot(data=df, x=feature, y= label_feature, ax=axes[i])
        sns.swarmplot(data=df, x=feature, y= label_feature, ax=axes[i], color=".25", size = 2)
        if len(df[feature].unique())>20:
            plt.xticks(rotation=45)
        axes[i].set_title(f"{label_feature} by {feature}")
    plt.tight_layout()
    plt.show()
    
def plot_res_corr(df, features, label): 
    import matplotlib.pyplot as plt
    import seaborn as sns
    n = len(features)-1

    fig, axes = plt.subplots(nrows=n,
                             ncols=2,
                             figsize=(10, 4*n))
    i = 0
    for f in features: 
        if f  != label:
                sns.regplot(data=df, x=f, y=label, color='blue', ax=axes[i, 0])
                sns.residplot(data=df, x=f, y=label, color='red', ax = axes[i, 1])
                i+=1
            
# show_plots
def show_plots():
    #prepare_data()
    return



## house price prediction
def prepare_widgets(df):
    house = {col: 0 for col in df.columns}
    for key in house.keys():
        if df[key].dtypes == 'int' or df[key].dtypes == 'float':
            if len(df[key].unique())>10:
                house[key] = [df[key].min(), df[key].max()]
            #print(f'{key} :{df[key].unique()} ')
            elif len(df[key].unique())<=10:
                house[key] = list(df[key].unique())
        else:
            house[key] = list(df[key].unique())
    return house


features = ['LotArea','OverallQual','YearBuilt', 'YearRemodAdd', 'GrLivArea', 'TotRmsAbvGrd', 'GarageArea', 'Fireplaces']

def create_house(a,b,c,d,e,f,g,j):
    import random
    df = prepare_data()
    m1 = HousePredictionModel(df,'SalePrice')
    df = df.drop(columns = ['SalePrice'])

    wid_vals = prepare_widgets(df)
    
    
    house = {col: [random.choice(wid_vals[col])] for col in df.columns}
    for i, feature in enumerate(features):
        if feature in house.keys():
            if i == 0: house[feature] = [a]
            if i == 1: house[feature] = [b]
            if i == 2: house[feature] = [c] 
            if i == 3: house[feature] = [d] 
            if i == 4: house[feature] = [e]
            if i == 5: house[feature] = [f]
            if i == 6: house[feature] = [g] 
            if i == 7: house[feature] = [j] 
    #predict_price(house)
    if house['YearBuilt'][0]>house['YearRemodAdd'][0]:
        house['YearRemodAdd'][0] = house['YearBuilt'][0]

    hdf = pd.DataFrame(house)

    h = hdf.values
    predict = m1.predict(h)

    print("Price of the house is {:,.0f} $".format(predict[0]))
 




def predict_house_price():
    df = prepare_data()
    wid_vals = prepare_widgets(df)
    w = interactive(create_house,  a=widgets.IntSlider(
                                                    min=min(wid_vals[features[0]]), 
                                                    max=max(wid_vals[features[0]]), 
                                                    value=min(wid_vals[features[0]]), 
                                                    step=1, 
                                                    description=features[0]),
                                b=widgets.IntSlider(
                                                    min=min(wid_vals[features[1]]), 
                                                    max=max(wid_vals[features[1]]), 
                                                    value=min(wid_vals[features[1]]), 
                                                    step=1, 
                                                    description=features[1]),
                                c=widgets.IntSlider(
                                                    min=min(wid_vals[features[2]]), 
                                                    max=max(wid_vals[features[2]]), 
                                                    value=min(wid_vals[features[2]]), 
                                                    step=1, 
                                                    description=features[2]),
                                d=widgets.IntSlider(
                                                    min=min(wid_vals[features[3]]), 
                                                    max=max(wid_vals[features[3]]), 
                                                    value=min(wid_vals[features[3]]), 
                                                    step=1, 
                                                    description=features[3]),
                                e=widgets.IntSlider(
                                                    min=min(wid_vals[features[4]]), 
                                                    max=max(wid_vals[features[4]]), 
                                                    value=min(wid_vals[features[4]]), 
                                                    step=1, 
                                                    description=features[4]),
                                f=widgets.IntSlider(
                                                    min=min(wid_vals[features[5]]), 
                                                    max=max(wid_vals[features[5]]), 
                                                    value=min(wid_vals[features[5]]), 
                                                    step=1, 
                                                    description=features[5]),
                                g=widgets.IntSlider(
                                                    min=min(wid_vals[features[6]]), 
                                                    max=max(wid_vals[features[6]]), 
                                                    value=min(wid_vals[features[6]]), 
                                                    step=1, 
                                                    description=features[6]),
                                j=widgets.IntSlider(
                                                    min=min(wid_vals[features[7]]), 
                                                    max=max(wid_vals[features[7]]), 
                                                    value=min(wid_vals[features[7]]), 
                                                    step=1, 
                                                    description=features[7]),
                       )

    display(w)



def show_model_info():
    df = prepare_data()
    m1 = HousePredictionModel(df,'SalePrice')
    m1.show_model_info()


def show_model_evaluation():
    df = prepare_data()
    m1 = HousePredictionModel(df,'SalePrice')
    m1.display_test_metrics()

