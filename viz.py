import data as d
import model as m

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pyplot as plt
import pandas as pd

features = ['LotArea','OverallQual','YearBuilt', 'YearRemodAdd', 'GrLivArea',
 #'TotRmsAbvGrd', 
 'GarageArea', 'Fireplaces']

class View:
    """Description of the class"""

    def __init__(self):
        self.__data = d.Data()
        self.__model = m.HousePredictionModel()
        self.__test = self.__data.get_prepared_test_data()
        self.__train = self.__data.get_prepared_train_data()
        self.__wid_val = widget_predict(self.__test)
        self.__model_type = 0
    
    def __set_model_type( self, mod_type):
        self.__model_type = mod_type

    def __set_model(self, mod_type):

        if mod_type == 0:
            model = self.__model.skit_mod
        elif mod_type == 1:
            model = self.__model.create_model(mod_type)
        elif mod_type == 2:
            model = self.__model.create_model(mod_type)
        else: 
            model = self.__model.skit_mod
        return model

    def __model_info_and_evoluation(self, mod_type):
        #mod = int(input("Type 0 for simple model and 1 for model with cross validation:"))

        model = self.__set_model(mod_type)
        self.__model.show_model_info(model)
        self.__model.show_test_metrics(model)


    def __choose_plot(self, plot):
        df = self.__train
        target = self.__data.target
        if plot == 0:
            m.plot_categorical_features(df, target)
        elif plot == 1:
            m.plot_numeric_features(df)
        elif plot == 2:
            m.plot_res_corr(df, target)


    def show_stats_model_info(self):
        self.__model.show_stat_model_info()


    def show_data_analysis(self):
    
        w = interactive(self.__choose_plot, plot = 
                                widgets.Dropdown(
                                options=[('Categorical', 0), ('Numerical', 1), ('Regression', 2)],
                                value=0, 
                                layout={'width': 'max-content'},
                                description='Plot Type:',)  )
        display(w)

        
    
    def __predict_house_price(self, a,b,c,d,e,f,g):#,j=0):
        import random

        house = {col: [random.choice(self.__wid_val[col])] for col in self.__test.columns}
        for i, feature in enumerate(features):
            if feature in house.keys():
                if i == 0: house[feature] = [a]
                if i == 1: house[feature] = [b]
                if i == 2: house[feature] = [c] 
                if i == 3: house[feature] = [d] 
                if i == 4: house[feature] = [e]
                if i == 5: house[feature] = [f]
                if i == 6: house[feature] = [g] 
                # if i == 7: house[feature] = [j] 
        #predict_price(house)
        if house['YearBuilt'][0] > house['YearRemodAdd'][0]:
            house['YearRemodAdd'][0] = house['YearBuilt'][0]

        h = pd.DataFrame(house).values

        predict = self.__model.predict(self.__model.predict_mod, h)

        print("Price of the house is {:,.0f} $".format(predict[0]))
    

    def display_model(self):

        w = interactive(self.__model_info_and_evoluation, mod_type = 
                                widgets.Dropdown(
                                options=[('Linear Regression', 0), ('Ridge', 1), ('Lasso', 2)],
                                value=0, 
                                layout={'width': 'max-content'},
                                description='Model:',)  )
        display(w)

    def display_house_price_prediction(self):

        w0 = interactive(self.__set_model_type,  mod_type = widgets.Dropdown(
                                options=[('Linear Regression', 0), ('Ridge', 1), ('Lasso', 2)],
                                value=0, 
                                layout={'width': 'max-content'},
                                description='Model:',) )
        display(w0)

        self.__model.predict_mod = self.__set_model(self.__model_type)

        w = interactive(self.__predict_house_price,  a=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[0]]), 
                                                        max=max(self.__wid_val[features[0]]), 
                                                        value=max(self.__wid_val[features[0]]), 
                                                        step=1, 
                                                        description=features[0]),
                                    b=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[1]]), 
                                                        max=max(self.__wid_val[features[1]]), 
                                                        value=max(self.__wid_val[features[1]]), 
                                                        step=1, 
                                                        description=features[1]),
                                    c=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[2]]), 
                                                        max=max(self.__wid_val[features[2]]), 
                                                        value=max(self.__wid_val[features[2]]), 
                                                        step=1, 
                                                        description=features[2]),
                                    d=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[3]]), 
                                                        max=max(self.__wid_val[features[3]]), 
                                                        value=max(self.__wid_val[features[3]]), 
                                                        step=1, 
                                                        description=features[3]),
                                    e=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[4]]), 
                                                        max=max(self.__wid_val[features[4]]), 
                                                        value=max(self.__wid_val[features[4]]), 
                                                        step=1, 
                                                        description=features[4]),
                                    f=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[5]]), 
                                                        max=max(self.__wid_val[features[5]]), 
                                                        value=max(self.__wid_val[features[5]]), 
                                                        step=1, 
                                                        description=features[5]),
                                    g=widgets.IntSlider(
                                                        min=min(self.__wid_val[features[6]]), 
                                                        max=max(self.__wid_val[features[6]]), 
                                                        value=max(self.__wid_val[features[6]]), 
                                                        step=1, 
                                                        description=features[6]),
                                    # j=widgets.IntSlider(
                                    #                     min=min(self.__wid_val[features[7]]), 
                                    #                     max=max(self.__wid_val[features[7]]), 
                                    #                     value=min(self.__wid_val[features[7]]), 
                                    #                     step=1, 
                                    #                     description=features[7]),
                        )

        display(w)

        
        
        


### help functions

## set parameters for widget house price prediction
def widget_predict(df):
    dict_params = {col: 0 for col in df.columns}
    for key in dict_params.keys():
        if df[key].dtypes == 'int' or df[key].dtypes == 'float':
            if len(df[key].unique())>10:
                dict_params[key] = [df[key].min(), df[key].max()]
            #print(f'{key} :{df[key].unique()} ')
            elif len(df[key].unique())<=10:
                dict_params[key] = list(df[key].unique())
        else:
            dict_params[key] = list(df[key].unique())
    return dict_params
