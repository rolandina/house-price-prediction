from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import data as d

### statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms

### sklearn
# preprocessing and metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

### models
from sklearn.linear_model import LinearRegression
### model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold





class HousePredictionModel:
    def __init__(self):
        data = d.Data()
        self.train_df = data.get_prepared_train_data()
        self.target = data.target
        
        self.X, self.y = self.train_df[[col for col in self.train_df.columns if col != self.target]], self.train_df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values, self.y.values, test_size=0.40, random_state=1)
        self.stat_mod = sm.OLS(self.y, sm.add_constant(self.X)).fit()

        ## create simple model based on train data set only
        self.skit_mod  = LinearRegression().fit(self.X_train, self.y_train)

##changed
    def create_model(self, mod_type):
        cv1 = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
        parameters= [
            {'normalize': [True, False]}  
        ]
        Grid = GridSearchCV(LinearRegression(), parameters, scoring='r2',cv= cv1)
        Grid.fit(self.X_train, self.y_train)
        return Grid.best_estimator_

    def show_stat_model_info(self):
        print(self.stat_mod.summary())

###changed
    def show_model_info(self, model):
        print(f"Model: {model}")
        print(f"Model parameters: \n {model.get_params()}")
        mod_score = model.score(self.X_test, self.y_test)
        print(f"Model score is {mod_score}")

    def predict(self, model, X):
        return model.predict(X)

    def show_test_metrics(self, model):
        # Get predictions
        predictions = model.predict(self.X_test)

        # Show metrics
        print('Metrics of y_test prediction:')
        mse = mean_squared_error(self.y_test, predictions)
        print("MSE:", mse)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)
        r2 = r2_score(self.y_test, predictions)
        print("R2:", r2)
        # Plot predicted vs actual
        
        plt.scatter(self.y_test, predictions)
        plt.xlabel(f'Actual {self.target}')
        plt.ylabel(f'Predicted {self.target}')
        plt.title('Predictions vs Actuals')
        z = np.polyfit(self.y_test, predictions, 1)
        p = np.poly1d(z)
        plt.plot(self.y_test, p(self.y_test), color='magenta')
        plt.show()




## visualization functions and other help functions

def create_list_numeric_columns(df):
    return [column for column in df.columns if df[column].dtypes == "float" or (len(df[column].unique())>=15 and df[column].dtypes == "int")]

def create_list_categoric_columns(df):
    return [column for column in df.columns if df[column].dtypes != "float" and df[column].dtypes != "int"] + [column for column in df.columns if df[column].dtypes == "int" and len(df[column].unique())<15]

def plot_numeric_features(df):
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


def plot_categorical_features(df, target):  
    categorical_features = create_list_categoric_columns(df)

    sns.set()  # Setting seaborn as default style even if use only matplotlib
    sns.set_palette("Paired")  # set color palette
    fig, axes = plt.subplots(nrows=len(categorical_features),
                            ncols=1,
                            figsize=(14, 4* len(categorical_features)))
    for i, feature in enumerate(categorical_features):
        
        df_group = df[[feature, target]].groupby(feature).count()
        #print(df_group)
        sns.boxplot(data=df, x=feature, y= target, ax=axes[i])
        sns.swarmplot(data=df, x=feature, y= target, ax=axes[i], color=".25", size = 2)
        if len(df[feature].unique())>20:
            plt.xticks(rotation=45)
        axes[i].set_title(f"{target} by {feature}")
    plt.tight_layout()
    plt.show()


def plot_res_corr(df, target): 
    numerical_features = create_list_numeric_columns(df)
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



