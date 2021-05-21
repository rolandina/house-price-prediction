from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
from data import *

class HousePredictionModel:
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.X, self.y = df[[col for col in df.columns if col != label]], df[label]
        self.stat_mod = sm.OLS(self.y, sm.add_constant(self.X)).fit()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values, self.y.values, test_size=0.30, random_state=0)
        self.skit_mod  = LinearRegression().fit(self.X_train, self.y_train)

    def show_model_info(self):
        print(self.stat_mod.summary())

    def predict(self, X):
        return self.skit_mod.predict(X)

    def display_test_metrics(self):
        # Get predictions
        predictions = self.skit_mod.predict(self.X_test)

        # Display metrics
        mse = mean_squared_error(self.y_test, predictions)
        print("MSE:", mse)
        rmse = np.sqrt(mse)
        print("RMSE:", rmse)
        r2 = r2_score(self.y_test, predictions)
        print("R2:", r2)
        # Plot predicted vs actual
        
        plt.scatter(self.y_test, predictions)
        plt.xlabel('Actual Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Predictions vs Actuals')
        z = np.polyfit(self.y_test, predictions, 1)
        p = np.poly1d(z)
        plt.plot(self.y_test, p(self.y_test), color='magenta')
        plt.show()

