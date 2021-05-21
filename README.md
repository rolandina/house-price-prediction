# *HOUSE PRICE PREDICTION* 

## *What is does?*

House price prediction project consists from files:

- [house_price_prediction.ipynb](https://github.com/rolandina/house-price-prediction/blob/main/house_price_prediction.ipynb) - notebook file where you can do prediction 
- [viz.py](https://github.com/rolandina/house-price-prediction/blob/main/viz.py) - file with view functions you will find in notebook
- [model.py](https://github.com/rolandina/house-price-prediction/blob/main/model.py)
- [data.py](https://github.com/rolandina/house-price-prediction/blob/main/data.py)
- [data/data_description.txt](https://github.com/rolandina/house-price-prediction/blob/main/data/data_description.txt)
- [data/train.csv](https://github.com/rolandina/house-price-prediction/blob/main/data/train.csv)
- [data/test.csv](https://github.com/rolandina/house-price-prediction/blob/main/data/test.csv)
- [environment.yml](https://github.com/rolandina/house-price-prediction/blob/main/environment.yml) -  file with prerequisites libraries

In notebook you have access to function:

- show_model_info() - to show regression model info
- show_data_analysis() - to see statistical analysis of the data
- show_model_evaluation() - to see model metrics based on test data 
- predict_house_price() - to set parameters for a house and predict it final price

## General Prerequisites (for running and building)

* [Anaconda](https://www.anaconda.com/products/individual)
* [GitHub](https://github.com)

After installing anaconda and python you have to set your environment with environment.yml file

```bash
conda env create -f environment.yml
conda activate house_predict
```

## To build and run


```
# Clone this repository 
git clone https://github.com/rolandina/house-price-prediction.git
```

## Contributing

The data from this project was taken from [Kaggle Data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
If you have any questions, please reach me at ms.nina.smirnova@gmail.com

