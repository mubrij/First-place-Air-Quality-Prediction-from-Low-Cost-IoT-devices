# README for 1st place notebook

## Overview
This Jupyter Notebook is designed for data preprocessing, model training, and evaluation using various machine learning models. It includes data visualization, model selection, and performance evaluation using cross-validation. The notebook leverages popular Python libraries such as pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, ExtraTreesRegressor and shap for model explainability.

### Folder Structure
all dataset, notebook and csv output are stored in the same directory or it follows the below structure for better organizing

project/
│
├── data/                    # Folder to store datasets
│   └── train.csv          # Dataset file used in the notebook
│   └── test.csv          # Dataset file used in the notebook
└──result/output -extratreesregressors.csv          # Example of an output file
│
├── 1st place notebook       # Main notebook file
└── README.md                # This README file

dependencies:
  - python=3.10.12
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost
  - shap
  - seaborn
  - matplotlib
  - tqdm
  - tabulate

### Hardware Requirements
- *Local Machine*: 
  -  2-8GB RAM 
  - 4-core CPU
  - GPU (optional but recommended for faster training, especially with xgboost, lightgbm, and catboost)
- *Google Colab*: 
  - The notebook can also be run on Google Colab with a GPU runtime for faster execution.
  - *Kaggle*: 

## Order of Execution
1. *Data Loading and Preprocessing*:
   - Load the dataset using pandas.
   - Perform necessary data cleaning and preprocessing steps.
   - Perform any necessary data cleaning, transformation, or feature engineering.
   
2. *Exploratory Data Analysis (EDA)*:
   - Visualize data distributions, correlations, and missing values using seaborn and matplotlib.

3. *Model Training and Evaluation*:
   - Train multiple machine learning models (LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, SVR, XGBRegressor, LGBMRegressor, CatBoostRegressor, KNeighborsRegressor, GaussianProcessRegressor, DecisionTreeRegressor).
   - Evaluate models using cross-validation (KFold) and calculate the mean squared error (mean_squared_error).

4. *Model Explainability*:
   - Use shap to explain model predictions and visualize feature importance.

5. *Saving Results*:
   - Save trained models and evaluation results in the models/ and output/ folders, respectively.
---

## Features Used
- *Data Preprocessing*: 
  - Libraries: pandas, numpy, scikit-learn
  - Techniques: feature scaling, handling missing values, and encoding categorical variables.
 
- *Machine Learning Models*:
  - Libraries: scikit-learn, xgboost, lightgbm, catboost
  - *Model Selection*: Training and evaluating multiple regression models which includes: Linear Regression, Ridge, Lasso, ElasticNet, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, SVR, XGBRegressor, LGBMRegressor, CatBoostRegressor.
- *Model Explainability*: Using shap to interpret model predictions.

- *Cross-Validation*: Using KFold to ensure robust model evaluation.
- *Visualization*: Plotting data distributions, model performance, and feature importance.
- *creation of csv file for respective submission*: 

## Expected Run Time
- *Local Machine (CPU)*: Approximately  30 mins, depending on the dataset size and model complexity(if user decide to drop featuresand some models)
- *Local Machine (GPU)*: less than 15 minutes.
- *Google Colab (GPU)*: less than 15 minutes.

## Notes
- Ensure that all dependencies are installed correctly using the provided requirement.txt file.
- The notebook is designed to be run sequentially from top to bottom. Running cells out of order may result in errors.
- If running on Google Colab, ensure that the GPU runtime is enabled for faster execution.
-- *Google Colab or Kaggle*: If running on Google Colab, ensure that the dataset is uploaded to the Colab environment or mounted from Google Drive.

## Troubleshooting
- *Missing Data*: Ensure that the dataset is placed in the data/ folder or in same directory as the notebook and is correctly referenced in the notebook.
- *Environment Issues*: If you encounter issues with the environment, try recreating it using the environment.yml file.
- *Model Training Errors*: If a model fails to train, check for missing or incorrect data preprocessing steps.

## Best submission info:
    ID: EnPkbHCZ
    Filename: ExtraTreesRegressor_10fold.csv
    Comment: —
    Submitted: 2 February 08:30 

## Experiments that doesn't worked out:
- Simple mean ensembling along different models.
- Feature engineering using sklearn's PolynomialFeatures.
- xgboost, lightgbm, catboost, laso, Ridge model did not learn well

By following this README, you should be able to set up and run the notebook smoothly. If you encounter any issues, please refer to the troubleshooting section or reach out for support.For any issues, suggestions, or queries, contact the maintainers at [musamuhammadtukur127@gmail.com] [edifonemmanuel14@gmail.com] [hussainkhasamwala23@gmail.com] [agboolayusuf2018@gmail.com] 