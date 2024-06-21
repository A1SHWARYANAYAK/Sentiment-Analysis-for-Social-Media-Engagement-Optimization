import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from lightgbm import LGBMRegressor
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import time
import os

Tuning_Method = "Halving_Randomized_Search_CV"

def visualization(results_df, parameters):

    def shorten_param(param_name):
        if "__" in param_name:
            return param_name.rsplit("__", 1)[1]
        return param_name

    column_results = [f"param_{name}" for name in parameters.keys()]
    column_results += ["mean_test_score", "std_test_score", "rank_test_score"]

    results_df = results_df[column_results].sort_values("mean_test_score", ascending=False)
    results_df = results_df.rename(shorten_param, axis=1)

    for col in results_df.columns:
        if col == 'param_random_state':
            continue
        try:
            results_df[col] = results_df[col].astype(np.float64)
        except:
            continue

    fig = px.parallel_coordinates(
    results_df,
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis,
    title='Hyper Parameter Tuning',
    )
    fig.show()

def hyperparameter_tuning(x_train, y_train, model, parameters, tuning_model):

    if tuning_model == 'Halving_Randomized_Search_CV':
        tuned_model = HalvingRandomSearchCV(model, param_distributions = parameters, scoring = "neg_mean_squared_error", n_jobs=-1, factor=3, cv = 3 )

    elif tuning_model == 'Randomized_Search_CV':
        tuned_model = RandomizedSearchCV(model, param_distributions = parameters, scoring = 'neg_mean_squared_error', cv = 3, n_iter = 50, n_jobs=-1)

    else:
        tuned_model = GridSearchCV(model, param_grid = parameters, scoring = 'neg_mean_squared_error', n_jobs=-1, cv = 3)


    start_time = time.time()

    if type(model).__name__ == 'CatBoostRegressor':
        tuned_model.fit(x_train, y_train, verbose = False)
    else:
        tuned_model.fit(x_train, y_train)

    stop_time = time.time()

    print('*****'*10+f'\nBest Score for {type(model).__name__} : {tuned_model.best_score_}','\n---')
    print(f'Best Parameters for {type(model).__name__} : {tuned_model.best_params_}\n'+'-----'*10)

    print('Elapsed Time:',time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time)))
    print('======'*5)

    return tuned_model

def evaluation_metrics(name, independent_feature_length , y_pred, y_test):

    n = len(y_pred)
    k = independent_feature_length

    metrics_dict = {}
    metrics_dict['MSE'] = [mean_squared_error(y_test,y_pred)]  #MSE
    metrics_dict['RMSE'] = [np.sqrt(mean_squared_error(y_test,y_pred))] #RMSE
    metrics_dict['MAE'] = [mean_absolute_error(y_test,y_pred)] #MAE
    metrics_dict['R2_Score'] = [r2_score(y_test,y_pred)] #R2 Score
    metrics_dict['Adjusted_R2_Score'] = [1 - ((1-metrics_dict['R2_Score'][0])*(n-1)/(n-k-1))] #Adjusted R2 score
    metrics_dict['RMSLE'] = [np.log(np.sqrt(mean_squared_error(y_test,y_pred)))] #RMSLE

    metrics_df = pd.DataFrame(metrics_dict)

    print(metrics_df)    
    return metrics_df

def ml_algorithm_implementation(df, model, parameters, social_media, tuning_model, feature_importance = False):

    if feature_importance == False:
        print('########'*8+'\n     <<<< '+f'Tuning Model: {tuning_model}'+' >>>>\n'+'********'*8)

    x_cols = [x for x in df.columns if x not in ['Facebook_scaled','LinkedIn_scaled','GooglePlus_scaled']]

    x = df[x_cols]
    y = df[[f'{social_media}_scaled']]

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=23)

    if feature_importance == True:
        model.fit(x_train, y_train)
        return x_train, model

    perform_ml_algorithm(x_train, x_test, y_train, y_test, model, parameters, social_media, tuning_model)


def perform_ml_algorithm(x_train, x_test, y_train, y_test, model, parameters, social_media, tuning_model, flag = True):
    print('-----'*10+f'\n{type(model).__name__} for {social_media}\n'+'-----'*10)

    if type(model).__name__ == 'CatBoostRegressor':
        model.fit(x_train, y_train, verbose = False)
    else:
        model.fit(x_train, y_train)
    untuned_pred = model.predict(x_test)

    # Evaluation Metrics before tuning
    print(f'\nEvaluation of {type(model).__name__} before tuning:\n'+'-----'*10)
    metrics_df_before = evaluation_metrics(type(model).__name__, len(list(x_train.columns)), untuned_pred, y_test)
    st.write("Evaluation Metrics Before Tuning:")
    st.dataframe(metrics_df_before)

    # Hyper-parameter tuning
    tuned_model = hyperparameter_tuning(x_train, y_train, model, parameters, tuning_model)
    tuned_pred = tuned_model.predict(x_test)

    # Evaluation Metrics after tuning
    print(f'\nEvaluation of {type(model).__name__} after tuning:\n'+'-----'*10)
    metrics_df_after =evaluation_metrics(type(model).__name__,len(list(x_train.columns)), tuned_pred, y_test)
    st.write("Evaluation Metrics After Tuning:")
    st.dataframe(metrics_df_after)

    if flag:
     visualization(pd.DataFrame(tuned_model.cv_results_), parameters)

# Load your data
def load_data():
    return pd.read_csv('sentiment_analysis.csv')

def user_input_features():
    # Collect user input for hyperparameters
    st.sidebar.title('Select Model Input Parameters: LightGBM')
    n_estimators = [st.sidebar.slider('Number of Estimators', min_value=50, max_value=200, value=100, step=50)]
    learning_rate = [st.sidebar.slider('Learning Rate', min_value=0.001, max_value=0.1, value=0.01, step=0.001)]
    max_depth = [st.sidebar.slider('Max Depth', min_value=1, max_value=10, value=4, step=1)]
    subsample = [st.sidebar.slider('Subsample', min_value=0.5, max_value=1.0, value=0.75, step=0.05)]
    return {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'subsample': subsample
    }

# Function to train model or load from pkl based on the user-selected platform
def train_or_load_model(data, platform, parameters):
    filename = f'{platform.lower()}_model.pkl'
    if os.path.exists(filename):
    # Delete the file
       os.remove(filename)
    # LightGBM
    parameters_lightgbm = {'n_estimators': parameters['n_estimators'],
                        'learning_rate':parameters['learning_rate'],
                        'max_depth':parameters['max_depth'],
                        'subsample':parameters['subsample'],
                        'random_state':[25]}

    x_train, model = ml_algorithm_implementation(data, LGBMRegressor(random_state = 20), parameters_lightgbm, platform, Tuning_Method, True)

    with open(filename, 'wb') as f:
        pickle.dump(model, f)

    with open(filename, 'rb') as f:
        file_content = f.read()  # Read the file content into bytes
        st.download_button(label='Download Trained Model', data=file_content, file_name=filename)
    
    return x_train, model

def plot_sentiment_distribution(data, platform):
    # Define the sentiment categories based on the scaled values
    conditions = [
        data[f'{platform}_scaled'] < 0,
        data[f'{platform}_scaled'] > 0,
        data[f'{platform}_scaled'] == 0
    ]
    choices = ['Negative', 'Positive', 'Neutral']
    sentiment_series = np.select(conditions, choices, default='Neutral')

    # Define custom colors for each sentiment
    colors = {
        'Negative': 'red',     
        'Positive': 'green',   
        'Neutral': 'blue'      
    }

    # Plotting the pie chart with the mapped colors
    fig = px.pie(
        names=sentiment_series,
        title=f'Sentiment Distribution for News on {platform}',
        color=sentiment_series,
        color_discrete_map=colors
    )
    st.plotly_chart(fig)

def main():
    st.title("Sentiment Analysis Across Social Media Platform")
    data = load_data()
    news_numeric_and_boolean = data.select_dtypes(include=[np.number, 'bool']).dropna()
    parameters = user_input_features()
    platform = st.selectbox("Select Social Media Platform", ["Facebook", "GooglePlus", "LinkedIn"])
    
    if st.button("Analyze"):
        plot_sentiment_distribution(data, platform)
        train_or_load_model(news_numeric_and_boolean, platform, parameters)
        x_cols = [x for x in data.columns if x not in ['Facebook_scaled','LinkedIn_scaled','GooglePlus_scaled']]

        x = data[x_cols]
        y = data[[f'{platform}_scaled']]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=23)
    
        perform_ml_algorithm(x_train, x_test, y_train, y_test, LGBMRegressor(random_state = 20), parameters, platform, Tuning_Method, False)
        
            
if __name__ == "__main__":
    main()
