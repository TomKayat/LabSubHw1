import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
# from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import optuna
from optuna.trial import TrialState
from time import time
from sklearn.tree import DecisionTreeClassifier
# from explore import t_test_exploration
import joblib
import os
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.neighbors import KNeighborsClassifier


import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go


# from sqlalchemy.sql.elements import Null
import warnings
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
import math

RESPRANGE = 6
HRRANGE = 40
UPPERTEMP = 38.2 ## 37.6 it was 37.8
LOWERTEMP = 35.5
LATESTHOURS = 15


warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', 100000)
pd.set_option('display.max_columns', 100000)


def find_abnormals(row, category, featruer_ranges_dict):
    counter = 0
    for feat_val, col in zip(row, category):
        lower = featruer_ranges_dict[col][0]
        uper = featruer_ranges_dict[col][1]
        if pd.isna(feat_val):
            continue
        elif lower > feat_val or uper < feat_val:
            counter += 1
    return counter


#  unseen in fit
def check_for_abnormal(data, categories, categories_names, new, featruer_ranges_dict):
    for category, category_name in zip(categories, categories_names):
        data[category_name + "_abnormal"] = data[category].apply(
            lambda x: find_abnormals(x, category, featruer_ranges_dict), axis=1)
        ab_counter = data[category_name + "_abnormal"].max()
        new[category_name + "_abnormal"] = ab_counter
    return new


def fill_Resp(current_df,data_dict ):
    resp_min = current_df["Resp"].min()
    resp_max = current_df["Resp"].max()
    if resp_min == resp_max:
        data_dict["Resp_NromMax"] = current_df["Resp"].max() / RESPRANGE
        data_dict["Resp_NromMin"] = current_df["Resp"].min() / RESPRANGE
    else:
        data_dict["Resp_NromMax"] = current_df["Resp"].max() / (resp_max - resp_min)
        data_dict["Resp_NromMin"] = current_df["Resp"].min() / (resp_max - resp_min)

    return data_dict


def fill_bp(current_df, data_dict):
    if len(list(current_df["SBP"].dropna())) > 0:
        data_dict["SBP_last"] = list(current_df["SBP"].dropna())[-1]
    else:
        data_dict["SBP_last"] = np.NaN
    if len(list(current_df["DBP"].dropna())) > 0:
        data_dict["DBP_last"] = list(current_df["DBP"].dropna())[-1]
    else:
        data_dict["DBP_last"] = np.NaN
    return data_dict



def load_data(categories, categories_names, featruer_ranges_dict, path):
    create_df = True
    for file in os.listdir(path):
        data_dict = {}
        current_df = pd.read_csv(os.path.join(path, file), sep='|')
        current_df = current_df.head(len(current_df[current_df['SepsisLabel']==0])+1)
        data_dict["ICULOS"] = current_df["ICULOS"].max()
        data_dict = check_for_abnormal(current_df, categories, categories_names, data_dict, featruer_ranges_dict)
        data_dict["Label"] = current_df["SepsisLabel"].max()
        data_dict["PID"] = file
        data_dict["O2Sat_min"] = current_df["O2Sat"].min()
        max_hr = current_df["HR"].max()
        data_dict["Fast_HR"] = 1 if max_hr > 90 else 0
        resp_mean = current_df["Resp"].mean()
        data_dict["Resp_mean"] = resp_mean
        data_dict = fill_bp(current_df, data_dict)

        colm = list(current_df.columns)
        for col in ["SepsisLabel"]: ## ,"Age", "Gender", "Unit1", "Unit2", "ICULOS", "HospAdmTime"
            colm.remove(col)
        for col in colm:
            data_dict[col + "_missing_tests"] = 1 - (len(current_df[col].dropna())/len(current_df[col]))

        df_columns = list(data_dict.keys())
        if create_df:
            data_df = pd.DataFrame(columns=df_columns)
            create_df = False

        data_df = pd.concat([data_df, pd.DataFrame([data_dict])], ignore_index=True)

    return data_df


def fill_by_similarty(dataframe, value_to_match, to_match, to_fill, range_toadd):
    twocoldf = dataframe[[to_match, to_fill]].dropna(how="any")
    df_sim = twocoldf[
        (twocoldf[to_match] <= value_to_match + range_toadd) & (twocoldf[to_match] >= value_to_match - range_toadd)][
        to_fill]
    if len(df_sim) == 0:
        value_to_fill = twocoldf[to_fill].iloc[(twocoldf[to_match] - value_to_match).abs().argsort()[:1]]
        value_to_fill = list(value_to_fill)[0]
    else:
        value_to_fill = df_sim.mean()
    return round(value_to_fill, 1)


def prep_df(categories, categories_names, featruer_ranges_dict, path):
    print(f"Start processing data from {path}")
    df = load_data(categories, categories_names, featruer_ranges_dict, path)
    pid = df["PID"]
    df = df.drop(["PID"], axis=1)
    range_of_values = 5

    df['DBP_last'] = df[["SBP_last", "DBP_last"]].apply(
        lambda x: fill_by_similarty(df, x[0], 'SBP_last', 'DBP_last', range_of_values) if pd.isna(x[1]) else x[1],
        axis=1)
    df['SBP_last'] = df[["DBP_last", "SBP_last"]].apply(
        lambda x: fill_by_similarty(df, x[0], 'DBP_last', 'SBP_last', range_of_values) if pd.isna(x[1]) else x[1],
        axis=1)

    fill_median = list(df.columns)
    df[fill_median] = df[fill_median].fillna(df[fill_median].median())
    return df, pid


def prepare_data_for_train(train_data, test_data):
    y_train = train_data["Label"]
    y_test = test_data["Label"]
    train_data = train_data.drop(["Label"], axis=1)
    test_data = test_data.drop(["Label"], axis=1)
    return train_data, y_train, test_data, y_test


def eval_f1(y_test, pred):
    return 1 - f1_score(y_test, pred)


def train_model(train_data, y_train, test_data, y_test, model, model_name="XGB", save=False):
    print(f"Training Model {model_name}")
    model.fit(train_data, y_train)
    pred = model.predict(test_data)
    f1 = f1_score(y_test, pred)
    training_pred = model.predict(train_data)
    training_f1 = f1_score(y_train, training_pred)
    print(f"F1 score on the TestSet: {f1}")
    print(f"F1 score on the TrainSet: {training_f1}")
    if save:
        pickle.dump(model, open("XGB_small_model.pkl", "wb"))
    return f1


def train_model_LogisticRegression(train_data, y_train, test_data, y_test, model):
    print(f"Training Model LogisticRegression")
    train_data =(train_data-train_data.min())/(train_data.max()-train_data.min())
    test_data =(test_data-test_data.min())/(test_data.max()-test_data.min())
    model.fit(train_data, y_train)
    pred = model.predict(test_data)
    f1 = f1_score(y_test, pred)
    training_pred = model.predict(train_data)
    training_f1 = f1_score(y_train, training_pred)
    print(f"F1 score on the TestSet: {f1}")
    print(f"F1 score on the TrainSet: {training_f1}")
    return f1



def train_model_KNN(train_data, y_train, test_data, y_test, model=KNeighborsClassifier(n_neighbors=3)):
    print(f"Training Model KNN")
    model.fit(train_data, y_train)
    pred = model.predict(test_data)
    f1 = f1_score(y_test, pred)
    training_pred = model.predict(train_data)
    training_f1 = f1_score(y_train, training_pred)
    print(f"F1 score on the TestSet: {f1}")
    print(f"F1 score on the TrainSet: {training_f1}")
    pickle.dump(model, open("KNN_small_model.pkl", "wb"))

    return f1


def train_random_forest(train_data, y_train, test_data, y_test, model):
    print(f"Training Model Random Forest")
    model.fit(train_data, y_train)
    pred = model.predict(test_data)
    f1 = f1_score(y_test, pred)
    training_pred = model.predict(train_data)
    training_f1 = f1_score(y_train, training_pred)
    print(f"F1 score on the TestSet: {f1}")
    print(f"F1 score on the TrainSet: {training_f1}")
    pickle.dump(model, open("RandomForest_small_model.pkl", "wb"))
    return f1


def randomForest_check_parms(trial, train_data, y_train, test_data, y_test):
    criterions = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_features = trial.suggest_categorical("max_features", ["sqrt", None, "log2"])
    class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
    estimators = trial.suggest_int("n_estimators", 50, 200)
    return train_random_forest(train_data, y_train, test_data, y_test, RandomForestClassifier(criterion=criterions,
                                                                                      max_features=max_features,
                                                                                      class_weight=class_weight,
                                                                                      n_estimators=estimators))


def optuna_random_forest(train_data, y_train, test_data, y_test):
    save_path = "RandomForest_small_optuna_study_batch.pkl"
    run_name = f'RandomForest_small_trail_res_{time()}.txt'
    for x in range(8):
        if os.path.isfile(save_path):
            study = joblib.load(save_path)
        else:
            study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: randomForest_check_parms(trial, train_data, y_train, test_data, y_test), n_trials=500,
                       timeout=1800)
        if os.path.isfile(save_path):
            os.remove(save_path)
        joblib.dump(study, save_path)
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        if os.path.isfile(run_name):
            os.remove(run_name)
        with open(run_name, 'w') as f:
            for key, value in trial.params.items():
                f.write("    {}: {}".format(key, value))
                print("    {}: {}".format(key, value))





def check_params(trial, train_data, y_train, test_data, y_test,):
    boosters = trial.suggest_categorical("booster", ["gbtree", "dart"])
    LR = trial.suggest_float("learning_rate", 0.05, 0.5)
    alphas = trial.suggest_int("alpha", 0, 50)
    child_wight = trial.suggest_int("min_child_weight", 1, 10)
    # n_estimators(200), child_wight(1), sub_sample, random_state
    subsamples = trial.suggest_float("subsample", 0.5, 1)
    estimators = trial.suggest_int("n_estimators", 50, 200)
    print(f"boosters={boosters}, LR={LR}, alphas={alphas},"
          f" child_wight={child_wight}, subsamples={subsamples}, estimators={estimators}")

    return train_model(train_data, y_train, test_data, y_test, XGBClassifier(learning_rate=LR, eval_metric=eval_f1,
                                                                      booster=boosters,
                                                                      alpha=alphas,
                                                                      min_child_weight=child_wight,
                                                                      subsample=subsamples,
                                                                      n_estimators = estimators))


def optuna_xgb(train_data, y_train, test_data, y_test):
    save_path = "xgb_small_optuna_study_batch.pkl"
    run_name = f'trail_res_{time()}.txt'
    for x in range(30):
        if os.path.isfile(save_path):
            study = joblib.load(save_path)
        else:
            study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: check_params(trial, train_data, y_train, test_data, y_test), n_trials=5000,
                       timeout=1800)
        if os.path.isfile(save_path):
            os.remove(save_path)
        joblib.dump(study, save_path)
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        if os.path.isfile(run_name):
            os.remove(run_name)
        with open(run_name, 'w') as f:
            for key, value in trial.params.items():
                f.write("    {}: {}".format(key, value))
                print("    {}: {}".format(key, value))




def optuna_logistic(train_data, y_train, test_data, y_test):
    save_path = "logistic_small_optuna_study_batch.pkl"
    run_name = f'Logistic_small_trail_res_{time()}.txt'
    for x in range(8):
        if os.path.isfile(save_path):
            study = joblib.load(save_path)
        else:
            study = optuna.create_study(direction="maximize")  ##, sampler=optuna.samplers.TPESampler()
        study.optimize(lambda trial: check_params_logistic(trial, train_data, y_train, test_data, y_test), n_trials=500,
                       timeout=1800)
        if os.path.isfile(save_path):
            os.remove(save_path)
        joblib.dump(study, save_path)
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        if os.path.isfile(run_name):
            os.remove(run_name)
        with open(run_name, 'w') as f:
            for key, value in trial.params.items():
                f.write("    {}: {}".format(key, value))
                print("    {}: {}".format(key, value))



def check_params_logistic(trial, train_data, y_train, test_data, y_test):
    class_weights = trial.suggest_categorical("class_weight", ["balanced", None])
    regularization = trial.suggest_float("C", 0, 1)
    max_iter = trial.suggest_int("max_iter", 100, 250)
    penalty_and_solver = [{"penalty": "l2", "solver":"lbfgs"}, {"penalty": "none", "solver":"lbfgs"},
                           {"penalty": "l2", "solver":"liblinear"}, {"penalty": "l1", "solver":"liblinear"},
                           {"penalty": "none", "solver": "saga"},
                           {"penalty": "l2", "solver": "saga"}, {"penalty": "l1", "solver": "saga"}
                           ]
    index = trial.suggest_int("", 0, len(penalty_and_solver)-1)
    config = penalty_and_solver[index]
    return train_model_LogisticRegression(train_data, y_train, test_data, y_test,
                                          LogisticRegression(penalty=config["penalty"],
                                                             class_weight=class_weights,
                                                             solver=config["solver"],
                                                             max_iter=max_iter,
                                                             C=regularization))


# def main():
#     print("RF,LR, with all feat, right, small")
#     save_data = False
#     coagulants = ["Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
#     acid_base_status = ["pH", "BaseExcess", "HCO3", "PaCO2", "Lactate", "Chloride"]
#     kidney = ["BUN", "Calcium", "Creatinine", "Magnesium", "Phosphate"]
#     liver = ["Bilirubin_direct", "AST", "Alkalinephos", "Bilirubin_total"]
#     oxygen = ["SaO2", "EtCO2"]  ##"FiO2"
#     general = ["Glucose"]
#     heart = ["Potassium", "TroponinI"]
#     categories = [coagulants, acid_base_status, kidney, liver, oxygen, general, heart]
#     categories_names = ["coagulants", "acid_base_status", "kidney", "liver", "oxygen", "general", "heart"]
#     featruer_ranges_dict = {"Hct": [35, 49],  # "Hct" : {0: [34.9,44.5], 1: [38.8,50]},
#                             "Hgb": [12.0, 17.5],  # {0:[12.0, 15.5], 1: [ 13.5, 17.5]}
#                             "PTT": [25, 35],
#                             "WBC": [4.5, 11],
#                             "Fibrinogen": [200, 40],
#                             "Platelets": [150, 450],
#                             "pH": [7.35, 7.45],
#                             "BaseExcess": [-2, 2],
#                             "HCO3": [22, 30],
#                             "PaCO2": [35, 45],
#                             "Lactate": [4.5, 19.8],
#                             "Chloride": [96, 106],
#                             "BUN": [6, 20],
#                             "Creatinine": [0.5, 1.3],  # {0: [0.5,1.1], 1:[0.6,1.3]}
#                             "Calcium": [8.6, 10.2],
#                             "Magnesium": [0.85, 1.1],
#                             "Phosphate": [2.5, 4.5],
#                             "Bilirubin_total": [0.3, 1.2],
#                             "Bilirubin_direct": [0.0, 0.3],
#                             "AST": [10, 40],
#                             "Alkalinephos": [44, 147],
#                             "SaO2": [95, 100],
#                             "EtCO2": [35, 45],
#                             "Glucose": [70, 140],
#                             "Potassium": [3.5, 5.0],
#                             "TroponinI": [0, 0.04]
#                             # "FiO2" [21,21]  21% in room air, but can be higher in supplemental oxygen therapy
#                             }
#     if save_data:
#         train_data, _ = prep_df(categories, categories_names, featruer_ranges_dict, r"data/train/")
#         train_data.to_csv("Final_Train_Data.csv", index=False)
#         print("Done preparing training set")
#         test_data, pid = prep_df(categories, categories_names, featruer_ranges_dict, r"data/test/")
#         test_data.to_csv("Final_Test_Data.csv", index=False)
#         pid.to_csv("Final_Test_PID.csv", index=False)
#         print("Done preparing Test set")
#     else:
#         train_data = pd.read_csv(r"Final_Train_Data.csv")
#         test_data = pd.read_csv(r"Final_Test_Data.csv")
#         pid = pd.read_csv(r"Final_Test_PID.csv")
#
#     train_data, y_train, test_data, y_test = prepare_data_for_train(train_data, test_data)
#     train_random_forest(train_data, y_train, test_data, y_test, model=RandomForestClassifier(criterion="gini",
#                                                                                              max_features=None,
#                                                                                              class_weight=None,
#                                                                                              n_estimators=75))
#     # # optuna_xgb(train_data, y_train, test_data, y_test)
#     # print("Optuna logistic")
#     # optuna_logistic(train_data, y_train, test_data, y_test)
#     # print("Optuna forrst")
#     # optuna_random_forest(train_data, y_train, test_data, y_test)
#
#     # train_model(train_data, y_train, test_data, y_test, model=XGBClassifier(booster="dart",
#     #                                                                         learning_rate=0.20067462290570948 ,
#     #                                                                         reg_alpha=13,
#     #                                                                         min_child_weight=2,
#     #                                                                         subsample=0.7605565076595467,
#     #                                                                         n_estimators=151),
#     #             model_name="XGB", save=True)
#
#     # train_model_KNN(train_data, y_train, test_data, y_test, model=KNeighborsClassifier(n_neighbors=3))
#
#     # booster: dart
#     # learning_rate: 0.20067462290570948
#     # alpha: 13    min_child_weight: 2    subsample: 0.7605565076595467    n_estimators: 151
#
#
# if __name__ == '__main__':
#     main()
