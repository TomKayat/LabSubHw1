import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
ALPHA = 0.05
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


def group_analysis(data, col, value1, range=False):
    "post analysis confusion_matrix printing"
    load_pred = pd.read_csv("XGB_prediction.csv")
    # data["Label"] = labels
    data["prediction"] = load_pred["prediction"]
    if not range:
        data1 = data[data[col] == value1][["prediction", "Label"]] # 0 = female
        data2 = data[data[col] != value1][["prediction", "Label"]]
    else:
        data1 = data[data[col] < value1][["prediction", "Label"]]
        data2 = data[data[col] >= value1][["prediction", "Label"]]
    cm1 = confusion_matrix(data1["Label"], data1["prediction"])#.ravel()
    # cm1 = cm1.astype('float') / cm1.sum(axis=1)
    cm2 = confusion_matrix(data2["Label"], data2["prediction"])#.ravel()
    # cm2 = cm2.astype('float') / cm2.sum(axis=1)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels = list(data2["Label"].unique()))
    disp1.plot()
    plt.show()
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels = list(data2["Label"].unique()))
    disp2.plot()
    plt.show()


def data_analysis(data):
    "histogram printing"
    spesis = data[data["Label"]==1]
    no_spesis = data[data["Label"]==0]

    for col in data.columns:
        # sns.histplot(data=data, x=col, hue="Label", stat='probability').set(title=f"Histogram for {col}")
        sns.histplot(data=spesis, x=col, stat='percent')
        sns.histplot(data=no_spesis, x=col, stat='percent')
        plt.show()
        break
        # print(data[col].describe())


def old_main():
    train_data = pd.read_csv(r"train_data.csv")
    # sns.histplot(data=train_data, x="acid_base_status_abnormal", hue="Label").set(title=f"Histogram for {'acid_base_status_abnormal'}")
    # plt.show()
    spesis = train_data[train_data["Label"] == 1]
    no_spesis = train_data[train_data["Label"] == 0]
    continuous = ['ICULOS', 'ICULOS_in_range',  'O2Sat_min',
       'O2Sat_diff', 'Age', 'HR_NromMax', 'SBP_last', 'DBP_last',
       'Resp_NromMax', 'Resp_NromMin', 'Map_Min', 'HospAdmTime_value',
        'Glucose_max', 'Glucose_min',
       'Glucose_std', 'HR_mean', 'O2Sat_mean', 'Temp_mean', 'SBP_mean',
       'MAP_mean', 'Resp_mean', 'Glucose_mean', 'HospAdmTime_mean', 'ICULOS_mean', 'coagulants_abnormal_mean',
       'acid_base_status_abnormal_mean', 'kidney_abnormal_mean',
       'liver_abnormal_mean', 'oxygen_abnormal_mean', 'general_abnormal_mean',
       'heart_abnormal_mean']

    for col in continuous:
        sns.histplot(data=spesis, x=col, stat='density', label="spesis", binwidth=0.8, color="skyblue")
        sns.histplot(data=no_spesis, x=col, stat='density', label="no_spesis", binwidth=0.8, color="plum")
        plt.title(f"Histogram for {col}")
        plt.legend()  # "spesis", "no_spesis"
        plt.savefig(f'{col}.png')
        plt.show()
    categorical = ['coagulants_abnormal',
                   'acid_base_status_abnormal', 'kidney_abnormal', 'liver_abnormal',
                   'oxygen_abnormal', 'general_abnormal', 'heart_abnormal', 'Label', 'Unit1',
                   'Unit2', 'Gender','Extream_Temp']
    for col in categorical:
        try:
            sns.histplot(data=spesis, x=col, stat='percent', label="spesis", binwidth=1, color="skyblue")
            sns.histplot(data=no_spesis, x=col, stat='percent', label="no_spesis", binwidth=1, color="plum")
            plt.title(f"Histogram for {col}")
            plt.legend()  # "spesis", "no_spesis"
            plt.savefig(f'{col}.png')
            plt.show()
        except:
            print(col)
            pass


    # data_analysis(train_data)
    # check(train_data)

def main():
    train_data = pd.read_csv(r"train_data.csv")
    t_test_exploration(train_data)


def passed_test(p_value, col):
    if p_value < ALPHA/2:
        print(f"For the feature {col}, the P_value is {round(p_value,5)} (less than {ALPHA/2}). "
              f"Therefor, is a significant difference between the distribution of sepsis according to this feature."
              " We reject the null theory.")
        return "Reject"
    else:
        print(f"For the feature {col}, the P_value is {round(p_value,5)} (same or greater than {ALPHA/2}). "
              f"Therefor, is no significant difference between the distribution of sepsis according to this feature."
              " We dont reject the null theory.")
        return "DontReject"


def t_test_exploration(data):
    """
    we want to check if there are statisticly significant changes in the distibution among the sick and not sick
    paitnt in the training data
    :return:
    """
    significant = []
    not_significant = []
    sepsis = data[data["Label"] == 1]
    no_sepsis = data[data["Label"] == 0]
    cols_to_check = list(data.columns)
    cols_to_check.remove("Label")
    for col in cols_to_check:
        _, p_value = stats.ttest_ind(no_sepsis[col], sepsis[col], equal_var=False, nan_policy='omit')
        if "Reject" == passed_test(p_value,col):
            significant.append(col)
        else:
            not_significant.append(col)
    print(f"The columns we`ll use for the model are the ones with a significant change of distribution.\n"
          f"The columns are: {significant}")
    print(f"The not_significant are {not_significant}")
    return significant



def describe_featuers():
    train_data = pd.read_csv(r"train_data.csv")
    stat_df = train_data.describe(include='all')
    stat_df.to_csv("Feature_statistics.csv")

    # for col in train_data.columns:
    #     print(train_data[col].describe())

def pie_chart():
    data = pd.read_csv(r"train_data.csv")
    spesis = data[data["Label"] == 1]
    no_spesis = data[data["Label"] == 0]

    # man_sick = spesis[spesis["Gender"]==0].count()
    # woman_sick = spesis[spesis["Gender"]==1].count()
    # man_notsick = no_spesis[no_spesis["Gender"]==0].count()
    # woman_notsick = no_spesis[no_spesis["Gender"]==1].count()
    # info = [int(man_sick), int(woman_sick),int(man_notsick), int(woman_notsick)]
    # labels = ["Man_sepsis", "Woman_sepsis", "Man_no_sepsis", "Woman_no_sepsis"]
    # plt.pie(info, labels=labels,  autopct='%.0f%%')
    # plt.show()
    # cloumns = ["Gender", "Unit1", "Unit2"]
    # for col in cloumns:
    #     positive_sepsis = spesis[col]



def histograms():
    data = pd.read_csv(r"Final_Test_Data.csv")
    spesis = data[data["Label"] == 1]
    no_spesis = data[data["Label"] == 0]
    a = ['ICULOS', 'coagulants_abnormal', 'acid_base_status_abnormal',
       'kidney_abnormal', 'liver_abnormal', 'oxygen_abnormal',
        'heart_abnormal', 'O2Sat_min',
       'Resp_mean', 'SBP_last', 'DBP_last']
    b = ["SBP_last"]
    binary = ['Fast_HR', 'general_abnormal','HR_missing_tests',
       'O2Sat_missing_tests', 'Temp_missing_tests', 'SBP_missing_tests',
       'MAP_missing_tests', 'DBP_missing_tests', 'Resp_missing_tests',
       'EtCO2_missing_tests', 'BaseExcess_missing_tests', 'HCO3_missing_tests',
       'FiO2_missing_tests', 'pH_missing_tests', 'PaCO2_missing_tests',
       'SaO2_missing_tests', 'AST_missing_tests', 'BUN_missing_tests',
       'Alkalinephos_missing_tests', 'Calcium_missing_tests',
       'Chloride_missing_tests', 'Creatinine_missing_tests',
       'Bilirubin_direct_missing_tests', 'Glucose_missing_tests',
       'Lactate_missing_tests', 'Magnesium_missing_tests',
       'Phosphate_missing_tests', 'Potassium_missing_tests',
       'Bilirubin_total_missing_tests', 'TroponinI_missing_tests',
       'Hct_missing_tests', 'Hgb_missing_tests', 'PTT_missing_tests',
       'WBC_missing_tests', 'Fibrinogen_missing_tests',
       'Platelets_missing_tests', 'Age_missing_tests', 'Gender_missing_tests',
       'Unit1_missing_tests', 'Unit2_missing_tests',
       'HospAdmTime_missing_tests', 'ICULOS_missing_tests',
       'coagulants_abnormal_missing_tests',
       'acid_base_status_abnormal_missing_tests',
       'kidney_abnormal_missing_tests', 'liver_abnormal_missing_tests',
       'oxygen_abnormal_missing_tests', 'general_abnormal_missing_tests',
       'heart_abnormal_missing_tests']

    for col in binary:
        # sns.histplot(data=data, x=col, hue="Label", stat='probability').set(title=f"Histogram for {col}")
        # sns.barplot(data=spesis, x=col, stat='percent', color="skyblue", binwidth=0.5,label="Spesis") #histplot
        # sns.barplot(data=no_spesis, x=col, stat='percent', color="plum", binwidth=0.5, label="No Spesis")
        sns.barplot(data=data, y=col, x="Label", palette=["skyblue","plum"]) #histplot
        # sns.barplot(data=no_spesis, y=col, color="plum",  label="No Spesis")
        plt.title(f"Histogram for {col}")
        plt.legend()
        plt.savefig(f'{col}.png')
        plt.show()



def main():
    print("Explore")
    # histograms()
    # train_data = pd.read_csv(r"Final_Test_Data.csv")
    # print(train_data.columns)
    # t_test_exploration(train_data)

    # # sns.histplot(data=train_data, x="acid_base_status_abnormal", hue="Label").set(title=f"Histogram for {'acid_base_status_abnormal'}")
    # # plt.show()
    # # ['ICULOS', 'ICULOS_in_range', 'O2Sat_min',



    # #  'O2Sat_diff', 'Age', 'HR_NromMax', 'SBP_last', 'DBP_last',
    # #  'Resp_NromMax', 'Resp_NromMin', 'Map_Min', 'HospAdmTime_value',
    # #  'Glucose_max', 'Glucose_min',
    # #  'Glucose_std', 'HR_mean', 'O2Sat_mean', 'Temp_mean', 'SBP_mean',
    # #  'MAP_mean', 'Resp_mean', 'Glucose_mean', 'HospAdmTime_mean', 'ICULOS_mean', 'coagulants_abnormal_mean',
    # #  'acid_base_status_abnormal_mean', 'kidney_abnormal_mean',
    # #  'liver_abnormal_mean', 'oxygen_abnormal_mean', 'general_abnormal_mean',
    # #  'heart_abnormal_mean']
    # categorical = ['coagulants_abnormal',
    #                'acid_base_status_abnormal', 'kidney_abnormal', 'liver_abnormal',
    #                'oxygen_abnormal', 'general_abnormal', 'heart_abnormal', 'Label', 'Unit1',
    #                'Unit2', 'Gender','Extream_Temp']
    # col = "Gender"
    # spesis = train_data[train_data["Label"] == 1]
    # no_spesis = train_data[train_data["Label"] == 0]
    # sns.histplot(data=spesis, x=col, stat='density',label="spesis", binwidth=0.5, color="skyblue") ##stat='density'
    # sns.histplot(data=no_spesis, x=col, stat='density', label="no_spesis", binwidth=0.5, color="plum")
    # plt.title(f"Histogram for {col}")
    # plt.legend()
    # plt.show()
    # # pie_chart()
    # # describe_featuers()
    # group_analysis(test_data, "general_abnormal", 1, range=True)
    # group_analysis(test_data, "ICULOS", 10, range=True)


if __name__ == '__main__':
    main()

    # # main()
