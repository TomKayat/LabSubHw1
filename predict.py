from hw1_main import prep_df
import sys
import pickle


def generate_prediction(test_data, model, pid_test):
    pred = model.predict(test_data)
    test_data["id"] = pid_test
    test_data["id"] = test_data["id"].map(lambda x: x.split(".")[0])
    test_data["prediction"] = pred
    result = test_data[["id", "prediction"]]
    result.to_csv("prediction.csv", index=False)
    print("Done generating the prediction")


def load_data(path):
    coagulants = ["Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
    acid_base_status = ["pH", "BaseExcess", "HCO3", "PaCO2", "Lactate", "Chloride"]
    kidney = ["BUN", "Calcium", "Creatinine", "Magnesium", "Phosphate"]
    liver = ["Bilirubin_direct", "AST", "Alkalinephos", "Bilirubin_total"]
    oxygen = ["SaO2", "EtCO2"]  ##"FiO2"
    general = ["Glucose"]
    heart = ["Potassium", "TroponinI"]
    categories = [coagulants, acid_base_status, kidney, liver, oxygen, general, heart]
    categories_names = ["coagulants", "acid_base_status", "kidney", "liver", "oxygen", "general", "heart"]
    featruer_ranges_dict = {"Hct": [35, 49],  # "Hct" : {0: [34.9,44.5], 1: [38.8,50]},
                            "Hgb": [12.0, 17.5],  # {0:[12.0, 15.5], 1: [ 13.5, 17.5]}
                            "PTT": [25, 35],
                            "WBC": [4.5, 11],
                            "Fibrinogen": [200, 40],
                            "Platelets": [150, 450],
                            "pH": [7.35, 7.45],
                            "BaseExcess": [-2, 2],
                            "HCO3": [22, 30],
                            "PaCO2": [35, 45],
                            "Lactate": [4.5, 19.8],
                            "Chloride": [96, 106],
                            "BUN": [6, 20],
                            "Creatinine": [0.5, 1.3],  # {0: [0.5,1.1], 1:[0.6,1.3]}
                            "Calcium": [8.6, 10.2],
                            "Magnesium": [0.85, 1.1],
                            "Phosphate": [2.5, 4.5],
                            "Bilirubin_total": [0.3, 1.2],
                            "Bilirubin_direct": [0.0, 0.3],
                            "AST": [10, 40],
                            "Alkalinephos": [44, 147],
                            "SaO2": [95, 100],
                            "EtCO2": [35, 45],
                            "Glucose": [70, 140],
                            "Potassium": [3.5, 5.0],
                            "TroponinI": [0, 0.04]
                            # "FiO2" [21,21]  21% in room air, but can be higher in supplemental oxygen therapy
                            }
    test_data, pid = prep_df(categories=categories, categories_names=categories_names,
                                featruer_ranges_dict=featruer_ranges_dict, path=path)
    test_data = test_data.drop(["Label"], axis=1)
    model_file = open('XGB_small_model.pkl', 'rb')
    model = pickle.load(model_file)
    generate_prediction(test_data, model, pid)



def main(path):
    load_data(path)


if __name__ == '__main__':
    main(sys.argv[1])
