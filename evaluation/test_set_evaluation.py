import pandas as pd
import os
import shutil
from time import sleep
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean

# predict class for all test data
setups = ['orig', 'augmented', 'synthetic']
cvs = [1, 2, 3, 4, 5]
syndromes = [0, 1, 2, 3, 12]

encodings_path = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/gestaltmatcher/encodings.csv"
target_path = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/test_accuracies/"

os.chdir('synthesis/gestaltmatcher')

for setup in setups:
    for cv in cvs:
        for syndrome in syndromes:
            os.system(f"python predict.py --num_classes 5 --data_dir ../../data/cross_validation/{cv}/test/{syndrome} --model_path saved_models/classifier/best_models/1_{cv}_{setup}.pt")
            sleep(5)  # otherwise the routine doesn't wait for GestaltMatcher process to close
            encodings_target_path = os.path.join(target_path, setup, f"{cv}_{syndrome}.csv")
            shutil.copy(encodings_path, encodings_target_path)

# get predictions
encodings_dir = "C:/Users/Manu/ownCloud/IGSB/thesis/synthesis/results/test_accuracies/"


# function to get index of class prediction
def find_index_of_max(lst):
    return lst.index(max(lst))


predictions = pd.DataFrame()
conf_matrices = {'setup': [], 'cv': [], 'conf_mat': [], 'accuracy': [], 'sensitivity_0': [],
                 'sensitivity_1': [], 'sensitivity_2': [], 'sensitivity_3': [], 'sensitivity_12': []}

for setup in setups:
    encodings_dir_setup = os.path.join(encodings_dir, setup)
    for cv in cvs:
        predictions_cv = pd.DataFrame()
        for syndrome in syndromes:
            f = os.path.join(encodings_dir_setup, f"{cv}_{syndrome}.csv")
            encodings = pd.read_csv(f, delimiter=";", converters={'class_conf': pd.eval})
            encodings['pred_index'] = encodings['class_conf'].apply(find_index_of_max)
            encodings['pred'] = encodings['pred_index'].apply(lambda x: syndromes[x])
            encodings = encodings.drop(axis=1, labels=['representations', 'img_name', 'pred_index', 'class_conf'])
            encodings['true'] = syndrome
            encodings['cv'] = cv
            encodings['setup'] = setup
            predictions = pd.concat([predictions, encodings], ignore_index=True)
            predictions_cv = pd.concat([predictions_cv, encodings], ignore_index=True)
            print(f"Setup: {setup}, fold: {cv}, syndrome: {syndrome}")

        # calculate confusion matrix
        conf_matrices['setup'].append(setup)
        conf_matrices['cv'].append(cv)
        cm = confusion_matrix(predictions_cv['true'], predictions_cv['pred'],
                              labels=([0, 1, 2, 3, 12]))
        class_names = ['CdLS', 'WBS', 'KS', 'AS', 'HPMRS']
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names).astype(int)
        acc = (df_cm.iloc[0, 0] + df_cm.iloc[1, 1] + df_cm.iloc[2, 2] + df_cm.iloc[3, 3] + df_cm.iloc[4, 4]) / df_cm.values.sum()
        sensitivity_0 = df_cm.iloc[0, 0] / df_cm.iloc[0, :].values.sum()
        sensitivity_1 = df_cm.iloc[1, 1] / df_cm.iloc[1, :].values.sum()
        sensitivity_2 = df_cm.iloc[2, 2] / df_cm.iloc[2, :].values.sum()
        sensitivity_3 = df_cm.iloc[3, 3] / df_cm.iloc[3, :].values.sum()
        sensitivity_12 = df_cm.iloc[4, 4] / df_cm.iloc[4, :].values.sum()

        conf_matrices['conf_mat'].append(df_cm)
        conf_matrices['accuracy'].append(acc)
        conf_matrices['sensitivity_0'].append(sensitivity_0)
        conf_matrices['sensitivity_1'].append(sensitivity_1)
        conf_matrices['sensitivity_2'].append(sensitivity_2)
        conf_matrices['sensitivity_3'].append(sensitivity_3)
        conf_matrices['sensitivity_12'].append(sensitivity_12)

conf_mats = pd.DataFrame(conf_matrices)
metrics = pd.DataFrame(conf_matrices).drop(axis=1, labels='conf_mat')
metrics.to_csv(os.path.join(target_path, 'metrics.csv'))

# plot confusion matrix
ix = 0
for setup in setups:
    for cv in cvs:
        plt.figure(figsize=(10, 7))
        class_names = ['CdLS', 'WBS', 'KS', 'AS', 'HPMRS']
        df_cm = conf_mats['conf_mat'][ix]
        heatmap = sns.heatmap(df_cm, cmap="crest", annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        fig_ = heatmap.get_figure()
        plt.savefig(os.path.join(target_path, f'cm_{setup}_{cv}.png'))
        plt.close()
        ix += 1


def df_mean(list_of_dfs):
    avg_df = (list_of_dfs[0] + list_of_dfs[1] + list_of_dfs[2] + list_of_dfs[3] + list_of_dfs[4]) / 5
    return avg_df


# mean confusion matrices
mean_cm_orig = (conf_mats['conf_mat'][0] + conf_mats['conf_mat'][1] + conf_mats['conf_mat'][2] + conf_mats['conf_mat'][3] + conf_mats['conf_mat'][4]) / 5
mean_cm_augmented = (conf_mats['conf_mat'][5] + conf_mats['conf_mat'][6] + conf_mats['conf_mat'][7] + conf_mats['conf_mat'][8] + conf_mats['conf_mat'][9]) / 5
mean_cm_synthetic = (conf_mats['conf_mat'][10] + conf_mats['conf_mat'][11] + conf_mats['conf_mat'][12] + conf_mats['conf_mat'][13] + conf_mats['conf_mat'][14]) / 5
mean_metrics = {'setup': [], 'cm': [], 'accuracy': [], 'sensitivity_0': [], 'sensitivity_1': [], 'sensitivity_2': [], 'sensitivity_3': [], 'sensitivity_12': []}
mean_metrics['cm'].append(mean_cm_orig)
mean_metrics['cm'].append(mean_cm_augmented)
mean_metrics['cm'].append(mean_cm_synthetic)

ix2 = 0
for setup in setups:
    mean_metrics['setup'].append(setup)
    mean_metrics['accuracy'].append(mean(conf_mats['accuracy'][conf_mats['setup'] == setup]))
    mean_metrics['sensitivity_0'].append(mean(conf_mats['sensitivity_0'][conf_mats['setup'] == setup]))
    mean_metrics['sensitivity_1'].append(mean(conf_mats['sensitivity_1'][conf_mats['setup'] == setup]))
    mean_metrics['sensitivity_2'].append(mean(conf_mats['sensitivity_2'][conf_mats['setup'] == setup]))
    mean_metrics['sensitivity_3'].append(mean(conf_mats['sensitivity_3'][conf_mats['setup'] == setup]))
    mean_metrics['sensitivity_12'].append(mean(conf_mats['sensitivity_12'][conf_mats['setup'] == setup]))

    plt.figure(figsize=(10, 7))
    class_names = ['CdLS', 'WBS', 'KS', 'AS', 'HPMRS']
    df_cm = mean_metrics['cm'][5]
    heatmap = sns.heatmap(df_cm, cmap="crest", annot=True, fmt=".1f")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig_ = heatmap.get_figure()
    plt.savefig(os.path.join(target_path, f'mean_cm_{setup}.png'))
    plt.close()
    ix2 += 1

mean_metrics_df = pd.DataFrame(mean_metrics)
mean_metrics_df.to_csv(os.path.join(target_path, 'mean_metrics.csv'))
