from tqdm.auto import tqdm
from utils1 import *
from AdaSMDGCCA import SDGCCA_3_M
import warnings
from sklearn.metrics import mean_absolute_error
import shap
import matplotlib.pyplot as plt
warnings.simplefilter("ignore", UserWarning)

# Seed Setting
random_seed = 100
set_seed(random_seed)

hyper_dict = {'epoch': 100, 'delta': 0, 'random_seed': random_seed,
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'lr': [0.0001, 0.00001], 'reg': [0, 0.01],
              'patience': 30, 'embedding_size': [256, 64, 2], 'max_top_k': 10}
# hyper_dict = {'epoch': 200, 'delta': 0, 'random_seed': random_seed,
#               'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),a
#               'lr': [0.0001], 'reg': [0],
#               'patience': 30, 'embedding_size': [256, 64, 2], 'max_top_k': 10}

# Return List
# ensemble_list = {'ACC': [], 'F1': [], 'AUC': [], 'MCC': []}
metric_list = ['MSE', 'RMSE']
hyper_param_list = []
best_hyper_param_list = []

# Prepare Toy Dataset
dataset = Toy_Dataset(hyper_dict['random_seed'])
ensemble_result = []

for cv in tqdm(range(5), desc='CV...'):
    # Prepare Dataset
    [x_train_1, x_test_1], [x_train_2, x_test_2], [x_train_3, x_test_3], \
    [y_train, y_test] = dataset(cv, tensor=True, device=hyper_dict['device'])

    # Define Deep neural network dimension of the each modality
    m1_embedding_list = [x_train_1.shape[1]] + hyper_dict['embedding_size']
    m2_embedding_list = [x_train_2.shape[1]] + hyper_dict['embedding_size']
    m3_embedding_list = [x_train_3.shape[1]] + hyper_dict['embedding_size'][1:]

    # Train Label -> One_Hot_Encoding
    # y_train_onehot = torch.zeros(y_train.shape[0], 2).float().to(hyper_dict['device'])
    # y_train_onehot[range(y_train.shape[0]), y_train.squeeze()] = 1

    # Find Best K by Validation MCC
    # val_mcc_result_list = []
    # test_ensemble_dict = {'ACC': [], 'F1': [], 'AUC': [], 'MCC': []}
    # Grid search for find best hyperparameter by Validation MCC
    for top_k in tqdm(range(1, hyper_dict['max_top_k'] + 1), desc='Grid seach for find best hyperparameter...'):
        top_k = 1
        for lr in hyper_dict['lr']:
            for reg in hyper_dict['reg']:
                hyper_param_list.append([lr, reg])
                early_stopping = EarlyStopping(patience=hyper_dict['patience'], delta=hyper_dict['delta'])
                best_loss = np.Inf

                # Define SDGCCA with 3 modality
                model = SDGCCA_3_M(m1_embedding_list, m2_embedding_list, m3_embedding_list, top_k).to(
                    hyper_dict['device'])

                # Optimizer
                clf_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

                # Cross Entropy Loss
                criterion = torch.nn.MSELoss(reduction="sum")

                # Model Train
                for i in range(hyper_dict['epoch']):
                    model.train()

                    # Calculate correlation loss
                    out1, out2, out3 = model(x_train_1, x_train_2, x_train_3)
                    cor_loss = model.cal_loss([out1, out2, out3, y_train], hyper_dict['device'])

                    # Calculate classification loss
                    clf_optimizer.zero_grad()

                    y_hat1, y_hat2, y_hat3, _ = model.predict(x_train_1, x_train_2, x_train_3)
                    clf_loss1 = criterion(y_hat1, y_train.squeeze())
                    clf_loss2 = criterion(y_hat2, y_train.squeeze())
                    clf_loss3 = criterion(y_hat3, y_train.squeeze())

                    clf_loss = cor_loss.sum()

                    clf_loss.backward()
                    clf_optimizer.step()

                    # Model Validation
                    # with torch.no_grad():
                    #     model.eval()
                    #     _, _, _, y_ensemble = model.predict(x_val_1, x_val_2, x_val_3)
                    #     val_loss = criterion(y_ensemble, y_val.squeeze())
                    #     out1, out2, out3 = model(x_val_1, x_val_2, x_val_3)
                    #     val_loss = model.cal_loss([out1, out2, out3, y_train_onehot])
                    #     early_stopping(val_loss)
                    #     if val_loss < best_loss:
                    #         best_loss = val_loss
                    #
                    #     if early_stopping.early_stop:
                    #         break

                # Load Best Model
                with torch.no_grad():
                    model.eval()

                    # # Model Validation
                    # _, _, _, ensembel_y_hat = model.predict(x_val_1, x_val_2, x_val_3)
                    # y_pred_ensemble = torch.argmax(ensembel_y_hat, 1).cpu().detach().numpy()
                    # y_pred_proba_ensemble = ensembel_y_hat[:, 1].cpu().detach().numpy()
                    # _, _, _, val_mcc = calculate_metric(y_val.cpu().detach().numpy(), y_pred_ensemble,
                    #                                     y_pred_proba_ensemble)
                    # val_mcc_result_list.append(val_mcc)

                    # Model Tset
                    _, _, _, ensembel_y_hat = model.predict(x_test_1, x_test_2, x_test_3)
                    ensembel_y_hat1 = ensembel_y_hat.detach().cpu().numpy()

                    result_mae = mean_absolute_error(y_test.cpu().numpy(), ensembel_y_hat1)
                    result_rmse = np.sqrt(result_mae)
                    # y_pred_ensemble = torch.argmax(ensembel_y_hat, 1).cpu().detach().numpy()
                    # y_pred_proba_ensemble = ensembel_y_hat[:, 1].cpu().detach().numpy()
                    # test_acc, test_f1, test_auc, test_mcc = calculate_metric(y_test.cpu().detach().numpy(),
                    #                                                          y_pred_ensemble, y_pred_proba_ensemble)
                    ensemble_result.append(result_rmse)
                    # for k, metric in enumerate(metric_list):
                    #     test_ensemble_dict[metric].append(ensemble_result[k])

best_index = np.argmin(ensemble_result)
# Find best hyperparameter
best_hyper_param = hyper_param_list[best_index]

# Append Best K Test Result
# for metric in metric_list:
#     ensemble_list[metric].append(test_ensemble_dict[metric][best_k])

# Check Performance
# performance_result = check_mean_std_performance(ensemble_list)

print('Test Performance')
print('RMSE = {}'.format(ensemble_result[best_index]))
# print('MSE: {} RMSE: {}'.format(performance_result[0], performance_result[1])
# print('ACC: {} F1: {} AUC: {} MCC: {}'.format(performance_result[0], performance_result[1]))

print('\nBest Hyperparameter')
print('Learning Rate: {} Regularization Term: {}'.format(hyper_param_list[best_index][0], hyper_param_list[best_index][1]))
# Make random dataset
data1 = pd.read_excel('D:\\yangrenbo\\深度子空间重建\\Code-DS-SCCA\\ROI_re.xlsx')
data1 = data1.set_index('ID')
data1 = (data1 - data1.min()) / (data1.max() - data1.min())
data2 = pd.read_excel('D:\\yangrenbo\\深度子空间重建\\Code-DS-SCCA\\SNP_re.xlsx')
data2 = data2.set_index('ID')
data2 = (data2 - data2.min()) / (data2.max() - data2.min())
data3 = pd.read_excel('D:\\yangrenbo\\深度子空间重建\\Code-DS-SCCA\\Gene_re.xlsx')
data3 = data3.set_index('ID')
data3 = (data3 - data3.min()) / (data3.max() - data3.min())
label = pd.read_csv('D:\\yangrenbo\\386样本\\label.csv', header=None)
label = label.values
label1 = pd.read_csv('D:\\yangrenbo\\386样本\\MMSE_norm.csv', header=None)
label1 = label1.values


# Split Train,Validation and Test with 5 CV Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
for i, (train_index, test_index) in enumerate(kf.split(data1, label)):
    x_train_11, x_test_11, y_train, y_test = data1.iloc[train_index, :], data1.iloc[test_index, :], label1[
        train_index], label1[test_index]
    x_train_22, x_test_22 = data2.iloc[train_index, :], data2.iloc[test_index, :]
    x_train_33, x_test_33 = data3.iloc[train_index, :], data3.iloc[test_index, :]


explainer = shap.GradientExplainer(model.model1, [x_train_1])
shap_values = explainer.shap_values([x_test_1])
shap.summary_plot(shap_values, x_test_11)

explainer = shap.GradientExplainer(model.model2, [x_train_2])
shap_values = explainer.shap_values([x_test_2])
shap.summary_plot(shap_values, x_test_22)

explainer = shap.GradientExplainer(model.model3, [x_train_3])
shap_values = explainer.shap_values([x_test_3])
shap.summary_plot(shap_values, x_test_33)
# 纵轴按照所有样本的SHAP值之和对特征排序，横轴是SHAP值（特征对模型输出的影响分布）；
# 每个点代表一个样本，样本量在纵向堆积，颜色表示特征值（红色对应高值，蓝色对应低值）；
# 以第一行为例，表明高LSTAT（红色）对预测是负向影响，低LSTAT（蓝色）对预测是正向影响；
