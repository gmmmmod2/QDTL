import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score

from utils.other import saveOutput

def calculate_metrics(outputs, targets, is_Save = False):
    """
    计算回归任务的RMSE指标
    
    参数:
        outputs: 模型输出的张量，形状为 [batch_size, num_labels]
        targets: 真实标签的张量，形状为 [batch_size, num_labels]
    
    返回:
        dict: 包含每个标签的RMSE和平均RMSE的字典
    """
    # 确保输出和目标都是张量
    outputs = torch.cat(outputs, dim=0).numpy()
    targets = torch.stack(targets, dim=0).numpy()
    if is_Save:
        saveOutput(outputs, targets)
    
    acc_per_label = []
    for col in range(targets.shape[1]):
        acc = accuracy_score(targets[:, col], outputs[:, col])
        acc_per_label.append(acc)
    
    # # 计算子集准确率 - 所有输出都正确才算正确
    # subset_accuracy = np.mean(np.all(outputs == targets, axis=1))
    All = float(np.mean(acc_per_label))
    
    metrics = {
        'Acc_professionalism': acc_per_label[0],
        'Acc_occupational': acc_per_label[1],
        'Acc_effectiveness': acc_per_label[2],
        'Acc_quality': acc_per_label[3],
        'Acc_other': acc_per_label[4],
        'mean_Acc': All
    }
    
    return metrics

def calculate_metricsALL(outputs, targets):
    # 确保输出和目标都是张量
    outputs = torch.nn.functional.softmax(torch.cat(outputs, dim=0), dim=-1).detach().cpu().numpy()
    y_outputs = np.argmax(outputs, axis=2)
    logits = outputs
    targets = torch.stack(targets, dim=0).detach().cpu().numpy()
    
    print(logits.shape, y_outputs.shape, targets.shape)

    n_columns = targets.shape[1]
    columns = [str(i) for i in range(n_columns)]
    
    results = {}
    all_y_true = []
    all_y_pred = []
    all_logits = []
    
    for idx, col in enumerate(columns):
        y_true = targets[:, idx]
        y_pred = y_outputs[:, idx]
        logit = logits[:, idx, :]
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_logits.extend(logit)
        
        auc = roc_auc_score(y_true, logit, multi_class='ovr', average='weighted')
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro" ,zero_division=np.nan)
        re = recall_score(y_true, y_pred, average="macro" ,zero_division=np.nan)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=np.nan)
        
        results[col] = {
            'Prec': prec,
            'ACC': acc,
            'F1': f1,
            'RE': re,
            'AUC': auc
        }
        
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=np.nan)
    overall_re = recall_score(all_y_true, all_y_pred, average="macro", zero_division=np.nan)
    overall_prec = precision_score(all_y_true, all_y_pred, average="macro", zero_division=np.nan)  # 整体精确率
    overall_auc = roc_auc_score(all_y_true, all_logits, multi_class='ovr', average='weighted')
    
    results['Overall'] = {
        'Prec': overall_prec,
        'ACC': overall_acc,
        'F1': overall_f1,
        'RE': overall_re,
        'AUC': overall_auc
    }
    
    return pd.DataFrame.from_dict(results, orient='index')

if __name__ == "__main__":
    targets = [
        torch.tensor([0, 1, 2, 1, 1]), 
        torch.tensor([0, 1, 2, 1, 1]), 
        torch.tensor([0, 1, 2, 1, 1]), 
        torch.tensor([0, 1, 2, 1, 1]), 
        torch.tensor([0, 1, 2, 1, 1]), 
    ]
    outputs =[
        torch.randint(0, 3, size=(1,5)),
        torch.randint(0, 3, size=(2,5)),
        torch.randint(0, 3, size=(2,5)),
    ]
    a = calculate_metricsALL(outputs, targets)
    print(a)