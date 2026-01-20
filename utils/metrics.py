import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from .data_loader import transform_label_to_perm

def evaluate_model(model, loader, device, model_type='base'):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        loader: 数据加载器
        device: 设备 (cuda/cpu)
        model_type: 模型类型 ('base', 'concat', 'attention')
    
    Returns:
        包含评估指标的字典
    """
    model.eval()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    predictions = []
    true_values = []
    percent_errors = []
    max_overestimation_error = float('-inf')
    min_underestimation_error = float('inf')
    
    with torch.no_grad():
        index = 0
        for batch in loader:
            if model_type == 'base':
                data, targets = batch
                data = data.to(device)
                outputs = model(data).cpu().numpy().squeeze()
            else:
                data, targets, extra_feature = batch
                data = data.to(device)
                extra_feature = extra_feature.to(device)
                outputs = model(data, extra_feature).cpu().numpy().squeeze()
            
            preds = transform_label_to_perm(outputs)
            trues = transform_label_to_perm(targets.numpy())
            
            # 确保 preds 和 trues 是数组形式
            preds = np.atleast_1d(preds)
            trues = np.atleast_1d(trues)
            
            eps = 1e-15
            batch_errors = [(p - t) / (t + eps) * 100 for p, t in zip(preds, trues)]
            
            for i, error in enumerate(batch_errors):
                if error > max_overestimation_error:
                    max_overestimation_error = error
                    worst_overestimation_index = index + i
                if error < min_underestimation_error:
                    min_underestimation_error = error
                    worst_underestimation_index = index + i

            predictions.extend(preds)
            true_values.extend(trues)
            percent_errors.extend(batch_errors)
            index += len(batch_errors)
    
    critical_samples = {
        'worst_overestimation': {
            'True_Value': true_values[worst_overestimation_index],
            'Predicted_Value': predictions[worst_overestimation_index],
            'Percentage_Error': percent_errors[worst_overestimation_index]
        },
        'worst_underestimation': {
            'True_Value': true_values[worst_underestimation_index],
            'Predicted_Value': predictions[worst_underestimation_index],
            'Percentage_Error': percent_errors[worst_underestimation_index]
        }
    }

    return {
        'r2': r2_score(true_values, predictions),
        'mse': mean_squared_error(true_values, predictions),
        'mae': mean_absolute_error(true_values, predictions),
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mape': np.mean(np.abs(percent_errors)),
        'mdape': np.median(np.abs(percent_errors)),
        'true': true_values,
        'pred': predictions,
        'percent_errors': percent_errors,
        'critical_samples': critical_samples
    }

def save_error_analysis(metrics, save_path):
    """保存详细误差分析报告"""
    df = pd.DataFrame({
        'True_Value': metrics['true'],
        'Predicted_Value': metrics['pred'],
        'Absolute_Error': np.abs(np.array(metrics['pred']) - np.array(metrics['true'])),
        'Percentage_Error': metrics['percent_errors'],
        'Absolute_Percentage_Error': np.abs(metrics['percent_errors'])
    })
    
    stats = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std', 'Min', 'Max'],
        'Absolute_Error': [
            df['Absolute_Error'].mean(),
            df['Absolute_Error'].median(),
            df['Absolute_Error'].std(),
            df['Absolute_Error'].min(),
            df['Absolute_Error'].max()
        ],
        'Percentage_Error': [
            df['Percentage_Error'].mean(),
            df['Percentage_Error'].median(),
            df['Percentage_Error'].std(),
            df['Percentage_Error'].min(),
            df['Percentage_Error'].max()
        ]
    })
    
    with pd.ExcelWriter(save_path) as writer:
        df.to_excel(writer, sheet_name='Detailed Errors', index=False)
        stats.to_excel(writer, sheet_name='Statistics', index=False)