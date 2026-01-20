#!/usr/bin/env python
"""
物理分支模型训练脚本
使用SVM或线性回归训练基于物理特征的模型
"""

import argparse
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Train physics branch model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data Excel file')
    parser.add_argument('--model_type', type=str, default='svm',
                       choices=['svm', 'linear'],
                       help='Type of model to train')
    parser.add_argument('--feature_cols', type=str, default='3:10',
                       help='Feature column indices (e.g., "3:10" for columns 3-9)')
    parser.add_argument('--target_col', type=int, default=5,
                       help='Target column index')
    parser.add_argument('--use_optuna', action='store_true',
                       help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--output_dir', type=str, default='physics_models',
                       help='Output directory for models and predictions')
    return parser.parse_args()

def load_data(data_path, feature_cols, target_col):
    """加载数据"""
    data = pd.read_excel(data_path)
    
    # 解析特征列
    if ':' in feature_cols:
        start, end = map(int, feature_cols.split(':'))
        features = data.iloc[:, start:end]
    else:
        # 假设是列名列表
        feature_names = feature_cols.split(',')
        features = data[feature_names]
    
    targets = data.iloc[:, target_col]
    
    return features.values, targets.values

def train_svm(X, y, use_optuna=True):
    """训练SVM模型"""
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    if use_optuna:
        # Optuna优化
        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-2, 1e2)
            epsilon = trial.suggest_loguniform('epsilon', 1e-2, 1e2)
            gamma = trial.suggest_loguniform('gamma', 1e-2, 1e2)
            
            svr = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            svr.fit(X_train, y_train)
            
            y_pred = svr.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        best_params = study.best_trial.params
        model = SVR(kernel='rbf', **best_params)
        print(f"Optuna最佳参数: {best_params}")
    else:
        model = SVR(kernel='rbf')
    
    # 在整个数据集上训练
    model.fit(X_scaled, y_scaled)
    
    return model, scaler_X, scaler_y

def train_linear_regression(X, y):
    """训练线性回归模型"""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    return model, scaler_X, scaler_y

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    X, y = load_data(args.data_path, args.feature_cols, args.target_col)
    print(f"Loaded data with shape: X={X.shape}, y={y.shape}")
    
    # 训练模型
    if args.model_type == 'svm':
        model, scaler_X, scaler_y = train_svm(X, y, args.use_optuna)
    elif args.model_type == 'linear':
        model, scaler_X, scaler_y = train_linear_regression(X, y)
    
    # 评估模型
    y_pred_scaled = model.predict(scaler_X.transform(X))
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    metrics = {
        'r2': r2_score(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred)
    }
    
    print(f"\nModel Performance:")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4e}")
    print(f"MAE: {metrics['mae']:.4e}")
    
    # 保存模型
    model_info = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'metrics': metrics
    }
    
    model_path = os.path.join(args.output_dir, f'physics_model_{args.model_type}.pkl')
    joblib.dump(model_info, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'true_value': y,
        'prediction': y_pred
    })
    
    pred_path = os.path.join(args.output_dir, f'physics_predictions_{args.model_type}.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")
    
    return model_info

if __name__ == "__main__":
    main()