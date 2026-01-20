"""
超参数优化模块
使用Optuna为所有模型优化超参数
支持：基础CNN、特征拼接、跨模态注意力、SVM融合
"""

import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# 导入自定义模块
from models.base_cnn import BaseConv3DNet
from models.feature_concat import FeatureConcatNet
from models.cross_attention import CrossAttentionNet
from models.decision_fusion import SVMFusion
from utils.data_loader import create_dataloader, set_seed, transform_label_to_perm

class HyperparameterOptimizer:
    """超参数优化器基类"""
    def __init__(self, model_type, device, data_config):
        self.model_type = model_type
        self.device = device
        self.data_config = data_config
        self.best_params = None
        self.best_value = None
        self.study = None
        
    def create_objective(self):
        """创建目标函数（需子类实现）"""
        raise NotImplementedError
    
    def optimize(self, n_trials=100, timeout=None, seed=42, n_jobs=1):
        """执行优化"""
        set_seed(seed)
        
        # 创建Optuna研究
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(),
            study_name=f'{self.model_type}_optimization'
        )
        
        # 优化
        print(f"\n开始优化 {self.model_type} 模型的超参数...")
        print(f"试验次数: {n_trials}")
        if timeout:
            print(f"超时时间: {timeout}秒")
        
        try:
            self.study.optimize(
                self.create_objective(),
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=True
            )
            
            # 保存最佳参数
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
            
            print(f"\n优化完成!")
            print(f"最佳验证损失: {self.best_value:.6f}")
            print(f"最佳超参数: {self.best_params}")
            
            return self.study
            
        except Exception as e:
            print(f"优化过程中出错: {e}")
            return None
    
    def get_best_params(self):
        """获取最佳参数"""
        return self.best_params
    
    def visualize_results(self, save_path=None):
        """可视化优化结果"""
        if self.study is None:
            print("警告: 没有优化结果可供可视化")
            return None, None, None
        
        try:
            import plotly
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice,
                plot_contour,
                plot_parallel_coordinate
            )
            
            figs = []
            
            # 优化历史
            fig1 = plot_optimization_history(self.study)
            figs.append(('optimization_history', fig1))
            
            # 参数重要性
            fig2 = plot_param_importances(self.study)
            figs.append(('param_importances', fig2))
            
            # 切片图
            fig3 = plot_slice(self.study)
            figs.append(('slice_plot', fig3))
            
            # 平行坐标图
            fig4 = plot_parallel_coordinate(self.study)
            figs.append(('parallel_coordinate', fig4))
            
            # 等高线图（如果有至少两个数值参数）
            try:
                fig5 = plot_contour(self.study)
                figs.append(('contour_plot', fig5))
            except:
                pass
            
            # 保存图表
            if save_path:
                for name, fig in figs:
                    try:
                        fig.write_html(f"{save_path}_{name}.html")
                    except Exception as e:
                        print(f"保存图表 {name} 时出错: {e}")
                
                print(f"可视化结果已保存到 {save_path}_*.html")
            
            return figs
            
        except ImportError:
            print("警告: 需要安装plotly以进行可视化 (pip install plotly optuna-dashboard)")
            return None
        except Exception as e:
            print(f"可视化过程中出错: {e}")
            return None


class BaseCNNOptimizer(HyperparameterOptimizer):
    """基础CNN模型优化器"""
    def create_objective(self):
        def objective(trial):
            # 定义超参数搜索空间
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            leaky_relu_slope = trial.suggest_float('leaky_relu_slope', 0.01, 0.2)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            
            # 隐藏层维度配置
            hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [256, 512, 1024, 2048])
            hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [128, 256, 512])
            hidden_dim_3 = trial.suggest_categorical('hidden_dim_3', [64, 128, 256])
            
            # 创建配置字典
            config = {
                'leaky_relu_slope': leaky_relu_slope,
                'hidden_dims': [hidden_dim_1, hidden_dim_2, hidden_dim_3, 128, 64, 32]
            }
            
            # 创建模型
            model = BaseConv3DNet(config=config)
            model = nn.DataParallel(model).to(self.device)
            
            # 创建数据加载器
            train_loader = create_dataloader(
                self.data_config['train_data'],
                self.data_config['train_excel'],
                batch_size=batch_size,
                shuffle=True,
                use_extra_features=False
            )
            
            val_loader = create_dataloader(
                self.data_config['val_data'],
                self.data_config['val_excel'],
                batch_size=batch_size // 2,
                shuffle=False,
                use_extra_features=False
            )
            
            # 设置优化器和损失函数
            criterion = nn.MSELoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum
            )
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            # 训练几轮来评估超参数
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 5
            
            for epoch in range(15):  # 训练15轮用于快速评估
                # 训练
                train_loss = 0.0
                for data, targets in train_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)
                
                # 验证
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, targets in val_loader:
                        data = data.to(self.device)
                        targets = targets.to(self.device)
                        outputs = model(data)
                        val_loss += criterion(outputs.squeeze(), targets).item()
                
                val_loss = val_loss / len(val_loader.dataset)
                scheduler.step(val_loss)
                
                # 更新最佳验证损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= max_patience:
                    break
                
                # Optuna剪枝
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
        
        return objective


class FeatureConcatOptimizer(HyperparameterOptimizer):
    """特征拼接模型优化器"""
    def create_objective(self):
        def objective(trial):
            # 定义超参数搜索空间
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            leaky_relu_slope = trial.suggest_float('leaky_relu_slope', 0.01, 0.2)
            
            # 隐藏层维度配置
            hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [512, 1024, 2048, 4096])
            hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [256, 512, 1024])
            hidden_dim_3 = trial.suggest_categorical('hidden_dim_3', [128, 256, 512])
            
            # 创建配置字典
            config = {
                'leaky_relu_slope': leaky_relu_slope,
                'hidden_dims': [hidden_dim_1, hidden_dim_2, hidden_dim_3, 256, 128, 64, 32]
            }
            
            # 创建模型
            model = FeatureConcatNet(config=config)
            model = nn.DataParallel(model).to(self.device)
            
            # 创建数据加载器
            train_loader = create_dataloader(
                self.data_config['train_data'],
                self.data_config['train_excel'],
                batch_size=batch_size,
                shuffle=True,
                use_extra_features=True
            )
            
            val_loader = create_dataloader(
                self.data_config['val_data'],
                self.data_config['val_excel'],
                batch_size=batch_size // 2,
                shuffle=False,
                use_extra_features=True
            )
            
            # 设置优化器和损失函数
            criterion = nn.MSELoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum
            )
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            # 训练几轮来评估超参数
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 5
            
            for epoch in range(15):  # 训练15轮用于快速评估
                # 训练
                train_loss = 0.0
                for data, targets, extra_feature in train_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    extra_feature = extra_feature.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(data, extra_feature)
                    loss = criterion(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)
                
                # 验证
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, targets, extra_feature in val_loader:
                        data = data.to(self.device)
                        targets = targets.to(self.device)
                        extra_feature = extra_feature.to(self.device)
                        outputs = model(data, extra_feature)
                        val_loss += criterion(outputs.squeeze(), targets).item()
                
                val_loss = val_loss / len(val_loader.dataset)
                scheduler.step(val_loss)
                
                # 更新最佳验证损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= max_patience:
                    break
                
                # Optuna剪枝
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
        
        return objective


class CrossAttentionOptimizer(HyperparameterOptimizer):
    """跨模态注意力模型优化器"""
    def create_objective(self):
        def objective(trial):
            # 定义超参数搜索空间
            lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            leaky_relu_slope = trial.suggest_float('leaky_relu_slope', 0.01, 0.2)
            attention_dim = trial.suggest_categorical('attention_dim', [2048, 4096, 8192, 16384])
            
            # 隐藏层维度配置
            hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [1024, 2048, 4096])
            hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [512, 1024, 2048])
            hidden_dim_3 = trial.suggest_categorical('hidden_dim_3', [256, 512, 1024])
            
            # 创建配置字典
            config = {
                'leaky_relu_slope': leaky_relu_slope,
                'attention_dim': attention_dim,
                'hidden_dims': [hidden_dim_1, hidden_dim_2, hidden_dim_3, 512, 256, 128, 64, 32]
            }
            
            # 创建模型
            model = CrossAttentionNet(config=config)
            model = nn.DataParallel(model).to(self.device)
            
            # 创建数据加载器
            train_loader = create_dataloader(
                self.data_config['train_data'],
                self.data_config['train_excel'],
                batch_size=batch_size,
                shuffle=True,
                use_extra_features=True
            )
            
            val_loader = create_dataloader(
                self.data_config['val_data'],
                self.data_config['val_excel'],
                batch_size=batch_size // 2,
                shuffle=False,
                use_extra_features=True
            )
            
            # 设置优化器和损失函数
            criterion = nn.MSELoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum
            )
            
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
            
            # 训练几轮来评估超参数
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0
            max_patience = 5
            
            for epoch in range(15):  # 训练15轮用于快速评估
                # 训练
                train_loss = 0.0
                for data, targets, extra_feature in train_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    extra_feature = extra_feature.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(data, extra_feature)
                    loss = criterion(outputs.squeeze(), targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * data.size(0)
                
                # 验证
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, targets, extra_feature in val_loader:
                        data = data.to(self.device)
                        targets = targets.to(self.device)
                        extra_feature = extra_feature.to(self.device)
                        outputs = model(data, extra_feature)
                        val_loss += criterion(outputs.squeeze(), targets).item()
                
                val_loss = val_loss / len(val_loader.dataset)
                scheduler.step(val_loss)
                
                # 更新最佳验证损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 早停
                if patience_counter >= max_patience:
                    break
                
                # Optuna剪枝
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_val_loss
        
        return objective


class SVMFusionOptimizer(HyperparameterOptimizer):
    """SVM融合模型优化器"""
    def create_objective(self):
        def objective(trial):
            # 定义超参数搜索空间
            C = trial.suggest_float('C', 1e-2, 1e2, log=True)
            epsilon = trial.suggest_float('epsilon', 1e-2, 1e2, log=True)
            gamma = trial.suggest_float('gamma', 1e-2, 1e2, log=True)
            
            # 尝试加载预测结果
            try:
                import pandas as pd
                
                image_pred_path = self.data_config.get('image_pred_path', 'predictions/image_predictions.csv')
                physics_pred_path = self.data_config.get('physics_pred_path', 'predictions/physics_predictions.csv')
                true_values_path = self.data_config.get('true_values_path', 'predictions/true_values.csv')
                
                # 检查文件是否存在
                if not (os.path.exists(image_pred_path) and os.path.exists(physics_pred_path) and os.path.exists(true_values_path)):
                    print(f"警告: 预测结果文件不存在，使用模拟数据")
                    return self._use_simulated_data(trial, C, epsilon, gamma)
                
                # 加载数据
                if image_pred_path.endswith('.csv'):
                    image_pred_df = pd.read_csv(image_pred_path)
                else:
                    image_pred_df = pd.read_excel(image_pred_path)
                
                if physics_pred_path.endswith('.csv'):
                    physics_pred_df = pd.read_csv(physics_pred_path)
                else:
                    physics_pred_df = pd.read_excel(physics_pred_path)
                
                if true_values_path.endswith('.csv'):
                    true_df = pd.read_csv(true_values_path)
                else:
                    true_df = pd.read_excel(true_values_path)
                
                # 提取预测值和真实值
                image_pred = image_pred_df['prediction'].values
                physics_pred = physics_pred_df['prediction'].values
                true_values = true_df['true_value'].values
                
                # 确保数据长度一致
                min_len = min(len(image_pred), len(physics_pred), len(true_values))
                image_pred = image_pred[:min_len]
                physics_pred = physics_pred[:min_len]
                true_values = true_values[:min_len]
                
                # 划分训练集和验证集
                X = np.column_stack([image_pred, physics_pred])
                y = true_values
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # 创建并训练SVM模型
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                X_train_scaled = scaler_X.fit_transform(X_train)
                y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
                X_val_scaled = scaler_X.transform(X_val)
                
                svr = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
                svr.fit(X_train_scaled, y_train_scaled)
                
                # 预测并计算损失
                y_pred_scaled = svr.predict(X_val_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                mse = mean_squared_error(y_val, y_pred)
                return mse
                
            except Exception as e:
                print(f"加载数据时出错: {e}，使用模拟数据")
                return self._use_simulated_data(trial, C, epsilon, gamma)
        
        return objective
    
    def _use_simulated_data(self, trial, C, epsilon, gamma):
        """使用模拟数据"""
        # 生成模拟数据
        n_samples = 200
        np.random.seed(42)
        
        # 生成真实值
        true_values = np.random.uniform(1e-11, 1e-9, n_samples)
        
        # 生成图像分支预测（有一定误差）
        image_pred = true_values * (1 + np.random.normal(0, 0.2, n_samples))
        
        # 生成物理分支预测（有一定误差）
        physics_pred = true_values * (1 + np.random.normal(0, 0.25, n_samples))
        
        # 划分训练集和验证集
        X = np.column_stack([image_pred, physics_pred])
        y = true_values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建并训练SVM模型
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_val_scaled = scaler_X.transform(X_val)
        
        svr = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(X_train_scaled, y_train_scaled)
        
        # 预测并计算损失
        y_pred_scaled = svr.predict(X_val_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        mse = mean_squared_error(y_val, y_pred)
        return mse


def optimize_all_models(data_config, output_dir='optimization_results', 
                        n_trials=50, timeout=None, seed=42, n_jobs=1):
    """
    优化所有模型的超参数
    
    Args:
        data_config: 数据配置字典
        output_dir: 输出目录
        n_trials: 每个模型的试验次数
        timeout: 超时时间（秒）
        seed: 随机种子
        n_jobs: 并行作业数
    
    Returns:
        所有模型的最佳参数字典
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义要优化的模型
    optimizers = [
        ('base_cnn', BaseCNNOptimizer('base_cnn', device, data_config)),
        ('feature_concat', FeatureConcatOptimizer('feature_concat', device, data_config)),
        ('cross_attention', CrossAttentionOptimizer('cross_attention', device, data_config)),
        ('svm_fusion', SVMFusionOptimizer('svm_fusion', device, data_config))
    ]
    
    best_params_all = {}
    
    for model_name, optimizer in optimizers:
        print(f"\n{'='*60}")
        print(f"优化 {model_name} 模型")
        print(f"{'='*60}")
        
        try:
            # 运行优化
            study = optimizer.optimize(
                n_trials=n_trials, 
                timeout=timeout, 
                seed=seed,
                n_jobs=n_jobs
            )
            
            if study:
                best_params = optimizer.get_best_params()
                best_params_all[model_name] = best_params
                
                # 保存结果
                params_path = os.path.join(output_dir, f'{model_name}_best_params.json')
                with open(params_path, 'w') as f:
                    json.dump(best_params, f, indent=2)
                
                print(f"最佳参数已保存到 {params_path}")
                
                # 可视化
                try:
                    optimizer.visualize_results(
                        os.path.join(output_dir, f'{model_name}_optimization')
                    )
                except Exception as e:
                    print(f"可视化时出错: {e}")
                    
        except Exception as e:
            print(f"优化 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存所有最佳参数
    all_params_path = os.path.join(output_dir, 'all_best_params.json')
    with open(all_params_path, 'w') as f:
        json.dump(best_params_all, f, indent=2)
    
    print(f"\n所有最佳参数已保存到 {all_params_path}")
    
    # 生成优化报告
    generate_optimization_report(best_params_all, output_dir, data_config)
    
    return best_params_all


def load_best_params(model_name, params_dir='optimization_results'):
    """加载已保存的最佳参数"""
    import os
    import json
    
    params_path = os.path.join(params_dir, f'{model_name}_best_params.json')
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"从 {params_path} 加载了 {model_name} 的最佳参数")
        return params
    else:
        print(f"警告: 未找到 {model_name} 的最佳参数文件")
        return None


def generate_optimization_report(best_params, output_dir, data_config):
    """生成优化报告"""
    import json
    from datetime import datetime
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_config': data_config,
        'best_parameters': best_params,
        'recommendations': {}
    }
    
    # 为每个模型生成建议
    for model_name, params in best_params.items():
        if model_name == 'base_cnn':
            report['recommendations'][model_name] = {
                'model_class': 'BaseConv3DNet',
                'config': {
                    'leaky_relu_slope': params.get('leaky_relu_slope', 0.01),
                    'hidden_dims': [
                        params.get('hidden_dim_1', 1024),
                        params.get('hidden_dim_2', 512),
                        params.get('hidden_dim_3', 256),
                        128, 64, 32
                    ]
                },
                'training': {
                    'lr': params.get('lr', 0.005),
                    'weight_decay': params.get('weight_decay', 0.0001),
                    'momentum': params.get('momentum', 0.9),
                    'batch_size': params.get('batch_size', 32)
                }
            }
        elif model_name == 'feature_concat':
            report['recommendations'][model_name] = {
                'model_class': 'FeatureConcatNet',
                'config': {
                    'leaky_relu_slope': params.get('leaky_relu_slope', 0.1),
                    'hidden_dims': [
                        params.get('hidden_dim_1', 2048),
                        params.get('hidden_dim_2', 1024),
                        params.get('hidden_dim_3', 512),
                        256, 128, 64, 32
                    ]
                },
                'training': {
                    'lr': params.get('lr', 0.005),
                    'weight_decay': params.get('weight_decay', 0.001),
                    'momentum': params.get('momentum', 0.9),
                    'batch_size': params.get('batch_size', 32)
                }
            }
        elif model_name == 'cross_attention':
            report['recommendations'][model_name] = {
                'model_class': 'CrossAttentionNet',
                'config': {
                    'leaky_relu_slope': params.get('leaky_relu_slope', 0.1),
                    'attention_dim': params.get('attention_dim', 8192),
                    'hidden_dims': [
                        params.get('hidden_dim_1', 4096),
                        params.get('hidden_dim_2', 2048),
                        params.get('hidden_dim_3', 1024),
                        512, 256, 128, 64, 32
                    ]
                },
                'training': {
                    'lr': params.get('lr', 0.01),
                    'weight_decay': params.get('weight_decay', 0.001),
                    'momentum': params.get('momentum', 0.9),
                    'batch_size': params.get('batch_size', 32)
                }
            }
        elif model_name == 'svm_fusion':
            report['recommendations'][model_name] = {
                'model_class': 'SVMFusion',
                'config': params,
                'usage': '用于决策级融合中的SVM融合方法'
            }
    
    # 保存报告
    report_path = os.path.join(output_dir, 'optimization_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"优化报告已保存到 {report_path}")
    
    # 生成Markdown格式的报告
    md_report_path = os.path.join(output_dir, 'optimization_report.md')
    with open(md_report_path, 'w') as f:
        f.write("# 超参数优化报告\n\n")
        f.write(f"生成时间: {report['generated_at']}\n\n")
        
        f.write("## 数据配置\n\n")
        for key, value in data_config.items():
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")
        
        f.write("## 最佳参数\n\n")
        for model_name, params in best_params.items():
            f.write(f"### {model_name}\n\n")
            f.write("```json\n")
            f.write(json.dumps(params, indent=2))
            f.write("\n```\n\n")
        
        f.write("## 使用建议\n\n")
        f.write("### 1. 基础CNN模型\n")
        f.write("```python\n")
        f.write("from models.base_cnn import BaseConv3DNet\n")
        f.write("config = {\n")
        config = report['recommendations']['base_cnn']['config']
        for key, value in config.items():
            if isinstance(value, list):
                f.write(f"    '{key}': {value},\n")
            else:
                f.write(f"    '{key}': {value},\n")
        f.write("}\n")
        f.write("model = BaseConv3DNet(config=config)\n")
        f.write("```\n\n")
        
        f.write("### 2. 训练命令\n")
        f.write("```bash\n")
        f.write("# 使用优化后的参数训练\n")
        f.write("python train.py --strategy base --use_best_params --params_dir optimization_results\n\n")
        
        f.write("# 手动指定参数训练\n")
        training = report['recommendations']['base_cnn']['training']
        f.write(f"python train.py --strategy base --lr {training['lr']} \\\n")
        f.write(f"  --weight_decay {training['weight_decay']} --momentum {training['momentum']} \\\n")
        f.write(f"  --batch_size {training['batch_size']}\n")
        f.write("```\n\n")
        
        f.write("### 3. 可视化结果\n")
        f.write("优化结果可视化图表已生成，可以使用以下方式查看：\n")
        f.write("1. 打开HTML文件: `optimization_results/*.html`\n")
        f.write("2. 使用Optuna仪表板: `optuna-dashboard sqlite:///optimization_results/studies.db`\n")
    
    print(f"Markdown报告已保存到 {md_report_path}")
    
    return report


if __name__ == "__main__":
    # 示例用法
    data_config = {
        'train_data': 'data/4.2T',
        'train_excel': 'data/4.2T.xlsx',
        'val_data': 'data/4.2V',
        'val_excel': 'data/4.2V.xlsx',
        'image_pred_path': 'predictions/image_predictions.csv',
        'physics_pred_path': 'predictions/physics_predictions.csv',
        'true_values_path': 'predictions/true_values.csv'
    }
    
    # 优化所有模型
    best_params = optimize_all_models(
        data_config=data_config,
        output_dir='optimization_results',
        n_trials=30,  # 每个模型30次试验
        timeout=3600,  # 1小时超时
        seed=42,
        n_jobs=1
    )