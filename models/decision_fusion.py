"""
决策级融合模型
包含：平均融合、熵权法融合、最小二乘法融合、SVM融合
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle

# 机器学习库
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import optuna

class DecisionFusion:
    """决策级融合基类"""
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.is_fitted = False
        self.training_history = []
    
    def fit(self, image_predictions: np.ndarray, physics_predictions: np.ndarray, 
            true_values: np.ndarray) -> None:
        """
        训练融合模型
        
        Args:
            image_predictions: 图像分支的预测值
            physics_predictions: 物理分支的预测值
            true_values: 真实值
        """
        raise NotImplementedError
    
    def predict(self, image_predictions: np.ndarray, physics_predictions: np.ndarray) -> np.ndarray:
        """
        使用融合模型进行预测
        
        Args:
            image_predictions: 图像分支的预测值
            physics_predictions: 物理分支的预测值
        
        Returns:
            融合后的预测值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练。请先调用 fit() 方法。")
        raise NotImplementedError
    
    def evaluate(self, image_predictions: np.ndarray, physics_predictions: np.ndarray,
                 true_values: np.ndarray) -> Dict[str, float]:
        """
        评估融合模型
        
        Returns:
            包含评估指标的字典
        """
        predictions = self.predict(image_predictions, physics_predictions)
        
        # 计算各种评估指标
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        # 计算百分比误差
        eps = 1e-15
        percent_errors = [(p - t) / (t + eps) * 100 for p, t in zip(predictions, true_values)]
        mape = np.mean(np.abs(percent_errors))
        mdape = np.median(np.abs(percent_errors))
        
        return {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'mdape': mdape,
            'predictions': predictions,
            'true_values': true_values,
            'percent_errors': percent_errors
        }
    
    def save(self, filepath: str) -> None:
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'DecisionFusion':
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_config(self) -> Dict:
        """获取模型配置"""
        return {
            'name': self.name,
            'config': self.config,
            'is_fitted': self.is_fitted
        }


class AverageFusion(DecisionFusion):
    """平均融合法"""
    def __init__(self, weights: Optional[List[float]] = None, config: Optional[Dict] = None):
        if config is None:
            config = {}
        
        if weights is None:
            weights = config.get('weights', [0.5, 0.5])
        
        super().__init__("AverageFusion", config)
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)  # 归一化
    
    def fit(self, image_predictions: np.ndarray, physics_predictions: np.ndarray, 
            true_values: np.ndarray) -> None:
        """
        平均法不需要训练，直接使用
        可选：根据验证集性能优化权重
        """
        # 如果需要优化权重
        if self.config.get('optimize_weights', False):
            self._optimize_weights(image_predictions, physics_predictions, true_values)
        
        self.is_fitted = True
    
    def _optimize_weights(self, image_predictions: np.ndarray, physics_predictions: np.ndarray,
                          true_values: np.ndarray) -> None:
        """优化权重"""
        # 使用简单网格搜索找到最佳权重
        best_score = -float('inf')
        best_weights = self.weights
        
        # 尝试不同的权重组合
        for w1 in np.linspace(0, 1, 11):
            w2 = 1 - w1
            weights = np.array([w1, w2])
            
            # 计算加权平均
            predictions = weights[0] * image_predictions + weights[1] * physics_predictions
            
            # 计算R²分数
            score = r2_score(true_values, predictions)
            
            if score > best_score:
                best_score = score
                best_weights = weights
        
        self.weights = best_weights
        print(f"优化后的权重: {self.weights}, R²: {best_score:.4f}")
    
    def predict(self, image_predictions: np.ndarray, physics_predictions: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            self.is_fitted = True  # 平均法总是可以预测
        
        return (self.weights[0] * image_predictions + 
                self.weights[1] * physics_predictions)


class EntropyWeightFusion(DecisionFusion):
    """熵权法融合"""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("EntropyWeightFusion", config)
        self.weights = None
    
    def _entropy_weight(self, data: np.ndarray) -> np.ndarray:
        """计算熵权"""
        # 数据标准化（归一化）
        data_normalized = data / np.sum(data, axis=0, keepdims=True)
        
        # 防止log(0)
        data_normalized = np.clip(data_normalized, 1e-10, 1)
        
        # 计算熵值
        k = 1 / np.log(data.shape[0])
        entropy = -k * np.sum(data_normalized * np.log(data_normalized), axis=0)
        
        # 计算权重
        weights = (1 - entropy) / np.sum(1 - entropy)
        
        return weights
    
    def fit(self, image_predictions: np.ndarray, physics_predictions: np.ndarray, 
            true_values: np.ndarray) -> None:
        """
        使用熵权法计算权重
        基于每个预测器与真实值的接近程度
        """
        # 计算每个预测器的绝对误差
        image_errors = np.abs(image_predictions - true_values)
        physics_errors = np.abs(physics_predictions - true_values)
        
        # 构建评估矩阵（列：图像分支误差，物理分支误差）
        # 注意：误差越小越好，所以我们需要将其转换为正向指标
        evaluation_matrix = np.column_stack([image_errors, physics_errors])
        
        # 方法1：使用误差的倒数（误差越小，倒数越大，性能越好）
        evaluation_matrix = 1 / (evaluation_matrix + 1e-10)
        
        # 方法2：或者使用负误差（误差越小，负值越大）
        # evaluation_matrix = -evaluation_matrix
        
        # 计算熵权
        self.weights = self._entropy_weight(evaluation_matrix)
        
        # 记录训练历史
        self.training_history.append({
            'weights': self.weights.copy(),
            'image_mean_error': np.mean(image_errors),
            'physics_mean_error': np.mean(physics_errors)
        })
        
        self.is_fitted = True
        
        print(f"熵权法权重计算完成:")
        print(f"  图像分支权重: {self.weights[0]:.4f}")
        print(f"  物理分支权重: {self.weights[1]:.4f}")
    
    def predict(self, image_predictions: np.ndarray, physics_predictions: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练。请先调用 fit() 方法。")
        
        return (self.weights[0] * image_predictions + 
                self.weights[1] * physics_predictions)


class LinearRegressionFusion(DecisionFusion):
    """最小二乘法融合"""
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("LinearRegressionFusion", config)
        self.model = LinearRegression()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.coefficients = None
        self.intercept = None
    
    def fit(self, image_predictions: np.ndarray, physics_predictions: np.ndarray, 
            true_values: np.ndarray) -> None:
        """
        训练最小二乘融合模型
        y = w1 * x1 + w2 * x2 + b
        """
        # 准备特征
        X = np.column_stack([image_predictions, physics_predictions])
        y = true_values
        
        # 可选：标准化
        if self.config.get('standardize', True):
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            # 训练模型
            self.model.fit(X_scaled, y_scaled)
            
            # 保存系数（标准化后的）
            self.coefficients = self.model.coef_
            self.intercept = self.model.intercept_
        else:
            # 不标准化
            self.model.fit(X, y)
            self.coefficients = self.model.coef_
            self.intercept = self.model.intercept_
        
        # 记录训练历史
        self.training_history.append({
            'coefficients': self.coefficients.copy(),
            'intercept': self.intercept,
            'r2_score': self.model.score(X, y) if not self.config.get('standardize', True) else 
                       self.model.score(self.scaler_X.transform(X), 
                                       self.scaler_y.transform(y.reshape(-1, 1)).ravel())
        })
        
        self.is_fitted = True
        
        print(f"线性回归融合模型训练完成:")
        print(f"  系数: {self.coefficients}")
        print(f"  截距: {self.intercept}")
        print(f"  公式: y = {self.coefficients[0]:.4f} * x_image + {self.coefficients[1]:.4f} * x_physics + {self.intercept:.4f}")
    
    def predict(self, image_predictions: np.ndarray, physics_predictions: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练。请先调用 fit() 方法。")
        
        X = np.column_stack([image_predictions, physics_predictions])
        
        if self.config.get('standardize', True):
            X_scaled = self.scaler_X.transform(X)
            y_pred_scaled = self.model.predict(X_scaled)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = self.model.predict(X)
        
        return y_pred


class SVMFusion(DecisionFusion):
    """SVM融合（支持Optuna超参数优化）"""
    def __init__(self, config: Optional[Dict] = None, use_optuna: bool = True):
        super().__init__("SVMFusion", config)
        self.use_optuna = use_optuna
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.best_params = None
        self.study = None
        
        # 默认配置
        self.default_config = {
            'C_low': 1e-2,
            'C_high': 1e2,
            'epsilon_low': 1e-2,
            'epsilon_high': 1e2,
            'gamma_low': 1e-2,
            'gamma_high': 1e2,
            'kernel': 'rbf',
            'n_trials': 50,
            'test_size': 0.2
        }
        
        # 更新配置
        if config:
            self.default_config.update(config)
    
    def _optuna_objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna目标函数"""
        # 从配置中获取搜索范围
        C = trial.suggest_float('C', 
                               self.default_config['C_low'], 
                               self.default_config['C_high'], 
                               log=True)
        epsilon = trial.suggest_float('epsilon', 
                                     self.default_config['epsilon_low'], 
                                     self.default_config['epsilon_high'], 
                                     log=True)
        gamma = trial.suggest_float('gamma', 
                                   self.default_config['gamma_low'], 
                                   self.default_config['gamma_high'], 
                                   log=True)
        
        svr = SVR(kernel=self.default_config['kernel'], 
                  C=C, epsilon=epsilon, gamma=gamma)
        svr.fit(X_train, y_train)
        
        y_pred = svr.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    def fit(self, image_predictions: np.ndarray, physics_predictions: np.ndarray, 
            true_values: np.ndarray) -> None:
        """
        训练SVM融合模型（可选择使用Optuna优化）
        """
        # 准备特征
        X = np.column_stack([image_predictions, physics_predictions])
        y = true_values
        
        # 标准化
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, 
            test_size=self.default_config['test_size'], 
            random_state=42
        )
        
        if self.use_optuna:
            print(f"使用Optuna优化SVM超参数，试验次数: {self.default_config['n_trials']}")
            
            # 创建Optuna研究
            self.study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            # 优化
            self.study.optimize(
                lambda trial: self._optuna_objective(trial, X_train, y_train, X_val, y_val),
                n_trials=self.default_config['n_trials'],
                show_progress_bar=True
            )
            
            self.best_params = self.study.best_params
            self.model = SVR(kernel=self.default_config['kernel'], **self.best_params)
            
            print(f"Optuna优化完成，最佳参数: {self.best_params}")
            print(f"最佳验证MSE: {self.study.best_value:.6f}")
        else:
            # 使用默认参数
            default_params = {
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale'
            }
            self.model = SVR(kernel=self.default_config['kernel'], **default_params)
            self.best_params = default_params
        
        # 在整个训练集上重新训练
        self.model.fit(X_scaled, y_scaled)
        
        # 记录训练历史
        self.training_history.append({
            'best_params': self.best_params,
            'scaler_X_mean': self.scaler_X.mean_,
            'scaler_X_scale': self.scaler_X.scale_,
            'scaler_y_mean': self.scaler_y.mean_,
            'scaler_y_scale': self.scaler_y.scale_
        })
        
        self.is_fitted = True
    
    def predict(self, image_predictions: np.ndarray, physics_predictions: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练。请先调用 fit() 方法。")
        
        X = np.column_stack([image_predictions, physics_predictions])
        X_scaled = self.scaler_X.transform(X)
        
        # 预测并反标准化
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def visualize_optimization(self, save_path: Optional[str] = None):
        """可视化Optuna优化结果"""
        if not self.study:
            print("警告: 没有Optuna优化结果可供可视化")
            return
        
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_slice
            )
            
            # 优化历史
            fig1 = plot_optimization_history(self.study)
            
            # 参数重要性
            fig2 = plot_param_importances(self.study)
            
            # 切片图
            fig3 = plot_slice(self.study)
            
            if save_path:
                fig1.write_html(f"{save_path}_optimization_history.html")
                fig2.write_html(f"{save_path}_param_importances.html")
                fig3.write_html(f"{save_path}_slice_plot.html")
                print(f"可视化图表已保存到 {save_path}_*.html")
            
            return fig1, fig2, fig3
            
        except ImportError:
            print("需要安装plotly以进行可视化: pip install plotly")
            return None


class DecisionLevelFusionManager:
    """决策级融合管理器"""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fusion_methods = {}
        self.image_predictions = None
        self.physics_predictions = None
        self.true_values = None
        self.results = {}
        
        # 初始化融合方法
        self._initialize_fusion_methods()
    
    def _initialize_fusion_methods(self):
        """初始化所有融合方法"""
        # 从配置中获取方法配置
        method_configs = self.config.get('methods', {})
        
        self.fusion_methods = {
            'average': AverageFusion(config=method_configs.get('average', {})),
            'entropy': EntropyWeightFusion(config=method_configs.get('entropy', {})),
            'linear': LinearRegressionFusion(config=method_configs.get('linear', {})),
            'svm': SVMFusion(config=method_configs.get('svm', {}), use_optuna=True)
        }
    
    def load_predictions(self, image_pred_path: str, physics_pred_path: str,
                         true_values_path: Optional[str] = None):
        """
        加载图像和物理分支的预测结果
        
        Args:
            image_pred_path: 图像分支预测结果文件路径
            physics_pred_path: 物理分支预测结果文件路径
            true_values_path: 真实值文件路径（可选）
        """
        # 加载预测结果
        if image_pred_path.endswith('.csv'):
            image_pred_df = pd.read_csv(image_pred_path)
        else:
            image_pred_df = pd.read_excel(image_pred_path)
        
        if physics_pred_path.endswith('.csv'):
            physics_pred_df = pd.read_csv(physics_pred_path)
        else:
            physics_pred_df = pd.read_excel(physics_pred_path)
        
        # 提取预测值
        # 尝试常见的列名
        def extract_predictions(df):
            possible_names = ['prediction', 'pred', 'predicted', 'predicted_value', 'value']
            for name in possible_names:
                if name in df.columns:
                    return df[name].values
            
            # 如果没有找到，使用第一列数值数据
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    return df[col].values
            
            # 最后使用第一列
            return df.iloc[:, 0].values
        
        self.image_predictions = extract_predictions(image_pred_df)
        self.physics_predictions = extract_predictions(physics_pred_df)
        
        # 加载真实值（如果提供）
        if true_values_path:
            if true_values_path.endswith('.csv'):
                true_df = pd.read_csv(true_values_path)
            else:
                true_df = pd.read_excel(true_values_path)
            
            # 提取真实值
            self.true_values = extract_predictions(true_df)
        else:
            self.true_values = None
        
        print(f"已加载 {len(self.image_predictions)} 个图像预测和 {len(self.physics_predictions)} 个物理预测")
        
        # 检查数据长度
        if len(self.image_predictions) != len(self.physics_predictions):
            print(f"警告: 图像预测 ({len(self.image_predictions)}) 和物理预测 ({len(self.physics_predictions)}) 长度不一致")
    
    def train_all_fusion_methods(self, train_ratio: float = 0.8, 
                                random_state: int = 42) -> Dict[str, Any]:
        """
        训练所有融合方法
        
        Args:
            train_ratio: 训练集比例
            random_state: 随机种子
        
        Returns:
            包含所有方法结果的字典
        """
        if self.image_predictions is None or self.physics_predictions is None or self.true_values is None:
            raise ValueError("请先加载预测结果和真实值")
        
        # 确保数据长度一致
        min_len = min(len(self.image_predictions), len(self.physics_predictions), len(self.true_values))
        image_pred = self.image_predictions[:min_len]
        physics_pred = self.physics_predictions[:min_len]
        true_vals = self.true_values[:min_len]
        
        print(f"使用 {min_len} 个样本进行训练和测试")
        
        # 划分训练集和测试集
        n_samples = min_len
        n_train = int(n_samples * train_ratio)
        
        indices = np.random.RandomState(random_state).permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # 准备训练数据
        image_train = image_pred[train_idx]
        physics_train = physics_pred[train_idx]
        true_train = true_vals[train_idx]
        
        # 准备测试数据
        image_test = image_pred[test_idx]
        physics_test = physics_pred[test_idx]
        true_test = true_vals[test_idx]
        
        self.results = {}
        
        for name, method in self.fusion_methods.items():
            print(f"\n训练 {name} 融合方法...")
            
            try:
                # 训练
                method.fit(image_train, physics_train, true_train)
                
                # 在训练集上评估
                train_metrics = method.evaluate(image_train, physics_train, true_train)
                
                # 在测试集上评估
                test_metrics = method.evaluate(image_test, physics_test, true_test)
                
                self.results[name] = {
                    'method': method,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'train_predictions': train_metrics['predictions'],
                    'test_predictions': test_metrics['predictions']
                }
                
                print(f"  {name} 训练集 R²: {train_metrics['r2']:.4f}, 测试集 R²: {test_metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"  训练 {name} 时出错: {e}")
                continue
        
        return self.results
    
    def get_best_method(self, metric: str = 'r2', dataset: str = 'test') -> Tuple[str, Any, float]:
        """获取最佳融合方法"""
        if not self.results:
            raise ValueError("请先训练融合方法")
        
        best_name = None
        best_score = -float('inf')
        best_method = None
        
        for name, result in self.results.items():
            score = result[f'{dataset}_metrics'][metric]
            if score > best_score:
                best_score = score
                best_name = name
                best_method = result['method']
        
        return best_name, best_method, best_score
    
    def compare_methods(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """比较所有融合方法"""
        if not self.results:
            raise ValueError("请先训练融合方法")
        
        comparison_data = []
        
        for name, result in self.results.items():
            train_metrics = result['train_metrics']
            test_metrics = result['test_metrics']
            
            comparison_data.append({
                'Method': name,
                'Train_R2': f"{train_metrics['r2']:.4f}",
                'Test_R2': f"{test_metrics['r2']:.4f}",
                'Test_RMSE': f"{test_metrics['rmse']:.2e}",
                'Test_MAE': f"{test_metrics['mae']:.2e}",
                'Test_MAPE': f"{test_metrics['mape']:.2f}%"
            })
        
        df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        if save_path:
            if save_path.endswith('.csv'):
                df.to_csv(save_path, index=False)
            else:
                df.to_excel(save_path, index=False)
            print(f"方法比较结果已保存到 {save_path}")
        
        return df
    
    def save_predictions(self, method_name: str, method: DecisionFusion, 
                         save_path: str, dataset_type: str = 'all'):
        """
        保存融合预测结果
        
        Args:
            method_name: 融合方法名称
            method: 融合方法实例
            save_path: 保存路径
            dataset_type: 数据集类型 ('train', 'test', 'all')
        """
        if dataset_type == 'all':
            predictions = method.predict(self.image_predictions, self.physics_predictions)
            data = pd.DataFrame({
                'image_prediction': self.image_predictions[:len(predictions)],
                'physics_prediction': self.physics_predictions[:len(predictions)],
                'fused_prediction': predictions,
                'true_value': self.true_values[:len(predictions)] if self.true_values is not None else np.nan
            })
        else:
            # 这里需要根据实际划分来获取相应数据
            # 简化处理：使用所有数据
            predictions = method.predict(self.image_predictions, self.physics_predictions)
            data = pd.DataFrame({
                'image_prediction': self.image_predictions[:len(predictions)],
                'physics_prediction': self.physics_predictions[:len(predictions)],
                'fused_prediction': predictions,
                'true_value': self.true_values[:len(predictions)] if self.true_values is not None else np.nan
            })
        
        if save_path.endswith('.csv'):
            data.to_csv(save_path, index=False)
        else:
            data.to_excel(save_path, index=False)
        
        print(f"预测结果已保存到: {save_path}")
    
    def save_all_results(self, output_dir: str):
        """保存所有结果"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存方法比较
        comparison_path = os.path.join(output_dir, 'fusion_methods_comparison.csv')
        self.compare_methods(comparison_path)
        
        # 保存最佳方法
        best_name, best_method, best_score = self.get_best_method()
        best_method_path = os.path.join(output_dir, f'best_fusion_{best_name}.pkl')
        best_method.save(best_method_path)
        
        # 保存结果摘要
        summary = {
            'best_method': best_name,
            'best_score': best_score,
            'num_samples': len(self.image_predictions) if self.image_predictions is not None else 0,
            'methods_trained': list(self.results.keys())
        }
        
        summary_path = os.path.join(output_dir, 'fusion_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"所有结果已保存到 {output_dir}")
        
        return best_method_path


def demo_decision_fusion():
    """决策级融合演示函数"""
    # 创建管理器
    manager = DecisionLevelFusionManager()
    
    # 加载数据（示例路径）
    image_pred_path = "predictions/image_predictions.csv"
    physics_pred_path = "predictions/physics_predictions.csv"
    true_values_path = "predictions/true_values.csv"
    
    try:
        manager.load_predictions(image_pred_path, physics_pred_path, true_values_path)
        
        # 训练所有融合方法
        results = manager.train_all_fusion_methods(train_ratio=0.8)
        
        # 获取最佳方法
        best_name, best_method, best_score = manager.get_best_method()
        print(f"\n最佳融合方法: {best_name}, 测试集R²: {best_score:.4f}")
        
        # 比较所有方法
        comparison_df = manager.compare_methods()
        print("\n融合方法比较:")
        print(comparison_df.to_string(index=False))
        
        # 使用最佳方法保存预测结果
        save_path = f"predictions/best_fusion_{best_name}.csv"
        manager.save_predictions(best_name, best_method, save_path)
        
        # 保存所有结果
        manager.save_all_results("fusion_results")
        
        return results, best_method
        
    except Exception as e:
        print(f"演示过程中出错: {e}")
        return None, None


if __name__ == "__main__":
    results, best_method = demo_decision_fusion()