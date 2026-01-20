#!/usr/bin/env python
"""
多源融合渗透率预测模型 - 主训练脚本
支持所有融合策略：基础CNN、特征拼接、跨模态注意力、决策级融合
包含超参数优化功能
"""

import argparse
import os
import sys
import json
import time
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from models.base_cnn import BaseConv3DNet
from models.feature_concat import FeatureConcatNet
from models.cross_attention import CrossAttentionNet
from models.decision_fusion import (
    DecisionLevelFusionManager, AverageFusion, EntropyWeightFusion,
    LinearRegressionFusion, SVMFusion
)
from utils.data_loader import create_dataloader, set_seed, transform_label_to_perm
from utils.logger import TrainingLogger
from utils.visualization import plot_loss_curve
from utils.model_config import ModelConfigManager
from utils.optimization import (
    BaseCNNOptimizer, FeatureConcatOptimizer, 
    CrossAttentionOptimizer, SVMFusionOptimizer
)

# 常量定义
SEED = 42
set_seed(SEED)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多源融合渗透率预测模型训练')
    
    # 模型选择
    parser.add_argument('--strategy', type=str, default='attention',
                       choices=['base', 'concat', 'attention', 'decision'],
                       help='融合策略: base (基础CNN), concat (特征拼接), '
                            'attention (跨模态注意力), decision (决策级融合)')
    
    # 决策级融合特定参数
    parser.add_argument('--fusion_method', type=str, default='all',
                       choices=['average', 'entropy', 'linear', 'svm', 'all'],
                       help='决策级融合方法 (仅在strategy=decision时使用)')
    parser.add_argument('--image_model_path', type=str, default=None,
                       help='图像分支模型路径 (用于决策级融合)')
    parser.add_argument('--physics_model_path', type=str, default=None,
                       help='物理分支预测结果路径 (CSV/Excel文件)')
    
    # 超参数优化
    parser.add_argument('--optimize', action='store_true',
                       help='训练前进行超参数优化')
    parser.add_argument('--optimize_only', action='store_true',
                       help='仅进行超参数优化，不训练')
    parser.add_argument('--optuna_trials', type=int, default=30,
                       help='Optuna优化试验次数')
    parser.add_argument('--use_best_params', action='store_true',
                       help='使用预优化的最佳参数')
    parser.add_argument('--params_dir', type=str, default='optimization_results',
                       help='优化参数存储目录')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, default='data/Train_set',
                       help='训练数据文件夹路径')
    parser.add_argument('--train_excel', type=str, default='data/Train_set.xlsx',
                       help='训练数据Excel文件路径')
    parser.add_argument('--val_data', type=str, default='data/Valid_set',
                       help='验证数据文件夹路径')
    parser.add_argument('--val_excel', type=str, default='data/Valid_set.xlsx',
                       help='验证数据Excel文件路径')
    parser.add_argument('--test_data', type=str, default='data/Test_set',
                       help='测试数据文件夹路径')
    parser.add_argument('--test_excel', type=str, default='data/Test_set.xlsx',
                       help='测试数据Excel文件路径')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (默认: 模型特定)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率 (默认: 模型特定)')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='权重衰减 (默认: 模型特定)')
    parser.add_argument('--momentum', type=float, default=None,
                       help='动量 (默认: 模型特定)')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='日志保存目录')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='实验名称 (用于日志记录)')
    parser.add_argument('--eval_only', action='store_true',
                       help='仅评估不训练')
    parser.add_argument('--plot_attention', action='store_true',
                       help='可视化注意力权重 (仅注意力模型)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()

def get_default_experiment_name(args):
    """获取默认的实验名称"""
    if args.experiment_name:
        return args.experiment_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.strategy}_{timestamp}"

def get_model_config(args, best_params=None):
    """获取模型配置参数"""
    # 如果使用最佳参数且提供了最佳参数
    if args.use_best_params and best_params:
        config = best_params.copy()
        
        # 确保必要的参数存在
        if args.strategy != 'decision':
            config.setdefault('lr', 0.001)
            config.setdefault('weight_decay', 0.0)
            config.setdefault('momentum', 0.9)
            config.setdefault('leaky_relu_slope', 0.01)
        
        # 设置训练轮数
        config['epochs'] = args.epochs if args.epochs else 100
        
        return config
    
    # 否则使用默认配置
    default_configs = {
        'base': {
            'lr': 0.0058,
            'weight_decay': 0.0001,
            'momentum': 0.8672,
            'leaky_relu_slope': 0.01,
            'epochs': 50,
            'hidden_dims': [1024, 512, 256, 128, 64, 32]
        },
        'concat': {
            'lr': 0.00503,
            'weight_decay': 0.0015,
            'momentum': 0.8874,
            'leaky_relu_slope': 0.1,
            'epochs': 100,
            'hidden_dims': [2048, 1024, 512, 256, 128, 64, 32]
        },
        'attention': {
            'lr': 0.01465,
            'weight_decay': 0.0015,
            'momentum': 0.8947,
            'leaky_relu_slope': 0.1,
            'attention_dim': 8192,
            'epochs': 100,
            'hidden_dims': [4096, 2048, 1024, 512, 256, 128, 64, 32]
        },
        'decision': {
            'epochs': 1  # 决策级融合只需要一轮训练
        }
    }
    
    config = default_configs.get(args.strategy, default_configs['attention'])
    
    # 覆盖命令行参数
    if args.lr is not None:
        config['lr'] = args.lr
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.momentum is not None:
        config['momentum'] = args.momentum
    if args.epochs is not None:
        config['epochs'] = args.epochs
    
    return config

def create_model(args, model_config=None):
    """创建模型实例"""
    if model_config is None:
        model_config = {}
    
    if args.strategy == 'base':
        model = BaseConv3DNet(config=model_config)
        print("创建基础CNN模型")
    elif args.strategy == 'concat':
        model = FeatureConcatNet(config=model_config)
        print("创建特征拼接模型")
    elif args.strategy == 'attention':
        model = CrossAttentionNet(config=model_config)
        print("创建跨模态注意力模型")
    elif args.strategy == 'decision':
        model = DecisionLevelFusionManager()
        print("创建决策级融合管理器")
    else:
        raise ValueError(f"未知策略: {args.strategy}")
    
    return model

def run_optimization(args, device):
    """运行超参数优化"""
    print("\n" + "="*60)
    print("开始超参数优化")
    print("="*60)
    
    # 准备数据配置
    data_config = {
        'train_data': args.train_data,
        'train_excel': args.train_excel,
        'val_data': args.val_data,
        'val_excel': args.val_excel,
        'image_pred_path': args.image_model_path,
        'physics_pred_path': args.physics_model_path
    }
    
    # 根据策略选择优化器
    optimizer_map = {
        'base': BaseCNNOptimizer,
        'concat': FeatureConcatOptimizer,
        'attention': CrossAttentionOptimizer,
        'decision': SVMFusionOptimizer if args.fusion_method == 'svm' else None
    }
    
    optimizer_class = optimizer_map.get(args.strategy)
    
    if optimizer_class is None:
        print(f"策略 {args.strategy} 不需要或尚未实现超参数优化")
        return None
    
    # 创建优化器
    optimizer = optimizer_class(f"{args.strategy}_model", device, data_config)
    
    # 运行优化
    print(f"优化 {args.strategy} 模型，试验次数: {args.optuna_trials}")
    study = optimizer.optimize(n_trials=args.optuna_trials)
    
    # 获取最佳参数
    best_params = optimizer.get_best_params()
    
    # 保存最佳参数
    os.makedirs(args.params_dir, exist_ok=True)
    params_path = os.path.join(args.params_dir, f'{args.strategy}_best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"最佳参数已保存到: {params_path}")
    
    return best_params

def train_single_model(args, device, model_config=None):
    """训练单个模型（基础CNN、特征拼接、跨模态注意力）"""
    if model_config is None:
        model_config = {}
    
    # 创建模型
    model = create_model(args, model_config)
    model = nn.DataParallel(model).to(device)
    
    # 获取训练参数
    epochs = model_config.get('epochs', 100)
    lr = model_config.get('lr', 0.001)
    weight_decay = model_config.get('weight_decay', 0.0)
    momentum = model_config.get('momentum', 0.9)
    
    # 设置优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay, 
        momentum=momentum
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 创建数据加载器
    use_extra_features = args.strategy != 'base'
    
    train_loader = create_dataloader(
        args.train_data, args.train_excel,
        batch_size=args.batch_size,
        shuffle=True,
        use_extra_features=use_extra_features,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        args.val_data, args.val_excel,
        batch_size=args.batch_size // 2,
        shuffle=False,
        use_extra_features=use_extra_features,
        num_workers=args.num_workers
    )
    
    # 创建日志记录器
    logger = TrainingLogger(args.log_dir, args.experiment_name)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"\n开始训练 {args.strategy} 模型")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {lr:.4f}")
    print(f"使用额外特征: {use_extra_features}")
    print("-" * 50)
    
    try:
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as tepoch:
                for batch in tepoch:
                    if args.strategy == 'base':
                        data, targets = batch
                        data = data.to(device)
                        targets = targets.to(device)
                        
                        with torch.autocast(device_type=device.type):
                            outputs = model(data)
                            loss = criterion(outputs.squeeze(), targets)
                    else:
                        data, targets, extra_feature = batch
                        data = data.to(device)
                        targets = targets.to(device)
                        extra_feature = extra_feature.to(device)
                        
                        with torch.autocast(device_type=device.type):
                            outputs = model(data, extra_feature)
                            loss = criterion(outputs.squeeze(), targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item() * data.size(0)
                    tepoch.set_postfix(loss=loss.item())
            
            # 验证阶段
            val_loss = 0.0
            model.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    if args.strategy == 'base':
                        data, targets = batch
                        data = data.to(device)
                        targets = targets.to(device)
                        outputs = model(data)
                    else:
                        data, targets, extra_feature = batch
                        data = data.to(device)
                        targets = targets.to(device)
                        extra_feature = extra_feature.to(device)
                        outputs = model(data, extra_feature)
                    
                    val_loss += criterion(outputs.squeeze(), targets).item()
            
            # 计算平均损失
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 记录指标
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'time': time.time() - start_time
            }
            
            logger.log_metrics(epoch + 1, metrics)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_save_path = os.path.join(args.save_dir, f'best_model_{args.strategy}.pth')
                torch.save(model.state_dict(), model_save_path)
                print(f"  [Epoch {epoch+1}] 保存最佳模型到 {model_save_path}")
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {metrics['time']:.1f}s")
    
    finally:
        logger.close()
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, f'final_model_{args.strategy}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型已保存到 {final_model_path}")
    
    # 绘制损失曲线
    loss_curve_path = os.path.join(args.save_dir, f'loss_curve_{args.strategy}.png')
    plot_loss_curve(train_losses, val_losses, loss_curve_path)
    print(f"损失曲线已保存到 {loss_curve_path}")
    
    return model

def train_decision_fusion(args, device):
    """训练决策级融合模型"""
    print("\n" + "="*60)
    print("训练决策级融合模型")
    print("="*60)
    
    # 检查必要的参数
    if not args.image_model_path:
        raise ValueError("决策级融合需要指定图像分支模型路径 (--image_model_path)")
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    
    # 1. 加载图像分支模型
    print("\n1. 加载图像分支模型...")
    image_model = BaseConv3DNet()
    image_model = nn.DataParallel(image_model).to(device)
    
    if os.path.exists(args.image_model_path):
        image_model.load_state_dict(torch.load(args.image_model_path, map_location=device, weights_only=True))
        image_model.eval()
        print(f"  已加载图像模型: {args.image_model_path}")
    else:
        raise FileNotFoundError(f"图像模型未找到: {args.image_model_path}")
    
    # 2. 加载或生成物理分支预测
    print("\n2. 加载/生成物理分支预测...")
    
    if args.physics_model_path and os.path.exists(args.physics_model_path):
        # 加载物理分支预测结果
        if args.physics_model_path.endswith('.csv'):
            physics_pred_df = pd.read_csv(args.physics_model_path)
        else:
            physics_pred_df = pd.read_excel(args.physics_model_path)
        
        # 假设数据已经按训练/验证/测试顺序排列
        total_samples = len(physics_pred_df)
        train_samples = int(total_samples * 0.7)
        val_samples = int(total_samples * 0.15)
        
        train_physics_pred = physics_pred_df['prediction'].values[:train_samples]
        val_physics_pred = physics_pred_df['prediction'].values[train_samples:train_samples+val_samples]
        
        print(f"  已加载物理分支预测: {args.physics_model_path}")
        print(f"  训练集样本数: {len(train_physics_pred)}, 验证集样本数: {len(val_physics_pred)}")
    else:
        print("  警告: 未提供物理分支预测路径")
        print("  将使用随机数据作为演示")
        
        # 加载训练集和验证集真实值
        from utils.data_loader import PorousMediaDataset
        train_dataset = create_dataloader(args.train_data, args.train_excel, batch_size=1, 
                                         use_extra_features=False).dataset
        val_dataset = create_dataloader(args.val_data, args.val_excel, batch_size=1,
                                       use_extra_features=False).dataset
        
        train_true = train_dataset.labels_real
        val_true = val_dataset.labels_real
        
        # 生成模拟物理预测
        np.random.seed(42)
        train_physics_pred = train_true * (1 + np.random.normal(0, 0.15, len(train_true)))
        val_physics_pred = val_true * (1 + np.random.normal(0, 0.15, len(val_true)))
    
    # 3. 生成图像分支预测
    print("\n3. 生成图像分支预测...")
    
    def generate_predictions(model, data_folder, excel_path, device, batch_size=16):
        """生成模型预测"""
        loader = create_dataloader(
            data_folder, excel_path,
            batch_size=batch_size,
            shuffle=False,
            use_extra_features=False
        )
        
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for data, targets in loader:
                data = data.to(device)
                outputs = model(data).cpu().numpy().squeeze()
                predictions.extend(outputs)
                true_values.extend(targets.numpy())
        
        # 转换回原始物理值
        predictions = transform_label_to_perm(np.array(predictions))
        true_values = transform_label_to_perm(np.array(true_values))
        
        return predictions, true_values
    
    # 生成训练集和验证集的图像预测
    train_image_pred, train_true = generate_predictions(
        image_model, args.train_data, args.train_excel, device, args.batch_size
    )
    val_image_pred, val_true = generate_predictions(
        image_model, args.val_data, args.val_excel, device, args.batch_size // 2
    )
    
    # 确保数据长度匹配
    min_train_len = min(len(train_image_pred), len(train_physics_pred), len(train_true))
    train_image_pred = train_image_pred[:min_train_len]
    train_physics_pred = train_physics_pred[:min_train_len]
    train_true = train_true[:min_train_len]
    
    min_val_len = min(len(val_image_pred), len(val_physics_pred), len(val_true))
    val_image_pred = val_image_pred[:min_val_len]
    val_physics_pred = val_physics_pred[:min_val_len]
    val_true = val_true[:min_val_len]
    
    print(f"  训练集: {min_train_len} 个样本")
    print(f"  验证集: {min_val_len} 个样本")
    
    # 4. 训练融合模型
    print("\n4. 训练融合模型...")
    
    # 创建融合管理器
    fusion_manager = DecisionLevelFusionManager()
    
    # 准备训练数据
    fusion_manager.image_predictions = train_image_pred
    fusion_manager.physics_predictions = train_physics_pred
    fusion_manager.true_values = train_true
    
    # 训练指定的融合方法
    if args.fusion_method == 'all':
        # 训练所有方法
        print("  训练所有融合方法...")
        results = fusion_manager.train_all_fusion_methods(train_ratio=0.8)
        
        # 获取最佳方法
        best_name, best_method, best_score = fusion_manager.get_best_method(results)
        print(f"\n  最佳融合方法: {best_name}, 测试集 R²: {best_score:.4f}")
        
        # 保存最佳方法
        best_method_path = os.path.join(args.save_dir, f'best_fusion_{best_name}.pkl')
        with open(best_method_path, 'wb') as f:
            pickle.dump(best_method, f)
        print(f"  最佳融合模型已保存到 {best_method_path}")
        
        # 在验证集上评估最佳方法
        val_predictions = best_method.predict(val_image_pred, val_physics_pred)
        
        val_metrics = {
            'r2': r2_score(val_true, val_predictions),
            'rmse': np.sqrt(mean_squared_error(val_true, val_predictions)),
            'mae': mean_absolute_error(val_true, val_predictions)
        }
        
        print("\n  最佳融合方法验证结果:")
        print(f"  R²: {val_metrics['r2']:.4f}")
        print(f"  RMSE: {val_metrics['rmse']:.4e}")
        print(f"  MAE: {val_metrics['mae']:.4e}")
        
        # 保存验证结果
        val_results_df = pd.DataFrame({
            'image_prediction': val_image_pred,
            'physics_prediction': val_physics_pred,
            'fused_prediction': val_predictions,
            'true_value': val_true
        })
        
        val_results_path = os.path.join(args.save_dir, 'validation_fusion_results.csv')
        val_results_df.to_csv(val_results_path, index=False)
        print(f"  验证结果已保存到 {val_results_path}")
        
        # 保存所有方法的结果
        results_df = pd.DataFrame({
            'Method': list(results.keys()),
            'Train_R2': [results[m]['train']['r2'] for m in results.keys()],
            'Test_R2': [results[m]['test']['r2'] for m in results.keys()],
            'Test_RMSE': [results[m]['test']['rmse'] for m in results.keys()]
        })
        
        results_path = os.path.join(args.save_dir, 'all_fusion_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"  所有融合方法结果已保存到 {results_path}")
        
        return best_method, results
    
    else:
        # 训练指定方法
        fusion_methods = {
            'average': AverageFusion(),
            'entropy': EntropyWeightFusion(),
            'linear': LinearRegressionFusion(),
            'svm': SVMFusion(use_optuna=True)
        }
        
        if args.fusion_method not in fusion_methods:
            raise ValueError(f"未知的融合方法: {args.fusion_method}")
        
        method = fusion_methods[args.fusion_method]
        print(f"\n  训练 {args.fusion_method} 融合方法...")
        
        # 划分训练集和测试集
        n_samples = len(train_true)
        n_train = int(n_samples * 0.8)
        
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        # 训练
        method.fit(
            train_image_pred[train_idx],
            train_physics_pred[train_idx],
            train_true[train_idx]
        )
        
        # 在测试集上评估
        test_metrics = method.evaluate(
            train_image_pred[test_idx],
            train_physics_pred[test_idx],
            train_true[test_idx]
        )
        
        print(f"  {args.fusion_method} 测试集 R²: {test_metrics['r2']:.4f}")
        
        # 保存模型
        model_path = os.path.join(args.save_dir, f'fusion_{args.fusion_method}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(method, f)
        print(f"  融合模型已保存到 {model_path}")
        
        return method, {args.fusion_method: test_metrics}

def evaluate_model_all_sets(args, model, device):
    """在所有数据集上评估模型"""
    print("\n" + "="*60)
    print("在所有数据集上评估模型")
    print("="*60)
    
    from utils.metrics import evaluate_model as evaluate_model_metrics
    
    results = {}
    
    datasets = [
        ('Train', args.train_data, args.train_excel),
        ('Validation', args.val_data, args.val_excel),
        ('Test', args.test_data, args.test_excel)
    ]
    
    for name, data_path, excel_path in datasets:
        if not os.path.exists(data_path):
            print(f"\n{name} 数据集不存在: {data_path}")
            continue
            
        print(f"\n{name} 集评估:")
        
        # 创建数据加载器
        use_extra_features = args.strategy != 'base'
        
        loader = create_dataloader(
            data_path, excel_path,
            batch_size=args.batch_size // 2,
            shuffle=False,
            use_extra_features=use_extra_features,
            num_workers=args.num_workers
        )
        
        # 评估模型
        metrics = evaluate_model_metrics(model, loader, device, args.strategy)
        results[name] = metrics
        
        # 打印结果
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4e}")
        print(f"  MAE: {metrics['mae']:.4e}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
    
    return results

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 设置实验名称
    if not args.experiment_name:
        args.experiment_name = get_default_experiment_name(args)
    
    # 如果是仅优化模式
    if args.optimize_only:
        print("\n" + "="*60)
        print("仅运行超参数优化模式")
        print("="*60)
        
        best_params = run_optimization(args, device)
        
        if best_params:
            print(f"\n优化完成！最佳参数: {best_params}")
        else:
            print("\n优化失败或不需要优化")
        
        return
    
    # 如果是仅评估模式
    if args.eval_only:
        print("\n" + "="*60)
        print("仅评估模式")
        print("="*60)
        
        if args.strategy == 'decision':
            print("决策级融合模型评估请使用 test.py")
            return
        
        # 加载模型
        model = create_model(args)
        model = nn.DataParallel(model).to(device)
        
        # 加载检查点
        checkpoint_path = os.path.join(args.save_dir, f'best_model_{args.strategy}.pth')
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print(f"已加载模型: {checkpoint_path}")
        else:
            print(f"警告: 检查点未找到: {checkpoint_path}")
            print("使用未训练模型进行评估")
        
        # 评估模型
        evaluate_model_all_sets(args, model, device)
        
        return
    
    # 训练模式
    print("\n" + "="*70)
    print("多源融合渗透率预测模型训练")
    print("="*70)
    
    # 打印配置信息
    print(f"策略: {args.strategy}")
    print(f"实验名称: {args.experiment_name}")
    print(f"保存目录: {args.save_dir}")
    print(f"日志目录: {args.log_dir}")
    print(f"随机种子: {args.seed}")
    
    if args.strategy == 'decision':
        print(f"融合方法: {args.fusion_method}")
        print(f"图像模型路径: {args.image_model_path}")
        print(f"物理模型路径: {args.physics_model_path}")
    else:
        print(f"批次大小: {args.batch_size}")
        print(f"使用最佳参数: {args.use_best_params}")
    
    # 运行超参数优化（如果需要）
    best_params = None
    if args.optimize:
        best_params = run_optimization(args, device)
        if best_params:
            args.use_best_params = True
    
    # 加载最佳参数（如果指定）
    elif args.use_best_params:
        # 尝试从文件加载最佳参数
        params_path = os.path.join(args.params_dir, f'{args.strategy}_best_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            print(f"\n已加载最佳参数: {best_params}")
        else:
            print(f"\n警告: 未找到最佳参数文件: {params_path}")
            print("将使用默认参数")
    
    # 获取模型配置
    model_config = get_model_config(args, best_params)
    
    # 根据策略选择训练函数
    if args.strategy == 'decision':
        trained_model, results = train_decision_fusion(args, device)
        
        # 可视化注意力（如果适用）
        if args.plot_attention and args.strategy == 'attention':
            try:
                # 加载验证集
                val_loader = create_dataloader(
                    args.val_data, args.val_excel,
                    batch_size=1,
                    shuffle=True,
                    use_extra_features=True
                )
                
                # 可视化注意力
                fig = trained_model.module.visualize_attention(val_loader, device, num_samples=3)
                fig.savefig(os.path.join(args.save_dir, 'attention_visualization.png'))
                print(f"\n注意力可视化已保存到 {os.path.join(args.save_dir, 'attention_visualization.png')}")
            except Exception as e:
                print(f"无法可视化注意力: {e}")
    else:
        trained_model = train_single_model(args, device, model_config)
        
        # 在所有数据集上评估模型
        evaluate_model_all_sets(args, trained_model, device)
        
        # 可视化注意力（如果适用）
        if args.plot_attention and args.strategy == 'attention':
            try:
                # 加载验证集
                val_loader = create_dataloader(
                    args.val_data, args.val_excel,
                    batch_size=1,
                    shuffle=True,
                    use_extra_features=True
                )
                
                # 可视化注意力
                fig = trained_model.module.visualize_attention(val_loader, device, num_samples=3)
                fig.savefig(os.path.join(args.save_dir, 'attention_visualization.png'))
                print(f"\n注意力可视化已保存到 {os.path.join(args.save_dir, 'attention_visualization.png')}")
            except Exception as e:
                print(f"无法可视化注意力: {e}")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)

if __name__ == "__main__":
    main()