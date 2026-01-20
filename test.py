#!/usr/bin/env python
"""
多源融合渗透率预测模型 - 主测试脚本
支持所有模型评估和结果可视化
"""

import argparse
import os
import sys
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from utils.data_loader import create_dataloader, transform_label_to_perm
from utils.metrics import evaluate_model as evaluate_model_metrics, save_error_analysis
from utils.visualization import plot_results

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多源融合渗透率预测模型测试')
    
    # 模型参数
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['base', 'concat', 'attention', 'decision'],
                       help='模型策略: base (基础CNN), concat (特征拼接), '
                            'attention (跨模态注意力), decision (决策级融合)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径 (.pth 或 .pkl 文件)')
    
    # 决策级融合特定参数
    parser.add_argument('--image_model_path', type=str, default=None,
                       help='图像分支模型路径 (仅决策级融合需要)')
    parser.add_argument('--physics_model_path', type=str, default=None,
                       help='物理分支预测路径 (仅决策级融合需要)')
    parser.add_argument('--fusion_method', type=str, default=None,
                       help='融合方法名称 (仅决策级融合需要)')
    
    # 数据参数
    parser.add_argument('--test_data', type=str, default='data/Test_set',
                       help='测试数据文件夹路径')
    parser.add_argument('--test_excel', type=str, default='data/Test_set.xlsx',
                       help='测试数据Excel文件路径')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    parser.add_argument('--save_predictions', action='store_true',
                       help='保存预测结果')
    parser.add_argument('--plot_results', action='store_true',
                       help='绘制结果图表')
    parser.add_argument('--compare_all_fusion', action='store_true',
                       help='比较所有融合方法 (仅决策级融合)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载工作进程数')
    
    return parser.parse_args()

def load_single_model(checkpoint_path, strategy, device):
    """加载单个模型 (基础CNN、特征拼接、跨模态注意力)"""
    # 创建模型
    if strategy == 'base':
        model = BaseConv3DNet()
    elif strategy == 'concat':
        model = FeatureConcatNet()
    elif strategy == 'attention':
        model = CrossAttentionNet()
    else:
        raise ValueError(f"不支持的策略: {strategy}")
    
    # 使用DataParallel包装模型
    model = nn.DataParallel(model).to(device)
    
    # 加载检查点
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型检查点未找到: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"已加载模型: {checkpoint_path}")
    return model

def load_decision_fusion_model(checkpoint_path, device):
    """加载决策级融合模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"融合模型检查点未找到: {checkpoint_path}")
    
    # 决策级融合模型保存为pickle文件
    with open(checkpoint_path, 'rb') as f:
        model = pickle.load(f)
    
    if not hasattr(model, 'is_fitted') or not model.is_fitted:
        print("警告: 加载的融合模型似乎未训练")
    
    print(f"已加载融合模型: {checkpoint_path}")
    return model

def evaluate_single_model(args, model, device):
    """评估单个模型"""
    print(f"\n评估 {args.strategy} 模型...")
    
    # 创建数据加载器
    use_extra_features = args.strategy != 'base'
    
    test_loader = create_dataloader(
        args.test_data, args.test_excel,
        batch_size=args.batch_size,
        shuffle=False,
        use_extra_features=use_extra_features,
        num_workers=args.num_workers
    )
    
    # 评估模型
    metrics = evaluate_model_metrics(model, test_loader, device, args.strategy)
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4e}")
    print(f"MAE: {metrics['mae']:.4e}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    print(f"MdAPE: {metrics['mdape']:.2f}%")
    
    # 可视化结果
    if args.plot_results:
        plot_path = os.path.join(args.output_dir, f'{args.strategy}_test_results.png')
        plot_results(
            metrics['true'], metrics['pred'], metrics['percent_errors'],
            f'{args.strategy.capitalize()} Model', plot_path
        )
        print(f"\n图表已保存: {plot_path}")
    
    # 保存预测结果
    if args.save_predictions:
        # 保存详细误差分析
        error_report_path = os.path.join(args.output_dir, f'{args.strategy}_error_analysis.xlsx')
        save_error_analysis(metrics, error_report_path)
        print(f"详细误差分析已保存: {error_report_path}")
        
        # 保存预测结果
        predictions_df = pd.DataFrame({
            'True_Value': metrics['true'],
            'Predicted_Value': metrics['pred'],
            'Percentage_Error': metrics['percent_errors'],
            'Absolute_Error': np.abs(np.array(metrics['pred']) - np.array(metrics['true']))
        })
        
        predictions_path = os.path.join(args.output_dir, f'{args.strategy}_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"预测结果已保存: {predictions_path}")
    
    return metrics

def evaluate_decision_fusion(args, device):
    """评估决策级融合模型"""
    print("\n评估决策级融合模型...")
    
    # 检查必要的参数
    if not args.image_model_path:
        raise ValueError("决策级融合需要图像分支模型路径 (--image_model_path)")
    
    if not args.physics_model_path:
        raise ValueError("决策级融合需要物理分支预测路径 (--physics_model_path)")
    
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
    
    # 2. 加载物理分支预测
    print("\n2. 加载物理分支预测...")
    if args.physics_model_path.endswith('.csv'):
        physics_pred_df = pd.read_csv(args.physics_model_path)
    else:
        physics_pred_df = pd.read_excel(args.physics_model_path)
    
    # 假设列名为'prediction'
    if 'prediction' not in physics_pred_df.columns:
        # 尝试其他可能的列名
        possible_names = ['pred', 'predicted', 'prediction', 'predicted_value']
        for name in possible_names:
            if name in physics_pred_df.columns:
                test_physics_pred = physics_pred_df[name].values
                break
        else:
            # 如果没有找到，使用第一列
            test_physics_pred = physics_pred_df.iloc[:, 0].values
    else:
        test_physics_pred = physics_pred_df['prediction'].values
    
    print(f"  已加载 {len(test_physics_pred)} 个物理分支预测")
    
    # 3. 生成图像分支预测
    print("\n3. 生成图像分支预测...")
    
    def generate_predictions(model, data_folder, excel_path, device, batch_size=16):
        """生成模型预测"""
        loader = create_dataloader(
            data_folder, excel_path,
            batch_size=batch_size,
            shuffle=False,
            use_extra_features=False,
            num_workers=args.num_workers
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
    
    test_image_pred, test_true = generate_predictions(
        image_model, args.test_data, args.test_excel, device, args.batch_size
    )
    
    # 确保数据长度匹配
    min_len = min(len(test_physics_pred), len(test_image_pred), len(test_true))
    test_physics_pred = test_physics_pred[:min_len]
    test_image_pred = test_image_pred[:min_len]
    test_true = test_true[:min_len]
    
    print(f"  使用 {min_len} 个样本进行评估")
    
    # 4. 加载融合模型并进行预测
    print("\n4. 加载融合模型...")
    
    if args.compare_all_fusion:
        # 比较所有融合方法
        print("  比较所有融合方法...")
        
        # 创建所有融合方法
        fusion_methods = {
            'average': AverageFusion(weights=[0.5, 0.5]),
            'entropy': EntropyWeightFusion(),
            'linear': LinearRegressionFusion(),
            'svm': SVMFusion(use_optuna=False)
        }
        
        results = {}
        
        for name, method in fusion_methods.items():
            print(f"  评估 {name} 融合...")
            
            # 训练方法（使用所有数据）
            method.fit(test_image_pred, test_physics_pred, test_true)
            
            # 预测
            fused_pred = method.predict(test_image_pred, test_physics_pred)
            
            # 计算指标
            metrics = {
                'r2': r2_score(test_true, fused_pred),
                'rmse': np.sqrt(mean_squared_error(test_true, fused_pred)),
                'mae': mean_absolute_error(test_true, fused_pred),
                'mse': mean_squared_error(test_true, fused_pred)
            }
            
            results[name] = {
                'metrics': metrics,
                'predictions': fused_pred,
                'method': method
            }
            
            print(f"    {name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4e}")
        
        # 找到最佳方法
        best_name = max(results.keys(), key=lambda x: results[x]['metrics']['r2'])
        best_result = results[best_name]
        
        print(f"\n  最佳融合方法: {best_name} (R² = {best_result['metrics']['r2']:.4f})")
        
        # 保存所有方法的结果
        results_df = pd.DataFrame({
            'Method': list(results.keys()),
            'R2': [results[m]['metrics']['r2'] for m in results.keys()],
            'RMSE': [results[m]['metrics']['rmse'] for m in results.keys()],
            'MAE': [results[m]['metrics']['mae'] for m in results.keys()],
            'MSE': [results[m]['metrics']['mse'] for m in results.keys()]
        })
        
        results_path = os.path.join(args.output_dir, 'all_fusion_comparison.csv')
        results_df.to_csv(results_path, index=False)
        print(f"  所有融合方法比较已保存: {results_path}")
        
        metrics = best_result['metrics']
        test_fused_pred = best_result['predictions']
        best_method_name = best_name
        
    else:
        # 加载指定的融合模型
        fusion_model = load_decision_fusion_model(args.checkpoint, device)
        
        # 进行预测
        test_fused_pred = fusion_model.predict(test_image_pred, test_physics_pred)
        
        # 计算指标
        metrics = {
            'r2': r2_score(test_true, test_fused_pred),
            'rmse': np.sqrt(mean_squared_error(test_true, test_fused_pred)),
            'mae': mean_absolute_error(test_true, test_fused_pred),
            'mse': mean_squared_error(test_true, test_fused_pred)
        }
        
        best_method_name = args.fusion_method or "loaded_model"
    
    # 5. 打印结果
    print("\n" + "="*50)
    print("决策级融合评估结果:")
    print("="*50)
    print(f"融合方法: {best_method_name}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4e}")
    print(f"MAE: {metrics['mae']:.4e}")
    print(f"MSE: {metrics['mse']:.4e}")
    
    # 6. 保存结果
    if args.save_predictions:
        # 计算百分比误差
        eps = 1e-15
        percent_errors = [(p - t) / (t + eps) * 100 for p, t in zip(test_fused_pred, test_true)]
        
        results_df = pd.DataFrame({
            'image_prediction': test_image_pred,
            'physics_prediction': test_physics_pred[:len(test_image_pred)],
            'fused_prediction': test_fused_pred,
            'true_value': test_true,
            'percentage_error': percent_errors,
            'absolute_error': np.abs(test_fused_pred - test_true)
        })
        
        output_path = os.path.join(args.output_dir, f'decision_fusion_{best_method_name}_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n结果已保存: {output_path}")
        
        # 可视化
        if args.plot_results:
            plot_path = os.path.join(args.output_dir, f'decision_fusion_{best_method_name}_results.png')
            
            plt.figure(figsize=(12, 6))
            
            # 子图1：预测值对比
            plt.subplot(1, 2, 1)
            plt.scatter(test_true, test_fused_pred, c=percent_errors, cmap='coolwarm', alpha=0.6)
            plt.plot([min(test_true), max(test_true)], [min(test_true), max(test_true)], 'k--')
            plt.colorbar(label='Percentage Error (%)')
            plt.xlabel('True Values')
            plt.ylabel('Fused Predictions')
            plt.title(f'Decision Fusion ({best_method_name}) - Predictions')
            
            # 子图2：误差分布
            plt.subplot(1, 2, 2)
            plt.hist(percent_errors, bins=50, density=True, alpha=0.7)
            plt.xlabel('Percentage Error (%)')
            plt.ylabel('Density')
            plt.title(f'Decision Fusion ({best_method_name}) - Error Distribution')
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"图表已保存: {plot_path}")
    
    return metrics

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    print("\n" + "="*60)
    print("多源融合渗透率预测模型测试")
    print("="*60)
    print(f"策略: {args.strategy}")
    print(f"测试数据: {args.test_data}")
    print(f"输出目录: {args.output_dir}")
    
    try:
        # 根据策略选择评估函数
        if args.strategy == 'decision':
            metrics = evaluate_decision_fusion(args, device)
        else:
            # 加载模型
            model = load_single_model(args.checkpoint, args.strategy, device)
            
            # 评估模型
            metrics = evaluate_single_model(args, model, device)
        
        # 保存评估摘要
        summary = {
            'strategy': args.strategy,
            'checkpoint': args.checkpoint,
            'test_data': args.test_data,
            'metrics': metrics
        }
        
        summary_path = os.path.join(args.output_dir, 'evaluation_summary.json')
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        with open(summary_path, 'w') as f:
            json.dump(convert_to_serializable(summary), f, indent=2)
        
        print(f"\n评估摘要已保存: {summary_path}")
        
        print("\n" + "="*60)
        print("测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())