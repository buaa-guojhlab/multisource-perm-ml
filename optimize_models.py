#!/usr/bin/env python
"""
超参数优化主脚本
使用Optuna优化所有模型的超参数
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

import torch
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour
)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.optimization import (
    BaseCNNOptimizer, FeatureConcatOptimizer,
    CrossAttentionOptimizer, SVMFusionOptimizer
)
from utils.data_loader import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='优化多源融合渗透率预测模型的超参数')
    
    # 优化目标
    parser.add_argument('--models', type=str, default='all',
                       choices=['all', 'base', 'concat', 'attention', 'svm'],
                       help='要优化的模型 (all: 所有模型)')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, default='data/4.2T',
                       help='训练数据文件夹路径')
    parser.add_argument('--train_excel', type=str, default='data/4.2T.xlsx',
                       help='训练数据Excel文件路径')
    parser.add_argument('--val_data', type=str, default='data/4.2V',
                       help='验证数据文件夹路径')
    parser.add_argument('--val_excel', type=str, default='data/4.2V.xlsx',
                       help='验证数据Excel文件路径')
    
    # 决策级融合数据
    parser.add_argument('--image_pred_path', type=str, default='predictions/image_predictions.csv',
                       help='图像分支预测路径')
    parser.add_argument('--physics_pred_path', type=str, default='predictions/physics_predictions.csv',
                       help='物理分支预测路径')
    parser.add_argument('--true_values_path', type=str, default='predictions/true_values.csv',
                       help='真实值路径')
    
    # 优化参数
    parser.add_argument('--trials', type=int, default=50,
                       help='每个模型的试验次数')
    parser.add_argument('--timeout', type=int, default=None,
                       help='优化超时时间 (秒)')
    parser.add_argument('--output_dir', type=str, default='optimization_results',
                       help='输出目录')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--parallel', action='store_true',
                       help='并行优化多个模型')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='并行作业数')
    
    return parser.parse_args()

def optimize_model(model_name, data_config, output_dir, n_trials=50, timeout=None, 
                   visualize=False, seed=42):
    """优化单个模型"""
    print(f"\n{'='*60}")
    print(f"优化 {model_name} 模型")
    print(f"{'='*60}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(seed)
    
    # 创建优化器
    optimizer_map = {
        'base': (BaseCNNOptimizer, 'base_cnn'),
        'concat': (FeatureConcatOptimizer, 'feature_concat'),
        'attention': (CrossAttentionOptimizer, 'cross_attention'),
        'svm': (SVMFusionOptimizer, 'svm_fusion')
    }
    
    if model_name not in optimizer_map:
        raise ValueError(f"不支持的模型: {model_name}")
    
    optimizer_class, internal_name = optimizer_map[model_name]
    optimizer = optimizer_class(internal_name, device, data_config)
    
    # 运行优化
    start_time = time.time()
    
    try:
        study = optimizer.optimize(n_trials=n_trials, timeout=timeout, seed=seed)
        optimization_time = time.time() - start_time
        
        # 获取最佳参数
        best_params = optimizer.get_best_params()
        best_value = optimizer.best_value
        
        print(f"\n优化完成!")
        print(f"最佳验证损失: {best_value:.6f}")
        print(f"优化时间: {optimization_time:.1f} 秒")
        print(f"最佳参数: {best_params}")
        
        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存最佳参数
        params_path = os.path.join(output_dir, f'{internal_name}_best_params.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"最佳参数已保存: {params_path}")
        
        # 保存研究
        study_path = os.path.join(output_dir, f'{internal_name}_study.pkl')
        import joblib
        joblib.dump(study, study_path)
        print(f"研究已保存: {study_path}")
        
        # 生成可视化图表
        if visualize and study:
            try:
                print("生成可视化图表...")
                
                # 优化历史
                fig1 = plot_optimization_history(study)
                fig1_path = os.path.join(output_dir, f'{internal_name}_optimization_history.html')
                fig1.write_html(fig1_path)
                
                # 参数重要性
                fig2 = plot_param_importances(study)
                fig2_path = os.path.join(output_dir, f'{internal_name}_param_importances.html')
                fig2.write_html(fig2_path)
                
                # 切片图
                fig3 = plot_slice(study)
                fig3_path = os.path.join(output_dir, f'{internal_name}_slice_plot.html')
                fig3.write_html(fig3_path)
                
                # 等高线图（如果有两个主要参数）
                try:
                    fig4 = plot_contour(study)
                    fig4_path = os.path.join(output_dir, f'{internal_name}_contour_plot.html')
                    fig4.write_html(fig4_path)
                except:
                    print("  无法生成等高线图（可能需要至少两个数值参数）")
                
                print(f"可视化图表已保存到 {output_dir}")
                
            except Exception as e:
                print(f"生成可视化图表时出错: {e}")
        
        return {
            'model': model_name,
            'internal_name': internal_name,
            'best_params': best_params,
            'best_value': best_value,
            'optimization_time': optimization_time,
            'study': study
        }
        
    except Exception as e:
        print(f"优化 {model_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("多源融合渗透率预测模型超参数优化")
    print("="*70)
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 准备数据配置
    data_config = {
        'train_data': args.train_data,
        'train_excel': args.train_excel,
        'val_data': args.val_data,
        'val_excel': args.val_excel,
        'image_pred_path': args.image_pred_path,
        'physics_pred_path': args.physics_pred_path,
        'true_values_path': args.true_values_path
    }
    
    print("\n数据配置:")
    for key, value in data_config.items():
        if value and os.path.exists(value):
            print(f"  {key}: {value} ✓")
        elif value:
            print(f"  {key}: {value} (未找到)")
    
    # 确定要优化的模型
    if args.models == 'all':
        models_to_optimize = ['base', 'concat', 'attention', 'svm']
    else:
        models_to_optimize = [args.models]
    
    print(f"\n要优化的模型: {', '.join(models_to_optimize)}")
    print(f"每个模型的试验次数: {args.trials}")
    if args.timeout:
        print(f"每个模型的超时时间: {args.timeout} 秒")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f'optimization_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_path = os.path.join(output_dir, 'optimization_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'data_config': data_config,
            'models': models_to_optimize,
            'trials': args.trials,
            'timeout': args.timeout,
            'seed': args.seed,
            'visualize': args.visualize,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"\n配置已保存: {config_path}")
    print(f"结果将保存到: {output_dir}")
    
    # 优化模型
    results = {}
    
    for model_name in models_to_optimize:
        result = optimize_model(
            model_name=model_name,
            data_config=data_config,
            output_dir=output_dir,
            n_trials=args.trials,
            timeout=args.timeout,
            visualize=args.visualize,
            seed=args.seed
        )
        
        if result:
            results[model_name] = result
    
    # 生成优化报告
    if results:
        print(f"\n{'='*70}")
        print("优化完成汇总")
        print(f"{'='*70}")
        
        report_path = os.path.join(output_dir, 'optimization_report.md')
        with open(report_path, 'w') as f:
            f.write("# 超参数优化报告\n\n")
            f.write(f"## 基本信息\n")
            f.write(f"- 优化时间: {timestamp}\n")
            f.write(f"- 设备: {device}\n")
            f.write(f"- 随机种子: {args.seed}\n")
            f.write(f"- 试验次数: {args.trials} 每模型\n")
            if args.timeout:
                f.write(f"- 超时时间: {args.timeout} 秒每模型\n")
            f.write(f"\n## 优化的模型\n")
            
            for model_name in results.keys():
                f.write(f"- {model_name}\n")
            
            f.write(f"\n## 最佳参数\n")
            
            for model_name, result in results.items():
                f.write(f"\n### {model_name}\n\n")
                f.write(f"- 最佳验证损失: {result['best_value']:.6f}\n")
                f.write(f"- 优化时间: {result['optimization_time']:.1f} 秒\n\n")
                f.write("```json\n")
                f.write(json.dumps(result['best_params'], indent=2))
                f.write("\n```\n")
            
            f.write(f"\n## 使用最佳参数训练模型的命令\n")
            
            for model_name in results.keys():
                strategy_map = {
                    'base': 'base',
                    'concat': 'concat',
                    'attention': 'attention',
                    'svm': 'decision --fusion_method svm'
                }
                
                strategy = strategy_map.get(model_name, model_name)
                f.write(f"\n### {model_name} 模型\n")
                f.write(f"```bash\n")
                f.write(f"python train.py --strategy {strategy} --use_best_params --params_dir {output_dir}\n")
                f.write(f"```\n")
        
        print(f"\n优化报告已生成: {report_path}")
        
        # 打印汇总表格
        print(f"\n{'='*70}")
        print("优化结果汇总")
        print(f"{'='*70}")
        print(f"{'模型':<15} {'最佳损失':<15} {'优化时间(秒)':<15} {'参数文件'}")
        print(f"{'-'*60}")
        
        for model_name, result in results.items():
            internal_name = result['internal_name']
            params_file = f"{internal_name}_best_params.json"
            print(f"{model_name:<15} {result['best_value']:<15.6f} {result['optimization_time']:<15.1f} {params_file}")
        
        print(f"\n下一步建议:")
        print(f"1. 使用最佳参数训练模型:")
        for model_name in results.keys():
            strategy_map = {
                'base': 'base',
                'concat': 'concat',
                'attention': 'attention',
                'svm': 'decision --fusion_method svm'
            }
            strategy = strategy_map.get(model_name, model_name)
            print(f"   python train.py --strategy {strategy} --use_best_params --params_dir {output_dir}")
        
        print(f"\n2. 查看可视化图表:")
        print(f"   优化历史: {output_dir}/*_optimization_history.html")
        print(f"   参数重要性: {output_dir}/*_param_importances.html")
        
        print(f"\n3. 使用Optuna仪表板查看详细结果:")
        print(f"   optuna-dashboard sqlite:///{output_dir}/studies.db")
        
    else:
        print("\n没有成功优化的模型")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())