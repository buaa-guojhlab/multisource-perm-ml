import matplotlib.pyplot as plt
import numpy as np

def plot_results(true, pred, errors, title, save_path):
    """绘制预测结果和误差分布"""
    plt.figure(figsize=(12, 6))
    
    # 子图1：预测值对比
    plt.subplot(1, 2, 1)
    plt.scatter(true, pred, c=errors, cmap='coolwarm', alpha=0.6)
    plt.plot([min(true), max(true)], [min(true), max(true)], 'k--')
    plt.colorbar(label='Percentage Error (%)')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{title} - Predictions')
    
    # 子图2：误差分布
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Density')
    plt.title(f'{title} - Error Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path)
    plt.close()