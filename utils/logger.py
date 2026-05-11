import logging
import os
import sys
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from typing import Dict, Any
from datetime import datetime
import torch

class TrainingLogger:
    """训练日志管理器"""
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        self.log_queue = Queue()
        self.listener = QueueListener(self.log_queue, self._create_csv_handler())
        self.listener.start()

        self.logger = logging.getLogger('training')
        self.logger.addHandler(QueueHandler(self.log_queue))
        self.logger.setLevel(logging.INFO)

        self._log_metadata()

    def _create_csv_handler(self):
        """创建CSV文件处理器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        try:
            csv_handler = logging.FileHandler(
                filename=os.path.join(self.log_dir, f'{self.experiment_name}_{timestamp}.csv'),
                mode='a',
                encoding='utf-8'
            )
            csv_handler.setFormatter(CSVFormatter())
            return csv_handler
        except Exception as e:
            print(f"Error creating CSV log handler: {e}")
            return None  

    def _log_metadata(self):
        """记录实验元数据"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.experiment_name,
            'environment': {
                'python_version': sys.version,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count()
            }
        }
        self.logger.info({'type': 'metadata', 'content': metadata})

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """记录训练指标"""
        log_entry = {
            'type': 'metrics',
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(log_entry)

    def close(self):
        """关闭日志系统"""
        self.listener.stop()

class CSVFormatter(logging.Formatter):
    """自定义CSV格式"""
    def format(self, record):
        if isinstance(record.msg, dict) and record.msg.get('type') == 'metrics':
            data = record.msg
            train_loss = data['metrics'].get('train_loss', 'nan')
            val_loss = data['metrics'].get('val_loss', 'nan')
            lr = data['metrics'].get('lr', 'nan')
            time = data['metrics'].get('time', 'nan')
            return (f"{data['epoch']},{train_loss:.6f},{val_loss:.6f},{lr:.2e},{time:.2f}")
        return super().format(record)