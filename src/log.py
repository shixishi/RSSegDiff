r"""借助Python标准库中的logging模块
实现日志记录函数
根据logger对象的创建时间命名日志文件
"""
import sys
import logging
from datetime import datetime


def get_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(message)s')

    # 创建Handler，分别将日志输出到终端和文件
    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 获得当前日期和时间并格式化为字符串
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 根据当前时间设置log文件名
    log_name = start_time+'.log'
    log_file = './log/'+log_name
    # Filehandler 
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

if __name__ == '__main__':
    import random
    logger = get_logger()
    # 将日期对象格式化为字符串
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'start_time \t {start_time}')

    for epoch in range(10):
        for step in range(10):
            logger.info(f'epoch {epoch+1:02d}\tstep {step+1:02d}\t'
                        f'\tloss {random.random():.4f}\t'
                        f"\tglobal loss {random.random():.4f}\t"
                        f"\tlocal loss {random.random():.4f}")
    
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'end_time \t {end_time}')