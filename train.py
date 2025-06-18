import os
import yaml
import math
import torch
import argparse
from torch.optim import Adam
from datetime import datetime
from src.log import get_logger
from src.loss import compute_loss
from torch.utils.data import DataLoader
from src.build_model import build_model
from dataset.LoveDA_dataset import LoveDADataset
from dataset.transform import get_train_transform
from transformers import get_cosine_schedule_with_warmup
from src.gaussian_diffusion import GaussianDiffusion
from src.resample import create_named_schedule_sampler


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = arg_parser()
    config = load_config(args.config)

    model = build_model(config, train_flag=True).to(device)
    gd = GaussianDiffusion()   # 扩散管理对象
    sampler_method="uniform"  # 对时间步的采样是均匀的
    schedule_sampler = create_named_schedule_sampler(sampler_method, gd)
    
    logger = get_logger()
        
    train_transform = get_train_transform()
    training_cfg = config.get("training", {})
    train_dataset = LoveDADataset(training_cfg['data_root_path'], split='train', transform=train_transform, train_flag=True)
    train_dataloader = DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True)
    
    optimizer = Adam(model.parameters(), lr = training_cfg['lr'], weight_decay=training_cfg['weight_decay'])
    total_steps = training_cfg["epochs"] * math.ceil(len(train_dataloader))
    warmup_steps = training_cfg["warmup_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    epochs = training_cfg['epochs']
    log_interval = training_cfg["log_interval"]
    save_interval = training_cfg["save_interval"]
    save_dir = training_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 训练前，设置为train模式
    model.train()
    # 将日期对象格式化为字符串
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'start_time \t {start_time}')

    global_step = 0
    for epoch in range(epochs):
        for i, (images, masks) in enumerate(train_dataloader):
            images, masks = images.to(device), masks.to(device)

            t, weights = schedule_sampler.sample(images.shape[0], device)
            noise = torch.randn_like(images)
            noise_image = gd.q_sample(images, t, noise)
            # 这里可以用 guassian_diffusion 类
            logits, eps_pred = model(images, noise_image, t)
            # 计算损失
            total_loss, loss_ce, loss_mse = compute_loss(logits, masks, eps_pred, noise)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            
            if (global_step) % log_interval == 0:
                logger.info(f'step {global_step:06d}\tloss {total_loss.item():.5f}  loss_ce {loss_ce.item():.5f}  loss_mse {loss_mse.item():.5f}')

        # 保存参数
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), save_path)
            
            logger.info(f"Model saved to {os.path.join(save_dir, f'epoch_{epoch+1}.pth')}")

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f'end_time \t {end_time}')


if __name__ == '__main__':
    # 开始训练
    main()


    