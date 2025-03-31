import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from fault_detection.models.FaultNet import FaultNet
from loss import torque_loss
from metadata import METADATA
from models.lightweight_evaluation import all_metrics


# 训练模型的函数
def train_model(model, train_loader, test_loader, epochs=int(2e3), learning_rate=1e-4,
                save_path="checkpoints/best_model.pth", loss_func=None):
    torch.save(model.state_dict(), "checkpoints/best_model.pth")

    import time
    time.sleep(50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training model on {}".format(device))
    model = model.to(device)
    criterion = nn.MSELoss()
    weight = 0.1

    if loss_func == "MSE":
        criterion = nn.MSELoss()
    elif loss_func == "MAE":
        criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.95)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.95)

    best_test_loss = float('inf')  # 初始化最佳测试损失为无穷大

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            # print(targets[0], outputs[0])

            # 计算损失
            if loss_func == "torque_loss":
                loss = torque_loss(outputs, targets, inputs)
            # elif loss_func == "equation_loss":
            elif loss_func == "combined_loss":
                loss = torque_loss(outputs, targets, inputs) + weight * criterion(outputs, targets)
            else:
                # loss = 10 * criterion(outputs[:, :, :3], targets[:, :, :3]) + criterion(outputs[:, :, -3:], targets[:, :, -3:])
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # 在测试集上评估模型
        model.eval()  # 评估模式
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                if loss_func == "torque_loss":
                    loss = torque_loss(outputs, targets, inputs)
                elif loss_func == "combined_loss":
                    loss = torque_loss(outputs, targets, inputs) + weight * criterion(outputs, targets)
                else:
                    # loss = 10 * criterion(outputs[:, :, :3], targets[:, :, :3]) + criterion(outputs[:, :, -3:], targets[:, :, -3:])
                    loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.20f}, Test Loss: {test_loss:.20f}')
        scheduler.step()

        # 如果当前的测试损失小于之前的最佳测试损失，则保存模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved the best model with test loss: {best_test_loss:.20f}")

        torch.save(model.state_dict(), "checkpoints/last_model.pth")


def load_and_validate_model(model, model_path, test_loader, de_norm=True):
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()  # 评估模式

    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    test_loss1 = 0.0
    test_loss2 = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)

            # 生成模型输出
            outputs = model(inputs)
            print(outputs.shape)

            if de_norm:
                targets[:, :, :3] = targets[:, :, :3] * (
                        METADATA['eloss_range'][1] - METADATA['eloss_range'][0]) + METADATA['eloss_range'][0]
                targets[:, :, -3:] = targets[:, :, -3:] * (
                        METADATA['bias_range'][1] - METADATA['bias_range'][0]) + METADATA['bias_range'][0]
                outputs[:, :, :3] = outputs[:, :, :3] * (
                        METADATA['eloss_range'][1] - METADATA['eloss_range'][0]) + METADATA['eloss_range'][0]
                outputs[:, :, -3:] = outputs[:, :, -3:] * (
                        METADATA['bias_range'][1] - METADATA['bias_range'][0]) + METADATA['bias_range'][0]

            loss1 = criterion1(outputs, targets)
            loss2 = criterion2(outputs, targets)
            test_loss1 += loss1.item() * inputs.size(0)
            test_loss2 += loss2.item() * inputs.size(0)

            # 打印每个目标和输出（格式化输出）
            if batch_idx <= 1:  # 只打印前1个batch的目标和输出
                for i in range(inputs.size(0)):
                    # 使用 numpy 的 flatten() 展平数组，并格式化每个数字的输出
                    target_flattened = targets[i].cpu().numpy().flatten()
                    output_flattened = outputs[i].cpu().numpy().flatten()
                    diff = target_flattened - output_flattened

                    # 格式化输出为整齐的数字，保留两位小数
                    target_str = " ".join([f"{t:.4f}" for t in target_flattened])
                    output_str = " ".join([f"{o:.4f}" for o in output_flattened])
                    diff_str = " ".join([f"{o:.4f}" for o in diff])

                    print(f"Sample {i + 1} - Target: [{target_str}], Output: [{output_str}], Diff: [{diff_str}], MSE: [{np.mean(diff ** 2)}]")

    test_loss1 /= len(test_loader.dataset)
    test_loss2 /= len(test_loader.dataset)
    print(f'MSE Loss (after loading best model): {test_loss1:.3e}')
    print(f'MAE Loss (after loading best model): {test_loss2:.3e}')

    all_metrics(model)
