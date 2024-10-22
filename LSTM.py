import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import KFold
import copy

# 超参数
input_size = 6  # x, y, z, vx, vy, vz
hidden_size = 128  # 增加隐藏单元数
output_size = 3  # 只预测 x, y, z
num_layers = 3  # 增加LSTM层数
learning_rate = 0.01
n_steps = 10  # 时间步长
batch_size = 64
num_epochs = 100
dropout = 0.5
n_splits = 5  # K折交叉验证的折数

# 创建带双向LSTM和注意力机制的模型类
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # 双向LSTM，hidden_size * 2

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size * 2]
        scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len, 1]
        context = torch.sum(weights * lstm_output, dim=1)  # [batch_size, hidden_size * 2]
        return context

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                            dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM，hidden_size * 2

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 双向LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: [batch_size, seq_len, hidden_size * 2]
        attn_out = self.attention(lstm_out)  # [batch_size, hidden_size * 2]
        out = self.fc(attn_out)  # [batch_size, output_size]
        return out

# 数据处理函数：从多个投篮文件加载数据并预处理
def load_and_process_data(data_dir, n_steps):
    X, y = [], []
    scaler = StandardScaler()  # 创建标准化对象

    # 遍历目录中的所有投篮数据文件
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            # 加载每次投篮数据
            data = pd.read_csv(os.path.join(data_dir, file_name))
            features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            data_values = data[features].values

            # 数据清洗：处理缺失值（插值）
            data = pd.DataFrame(data_values, columns=features).interpolate(method='linear').dropna().values

            # 对每个投篮单独进行标准化
            data_scaled = scaler.fit_transform(data)

            # 生成序列数据
            X_shot, y_shot = create_sequences(data_scaled, n_steps, augment=True)
            X.append(X_shot)
            y.append(y_shot)

    # 将所有投篮的数据合并
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, scaler  # 返回scaler

# 数据增强函数
def augment_data(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# 将数据分割为时间序列格式 (n_steps) 作为 LSTM 的输入
def create_sequences(data, n_steps, augment=False):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x = data[i:i + n_steps]
        if augment:
            seq_x = augment_data(seq_x)
        seq_y = data[i + n_steps]
        X.append(seq_x)
        y.append(seq_y[:3])  # 仅预测位置 (x, y, z)
    return np.array(X), np.array(y)

# 预测轨迹函数
def predict_trajectory(model, X_tensor, scaler, n_future_steps=50):
    model.eval()

    # 准备一个初始的输入序列
    test_input = X_tensor[0].unsqueeze(0).to(device)

    predicted_trajectory = []

    # 使用模型预测未来轨迹
    for _ in range(n_future_steps):  # 预测未来 n_future_steps 个时间步
        with torch.no_grad():
            pred = model(test_input)
            predicted_trajectory.append(pred.cpu().numpy())

        # 构造新的输入，保留速度 vx, vy, vz 不变
        new_position_input = torch.cat((test_input[:, 1:, :3], pred.unsqueeze(1)), dim=1)  # 更新位置信息
        new_velocity_input = test_input[:, 1:, 3:]  # 保留速度信息
        new_velocity_input = torch.cat((new_velocity_input, new_velocity_input[:, -1:, :]), dim=1)  # 保证时间步一致
        new_input = torch.cat((new_position_input, new_velocity_input), dim=2)

        test_input = new_input

    # 预测的轨迹数据
    predicted_trajectory = np.array(predicted_trajectory).reshape(-1, 3)

    # 将预测结果转换为原始坐标系
    predicted_trajectory = scaler.inverse_transform(np.hstack([predicted_trajectory, np.zeros((predicted_trajectory.shape[0], 3))]))[:, :3]

    return predicted_trajectory

# 加载并处理前十个csv文件并进行预测
def predict_from_multiple_csv(data_dir, model, scaler, n_steps, n_future_steps=50, num_files=10):
    # 统计已经处理的文件数量
    processed_files = 0

    # 遍历目录中的所有投篮数据文件，只处理前 num_files 个文件
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            # 如果已经处理了 num_files 个文件，停止处理
            if processed_files >= num_files:
                break

            # 加载和处理当前投篮数据
            data_path = os.path.join(data_dir, file_name)
            data = pd.read_csv(data_path)
            features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            data_values = data[features].values

            # 数据清洗
            data_clean = pd.DataFrame(data_values, columns=features).interpolate(method='linear').dropna().values

            # 标准化数据
            data_scaled = scaler.transform(data_clean)  # 使用相同的scaler进行标准化

            # 生成序列数据
            X_shot, y_shot = create_sequences(data_scaled, n_steps)
            X_tensor = torch.tensor(X_shot, dtype=torch.float32)

            # 预测轨迹
            predicted_trajectory = predict_trajectory(model, X_tensor, scaler, n_future_steps=n_future_steps)

            # 真实轨迹数据，用于对比（与预测长度一致）
            true_trajectory = data_clean[n_steps:n_steps + n_future_steps, :3]

            # 可视化真实轨迹与预测结果
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            # 绘制真实轨迹
            ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], true_trajectory[:, 2], label='True Trajectory', color='b')

            # 绘制预测轨迹
            ax.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], label='Predicted Trajectory', color='r', linestyle='--')

            # 添加图例和标题
            plt.title(f'Comparison of True and Predicted Trajectory for {file_name}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.legend()
            plt.show()

            # 增加已处理文件计数
            processed_files += 1

# 评价模型性能
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    mse = total_loss / len(dataloader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return mse, mae, r2

# 主程序
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据目录
    data_dir = '/root/shots_data'  # 投篮数据文件夹路径

    # 加载并处理数据集
    X_tensor, y_tensor, scaler = load_and_process_data(data_dir, n_steps)

    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)

    # K折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []

    for train_index, val_index in kf.split(dataset):
        print(f'--- Fold {fold} ---')
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 创建模型
        model = BiLSTMWithAttention(input_size, hidden_size, output_size, num_layers, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 添加权重衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        scaler_amp = GradScaler()  # 混合精度训练

        # 训练循环
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        patience = 10
        counter = 0

        for epoch in range(num_epochs):
            model.train()
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()

            # 调度器步进
            scheduler.step()

            # 评估验证集
            val_loss, val_mae, val_r2 = evaluate_model(model, val_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')

            # 早停判断
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered")
                    break

        # 加载最佳模型权重
        model.load_state_dict(best_model_wts)

        # 在验证集上评估
        val_mse, val_mae, val_r2 = evaluate_model(model, val_loader)
        print(f'Fold {fold} - Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}')

        results.append({'Fold': fold, 'MSE': val_mse, 'MAE': val_mae, 'R²': val_r2})

        fold += 1

    # 打印交叉验证结果
    print('--- Cross-Validation Results ---')
    for res in results:
        print(f"Fold {res['Fold']}: MSE={res['MSE']:.4f}, MAE={res['MAE']:.4f}, R²={res['R²']:.4f}")

    # 训练完成后，可以选择训练一个最终模型使用所有数据
    print('--- Training Final Model on All Data ---')
    # 使用所有数据进行训练
    final_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建最终模型
    final_model = BiLSTMWithAttention(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scaler_amp = GradScaler()

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(final_model.state_dict())
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        final_model.train()
        for inputs, targets in final_train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = final_model(inputs)
                loss = criterion(outputs, targets)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

        # 调度器步进
        scheduler.step()

        # 评估整个数据集（这里仅用于示例，建议使用单独的验证集）
        train_loss, train_mae, train_r2 = evaluate_model(final_model, final_train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}')

        # 早停判断
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_wts = copy.deepcopy(final_model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # 加载最佳模型权重
    final_model.load_state_dict(best_model_wts)

    # 预测并可视化
    predict_from_multiple_csv(data_dir, final_model, scaler, n_steps, n_future_steps=10,num_files=10)  # 可以调整 n_future_steps 预测更多时间步