import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルからデータを読み込む
original_csv_path = 'csv/base10.csv'
modified_csv_path = 'csv/12_1000.csv'
path3 = 'csv/12.csv'
path4 = 'csv/1000.csv'

original_data = pd.read_csv(original_csv_path)
modified_data = pd.read_csv(modified_csv_path)
data3 = pd.read_csv(path3)
data4 = pd.read_csv(path4)

# エポックごとの損失と精度のデータを取得
original_train_loss_data = original_data[original_data['batch'].notna()]
modified_train_loss_data = modified_data[modified_data['batch'].notna()]
loss_data3 = data3[data3['batch'].notna()]
loss_data4 = data4[data4['batch'].notna()]

original_val_data = original_data[original_data['batch'].isna()]
modified_val_data = modified_data[modified_data['batch'].isna()]
val_data3 = data3[data3['batch'].isna()]
val_data4 = data4[data4['batch'].isna()]

# トレーニング損失をフィルタリング
original_train_loss = original_train_loss_data['loss'].values
original_batches = original_train_loss_data['batch'].values
original_epochs = original_train_loss_data['epoch'].values

modified_train_loss = modified_train_loss_data['loss'].values
modified_batches = modified_train_loss_data['batch'].values
modified_epochs = modified_train_loss_data['epoch'].values

loss3 = loss_data3['loss'].values
batches3 = loss_data3['batch'].values
epochs3 = loss_data3['epoch'].values

loss4 = loss_data4['loss'].values
batches4 = loss_data4['batch'].values
epochs4 = loss_data4['epoch'].values

# 検証精度を取得
original_val_accuracy = original_val_data['accuracy'].values
original_val_epochs = original_val_data['epoch'].values

modified_val_accuracy = modified_val_data['accuracy'].values
modified_val_epochs = modified_val_data['epoch'].values

accuracy3 = val_data3['accuracy'].values
val_epochs3 = val_data3['epoch'].values

accuracy4 = val_data4['accuracy'].values
val_epochs4 = val_data4['epoch'].values

# エポックとバッチの計算
original_epoch_batch = original_epochs + (original_batches / 12000.0)
modified_epoch_batch = modified_epochs + (modified_batches / 12000.0)
epoch_batch3 = epochs3 + (batches3 / 12000.0)
epoch_batch4 = epochs4 + (batches4 / 12000.0)

# トレーニング損失のプロット
fig, ax1 = plt.subplots(figsize=(12, 6))

# 元のプロット
ax1.plot(original_epoch_batch, original_train_loss, label='Original', color='blue', linestyle='dashed')

# 変更後のプロット
ax1.plot(modified_epoch_batch, modified_train_loss, label='ch12_output1000', color='red', linestyle='dashed')

ax1.plot(epoch_batch3, loss3, label='ch12', color='green', linestyle='dashed')
ax1.plot(epoch_batch4, loss4, label='fc1_output1000', color='grey')

# プロット設定
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_xticks(range(1, int(max(max(original_epochs), max(modified_epochs))) + 1))  # 横軸に1ずつの目盛りを設定
ax1.legend()
ax1.grid(True)
ax1.set_title('Training Loss')

fig.tight_layout(rect=[0, 0, 1, 0.95])  # 余白を調整
plt.show()

# 検証精度のプロット
fig, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(original_val_epochs, original_val_accuracy, label='Original', color='blue', linestyle='dashed')
ax2.plot(modified_val_epochs, modified_val_accuracy, label='ch12_output1000', color='red', linestyle='dashed')
ax2.plot(val_epochs3, accuracy3, label='ch12', color='green', linestyle='dashed')
ax2.plot(val_epochs4, accuracy4, label='fc1_output1000', color='grey')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_xticks(range(1, int(max(max(original_val_epochs), max(modified_val_epochs))) + 1))  # 横軸に1ずつの目盛りを設定
ax2.legend(loc='lower right')
ax2.grid(True)

fig.tight_layout(rect=[0, 0, 1, 1])  # 余白を調整
plt.show()