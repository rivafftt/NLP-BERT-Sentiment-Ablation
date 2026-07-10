import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 1. 基础配置
DATA_DIR = os.path.dirname(__file__)
train_path = os.path.join(DATA_DIR, 'train.tsv')
dev_path = os.path.join(DATA_DIR, 'dev.tsv')

# 检测 GPU (大模型微调最好有 GPU，如果没有，代码会自动在 CPU 上跑，但会比较慢)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前训练设备: {device}")

# 2. 数据加载与规整
print("正在加载并处理 BERT 数据集...")
train_df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(dev_path, sep='\t')

train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

train_df = train_df.rename(columns={'# label': 'label'})
test_df = test_df.rename(columns={'# label': 'label'})


# 3. 构建 PyTorch Dataset
class CommentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['text_a'].astype(str).tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 加载中文 BERT Tokenizer (会自动联网下载，请确保网络通畅)
print("正在下载/加载 bert-base-chinese 分词器...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_dataset = CommentDataset(train_df, tokenizer)
test_dataset = CommentDataset(test_df, tokenizer)

# 如果是用 CPU 跑，把 batch_size 调小一点（比如 4 或 8），如果是 GPU 可以设为 16 或 32
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 4. 初始化 BERT 模型
print("正在初始化 BERT 情感分类模型...")
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
model.to(device)

# 5. 设置优化器与超参数 [cite: 29]
optimizer = AdamW(model.parameters(), lr=2e-5)
EPOCHS = 3  # 预训练模型通常微调 3 个 Epoch 即可收敛

# 6. 开始训练
print("开始训练微调 BERT...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

# 7. 测试集评估
print("\n正在测试集上评估 BERT 模型...")
model.eval()
predictions = []
real_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        real_labels.extend(labels.cpu().numpy())

# 输出结果
print("\n" + "=" * 20 + " BERT 测试集结果 " + "=" * 20)
print(f"Accuracy: {accuracy_score(real_labels, predictions):.4f}")
print(classification_report(real_labels, predictions, digits=4))

# 保存 BERT 预测结果以便对比分析 [cite: 34, 35]
test_df['bert_pred'] = predictions
test_df.to_csv(os.path.join(DATA_DIR, 'bert_predictions.csv'), index=False, encoding='utf-8-sig')
print("[提示] 预测结果已保存至 bert_predictions.csv")