import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# 1. 基础配置
# 注意：使用 __file__ 动态定位，向上寻找根目录下的 train.tsv 和 dev.tsv
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(DATA_DIR, 'train.tsv')
dev_path = os.path.join(DATA_DIR, 'dev.tsv')

# 自动检测并切换到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"【消融实验一：BERT参数冻结】当前训练设备: {device}")
if torch.cuda.is_available():
    print(f"当前激活显卡: {torch.cuda.get_device_name(0)}")

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
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = str(self.df.iloc[index]['text_a'])
        label = int(self.df.iloc[index]['label'])

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

# 4. 初始化分词器与模型
print("正在加载 bert-base-chinese 分词器...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

print("正在初始化 BERT 情感分类模型（带预训练权重）...")
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# ==================== 【消融实验核心改动：冻结BERT主体】 ====================
print(">>> [消融动作] 正在锁死 BERT 骨干网络参数，仅允许分类头进行训练...")
for param in model.bert.parameters():
    param.requires_grad = False
# ============================================================================

model.to(device)

# 5. 准备 DataLoader
train_dataset = CommentDataset(train_df, tokenizer)
test_dataset = CommentDataset(test_df, tokenizer)

# 1. batch_size 设为 64
# 2. Windows 系统下 num_workers > 0 极易引发死锁报错，稳妥起见设为 0
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 6. 优化器配置
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
EPOCHS = 3

# 开始训练
print("开始训练微调 BERT（消融对照组：参数冻结）...")
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

# 7. 测试集评估与结果保存
print("\n正在测试集上评估【BERT冻结版】模型...")
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

print("\n" + "=" * 20 + " BERT 冻结消融实验测试集结果 " + "=" * 20)
print(f"Accuracy: {accuracy_score(real_labels, predictions):.4f}")
print(classification_report(real_labels, predictions, digits=4))

# 将预测结果单独归档到输出文件夹，防止覆盖完全体预测
output_dir = os.path.join(DATA_DIR, 'outputs_and_analysis')
os.makedirs(output_dir, exist_ok=True)
test_df['bert_frozen_pred'] = predictions
output_path = os.path.join(output_dir, 'bert_frozen_predictions.csv')
test_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"[提示] 消融预测结果已保存至 {output_path}\n")