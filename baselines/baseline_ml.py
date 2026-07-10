import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. 路径设置
DATA_DIR = os.path.dirname(__file__)
train_path = os.path.join(DATA_DIR, 'train.tsv')
dev_path = os.path.join(DATA_DIR, 'dev.tsv')

print("正在加载数据...")
# 精准适配 Kaggle 上的真实表头
train_df = pd.read_csv(train_path, sep='\t')
test_df = pd.read_csv(dev_path, sep='\t')

# 清洗可能存在的列名空格（防止表头有隐藏空格）
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns = [c.strip() for c in test_df.columns]

# 统一映射到规范列名，避免 '# label' 的符号引发读取问题
train_df = train_df.rename(columns={'# label': 'label'})
test_df = test_df.rename(columns={'# label': 'label'})

print(f" 数据集加载成功：")
print(f" - 训练集规模: {len(train_df)} 条")
print(f" - 测试集(原dev)规模: {len(test_df)} 条")

# 2. 中文分词
def chinese_tokenizer(text):
    if not isinstance(text, str):
        return ""
    return " ".join(jieba.cut(text))

print("\n正在进行中文分词（请稍候...）...")
train_df['text_cut'] = train_df['text_a'].apply(chinese_tokenizer)
test_df['text_cut'] = test_df['text_a'].apply(chinese_tokenizer)

# 3. 特征工程：TF-IDF 向量化
print("正在提取 TF-IDF 特征...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text_cut'])
X_test = vectorizer.transform(test_df['text_cut'])

y_train = train_df['label']
y_test = test_df['label']

# 4. 模型训练：逻辑回归
print("正在训练逻辑回归模型...")
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# 5. 在测试集上评估
print("\n" + "="*20 + " 测试集结果 (Test Set) " + "="*20)
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Accuracy (准确率): {test_acc:.4f}")
print(classification_report(y_test, y_test_pred, digits=4))

# 6. 保存预测错误的结果
test_df['pred'] = y_test_pred
errors = test_df[test_df['label'] != test_df['pred']]
errors.to_csv(os.path.join(DATA_DIR, 'ml_error_cases.csv'), index=False, encoding='utf-8-sig')
print(f"\n[提示] 运行成功！已将测试集预测错误的样本保存至 ml_error_cases.csv")