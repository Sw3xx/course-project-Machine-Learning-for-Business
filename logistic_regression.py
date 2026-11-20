# LogisticRegression_model.py
# 功能：
# 1. 从清洗后的 CSV 读取数据（你之前生成的 loan_dataset_cleaned_encoded.csv）
# 2. 训练逻辑回归模型（Logistic Regression）
# 3. 输出 AUC、分类报告、混淆矩阵
# 4. 保存模型和预测结果

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ========================= 1. 读取清洗后数据 =========================

# TODO：改成你自己的 csv 路径
data_path = "data/loan_dataset_cleaned_encoded.csv"

df = pd.read_csv(data_path)
print("读取数据维度：", df.shape)
print(df.head())

target_col = "loan_paid_back"

X = df.drop(columns=[target_col])
y = df[target_col]

print("特征维度：", X.shape)
print("目标分布：")
print(y.value_counts())

# ========================= 2. 划分训练集 / 测试集 =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # 保持 0/1 比例
)

print("训练集维度：", X_train.shape)
print("测试集维度：", X_test.shape)

# ========================= 3. 定义逻辑回归模型（带标准化） =========================
# 标准化对逻辑回归有帮助（数值特征尺度差异会被消除）

log_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=1000,
        solver="lbfgs",   # 默认即可，适合中等维度
        n_jobs=-1
    ))
])

# 训练
log_clf.fit(X_train, y_train)

# ========================= 4. 模型评估（阈值 = 0.5） =========================

# predict_proba 得到预测“还款=1”的概率
y_proba = log_clf.predict_proba(X_test)[:, 1]

threshold = 0.5
y_pred = (y_proba >= threshold).astype(int)

auc = roc_auc_score(y_test, y_proba)

print("\n========== 逻辑回归模型评估结果 ==========")
print(f"AUC: {auc}")
print("\n分类报告：")
print(classification_report(y_test, y_pred, digits=4))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# ========================= 5. 保存模型和预测结果 =========================

model_path = "results/logistic_model.pkl"
joblib.dump(log_clf, model_path)
print(f"\n逻辑回归模型已保存到：{model_path}")

result_df = X_test.copy()
result_df["loan_paid_back_true"] = y_test.values
result_df["loan_paid_back_pred_proba"] = y_proba
result_df["loan_paid_back_pred"] = y_pred

result_path = "results/logistic_predictions.csv"
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
print(f"预测结果已保存到：{result_path}")
