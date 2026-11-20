# xgboost_from_clean_csv.py

# 如果还没装必要库：
# pip install xgboost joblib pandas scikit-learn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import joblib

# ========================= 1. 读取清洗好的数据 =========================

# TODO：这里换成你的 CSV 文件路径
data_path = "data/loan_dataset_cleaned_encoded.csv"

df = pd.read_csv(data_path)
print("读取数据维度：", df.shape)
print(df.head())

# ========================= 2. 划分特征和标签 =========================

target_col = "loan_paid_back"

X = df.drop(columns=[target_col])
y = df[target_col]

print("特征维度：", X.shape)
print("目标分布：")
print(y.value_counts())

# ========================= 3. 划分训练集 / 测试集 =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n训练集维度：", X_train.shape)
print("测试集维度：", X_test.shape)

# ========================= 4. 训练 XGBoost 模型 =========================

sample_weight = np.where(y_train == 0, 3.0, 1.0)

xgb_clf = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",   # CPU 上较快
    random_state=42,
    n_jobs=-1,
)

xgb_clf.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,    # 想安静一点就改成 False
)

# ========================= 5. 模型评估 =========================
y_proba = xgb_clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_proba)
print("\n========== XGBoost 模型评估结果 ==========")
print("AUC:", auc)
print("\n分类报告：")
print(classification_report(y_test, y_pred))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# ========================= 6. 保存模型 & 保存预测结果 =========================

model_path = "results/xgb_trained_model.pkl"
joblib.dump(xgb_clf, model_path)
print(f"\n模型已保存到：{model_path}")

result_df = X_test.copy()
result_df["loan_paid_back_true"] = y_test.values
result_df["loan_paid_back_pred_proba"] = y_proba
result_df["loan_paid_back_pred"] = y_pred

result_path = "results/xgb_predictions_from_clean.csv"
result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

print(f"预测结果已保存到：{result_path}")
