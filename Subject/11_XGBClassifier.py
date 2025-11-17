import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb

# 1) Load training and test data
train = pd.read_excel("/Users/jhm/Documents/GitHub/MachineLearning/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("/Users/jhm/Documents/GitHub/MachineLearning/Data/lendingclub_testdata.xlsx")

# Feature / Target
features = ["home_ownership", "income", "dti", "fico"]
target = "loan_status"

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# XGBClassifier
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=0,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predict
y_pred_proba = model.predict_proba(X_test)[:, 1]

# AUC
auc = roc_auc_score(y_test, y_pred_proba)
print("XGBoost AUC:", auc)

#  ROC Curve 생성
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--', label="Baseline")

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("XGBClassifier")

output_path = "/Users/jhm/Documents/GitHub/MachineLearning/Subject/XGBClassifier.png"

plt.savefig(output_path, dpi=300)

# XGBoost AUC: 0.6316314389090669