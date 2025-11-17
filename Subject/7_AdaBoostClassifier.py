import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt

# 1) Load training and test data
train = pd.read_excel("/Users/jhm/Documents/GitHub/MachineLearning/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("/Users/jhm/Documents/GitHub/MachineLearning/Data/lendingclub_testdata.xlsx")

# 2) Feature / Target 설정
features = ["home_ownership", "income", "dti", "fico"]
target = "loan_status"

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# 3) AdaBoost Classifier 설정
model = AdaBoostClassifier(
    n_estimators=200,
    random_state=0
)

# 4) 모델 학습
model.fit(X_train, y_train)

# 5) 예측 확률
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 6) AUC 계산
auc = roc_auc_score(y_test, y_pred_proba)
print("AdaBoost AUC:", auc)

# AdaBoost AUC: 0.6423859361864753

# 7) ROC Curve 생성
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--', label="Baseline") 
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("AdaBoostClassifier Roc Curve")

# 8) 이미지 저장
output_path = "/Users/jhm/Documents/GitHub/MachineLearning/Subject/AdaBoostClassifier RocCurve.png"  # 저장 경로 (Subject 폴더에 저장됨)
plt.savefig(output_path, dpi=300)

# AdaBoost AUC: 0.6423859361864753