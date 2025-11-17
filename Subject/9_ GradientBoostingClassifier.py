import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# 1) Load training and test data
train = pd.read_excel("/Users/jhm/Documents/GitHub/MachineLearning/Data/lendingclub_traindata.xlsx")
test = pd.read_excel("/Users/jhm/Documents/GitHub/MachineLearning/Data/lendingclub_testdata.xlsx")

# 2) Features & Target 설정
features = ["home_ownership", "income", "dti", "fico"]
target = "loan_status"

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# 3) Gradient Boosting 모델 정의
model = GradientBoostingClassifier(
    n_estimators=100,
    random_state=0,
    max_depth=4,
    subsample=0.7
)

# 4) 학습
model.fit(X_train, y_train)

# 5) 예측 확률
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 6) AUC 계산
auc = roc_auc_score(y_test, y_pred_proba)
print("Gradient Boosting AUC:", auc)

# 7) ROC Curve 생성
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--', label="Baseline") 
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("GradientBoostingClassifier")

# 8) 이미지 저장
output_path = "/Users/jhm/Documents/GitHub/MachineLearning/Subject/GradientBoostingClassifier.png"  # 저장 경로 (Subject 폴더에 저장됨)

plt.savefig(output_path, dpi=300)

# Gradient Boosting AUC: 0.6433503199001851