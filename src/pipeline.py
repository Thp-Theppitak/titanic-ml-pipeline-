import sys
import os 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG


class TitanicPipeline:
    """End-to-end Titanic ML pipeline."""

    FEATURES = CONFIG["features"]
    TARGET = CONFIG["target"]

    def __init__(self, model_type: str = 'forest', n_estimators: int = 200):
        self.model_type = model_type
        self.is_fitted = False 
        self.metrics_ = {}

        if model_type == 'forest': 
            self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        elif model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _preprocess(self, df):
        df = df.copy()
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        return df[self.FEATURES]
    
    def fit(self, df):
        X = self._preprocess(df)
        Y = df[self.TARGET]
        self.model.fit(X, Y)
        self.is_fitted = True

        cv = cross_val_score(self.model, X, Y, cv=5)
        self.metrics_ = {
            'cv_mean': round(cv.mean(), 4),
            'cv_std': round(cv.std(), 4)

        }
        return self # คืน self เพื่อ chain ได้
    
    def predict(self, df):
        if not self.is_fitted:
            raise ValueError("Call fit() before predict().")
        X = self._preprocess(df)
        return self.model.predict(X).tolist()

    def summary(self):
        print(f"Model  : {self.model_type}")
        print(f"Fitted : {self.is_fitted}")
        if self.metrics_:
            print(f"CV     : {self.metrics_['cv_mean']:.3f} ± {self.metrics_['cv_std']:.3f}")

    def get_feature_importances(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before get_feature_importances().")
    
    # Logistic Regression ไม่มี feature_importances_
        if self.model_type != "forest":
            print(f"Feature importance ใช้ได้แค่กับ Random Forest ครับ")
            print(f"Model ปัจจุบัน: {self.model_type}")    
            return {}
        importances = self.model.feature_importances_
        result = dict(zip(self.FEATURES, importances))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    def __repr__(self):
        state = "fitted" if self.is_fitted else "not fitted"
        return f"TitanicPipeline(state={state}, n_estimators={self.n_estimators})"
    
    def get_feature_importance_plot(self):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before get_feature_importance_plot().")
        
        # จับคู่ชื่อ feature กับตัวเลข importance
        importances = self.get_feature_importances()
        result = dict(zip(self.FEATURES, importances))
        result = self.get_feature_importances()
        if not result:
            return

        # เรียงจากมากไปน้อย
        result  = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

        # วาดกราฟ
        plt.figure(figsize=(8, 6))
        plt.barh(list(result.keys()), list(result.values()), color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Plot')
        plt.show()

        return result
    
    @classmethod
    def compare_models(cls, df, model_types=None):
        """"เปรียบเทียบหลาย model แล้วคืน ranked results"""
        model_types = model_types or ["forest", "logistic"]
        pipelines = [cls(model_type=m).fit(df) for m in model_types]
        results = {
            p.model_type: {
                "cv_mean": p.metrics_["cv_mean"],
                "cv_std": p.metrics_["cv_std"]
            }
            for p in pipelines
        }
        ranked = sorted(results.items(),
                        key=lambda x: x[1]["cv_mean"],
                        reverse=True)
        print("Model Comparison:")
        for rank, (name, metrics) in enumerate(ranked, 1):
            print(f"#{rank} {name}: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
        best = ranked[0][0]
        print(f"\nBest model: {best}")
        return dict(ranked)

# ───────────────────────────────────────────
# ส่วนที่รันจริง — เขียนไว้ด้านล่าง class เสมอ
# ───────────────────────────────────────────

# 1. โหลด dataset จาก URL (ไม่ต้องดาวน์โหลดไฟล์)
url = CONFIG["data_path"]
df = pd.read_csv(url)

# 2. แบ่ง train/test
train_df, test_df = train_test_split(df, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"])

# 3. สร้าง pipeline และ train
pipeline = TitanicPipeline(n_estimators=CONFIG["n_estimators"], model_type=CONFIG["model_type"])
pipeline.fit(train_df)
results = TitanicPipeline.compare_models(train_df)

# 4. สรุปผล
pipeline.summary()
pipeline.get_feature_importance_plot()

print("\nFeature Importances:")
print(pipeline.get_feature_importances())

# 5. ทำนายผล
predictions = pipeline.predict(test_df)
print(f"\nตัวอย่าง predictions 100 ตัวแรก: {predictions[:100]}")

