CONFIG = {
    #ข้อมูล
    "data_path": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "features": ["Pclass", "Age", "SibSp", "Fare"],
    "target": "Survived",

    "test_size": 0.2,
    "random_state": 42,

    "model_type":   "logistic",
    "n_estimators": 200,
    "cv_folds":     7,
}