<<<<<<< HEAD
import optuna, lightgbm as lgb
from sklearn.model_selection import train_test_split

def objective(trial, X, y):
    params = {
      "objective":"regression",
      "metric":"None",
      "learning_rate":trial.suggest_loguniform("lr", 1e-3, 1e-1),
      "num_leaves":trial.suggest_int("num_leaves", 31, 256),
      "feature_fraction":trial.suggest_uniform("feature_fraction", 0.5, 1.0),
      "bagging_fraction":trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
      "lambda_l1":trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
      "lambda_l2":trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
      "seed":42
    }
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval], feval=lambda p, d: ('smape', smape(d.get_label(), p), False),
                      num_boost_round=2000, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
    best = model.best_score['valid_0']['smape'] if model.best_iteration else None
    return best

=======
import optuna, lightgbm as lgb
from sklearn.model_selection import train_test_split

def objective(trial, X, y):
    params = {
      "objective":"regression",
      "metric":"None",
      "learning_rate":trial.suggest_loguniform("lr", 1e-3, 1e-1),
      "num_leaves":trial.suggest_int("num_leaves", 31, 256),
      "feature_fraction":trial.suggest_uniform("feature_fraction", 0.5, 1.0),
      "bagging_fraction":trial.suggest_uniform("bagging_fraction", 0.5, 1.0),
      "lambda_l1":trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
      "lambda_l2":trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
      "seed":42
    }
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    model = lgb.train(params, dtrain, valid_sets=[dval], feval=lambda p, d: ('smape', smape(d.get_label(), p), False),
                      num_boost_round=2000, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)])
    best = model.best_score['valid_0']['smape'] if model.best_iteration else None
    return best

>>>>>>> 18d577671530640494aa83c90e563dd2dd87cd75
