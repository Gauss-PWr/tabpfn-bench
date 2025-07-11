import numpy as np
from hyperopt import hp

xgb_params = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-7), np.log(1)),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "subsample": hp.uniform("subsample", 0.2, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.2, 1.0),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0.2, 1.0),
    "min_child_weight": hp.loguniform(
        "min_child_weight", np.log(1e-16), np.log(np.exp(5))
    ),
    "alpha": hp.loguniform("alpha", np.log(1e-16), np.log(np.exp(2))),
    "lambda": hp.loguniform("lambda", np.log(1e-16), np.log(np.exp(2))),
    "gamma": hp.loguniform("gamma", np.log(1e-16), np.log(np.exp(2))),
    "n_estimators": hp.quniform("n_estimators", 100, 4000, 1),
}
lgbm_params = {
    "num_leaves": hp.quniform("num_leaves", 5, 50, 1),
    "max_depth": hp.quniform("max_depth", 3, 20, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(1)),
    "n_estimators": hp.quniform("n_estimators", 50, 2000, 1),
    "min_child_weight": hp.choice(
        "min_child_weight", [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    ),
    "subsample": hp.uniform("subsample", 0.2, 0.8),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.2, 0.8),
    "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
    "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 5, 10, 20, 50, 100]),
}
catboost_params = {
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1)),
    "random_strength": hp.quniform("random_strength", 1, 20, 1),
    "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(1), np.log(10)),
    "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0),
    "leaf_estimation_iterations": hp.quniform("leaf_estimation_iterations", 1, 20, 1),
    "iterations": hp.quniform("iterations", 100, 4000, 1),
}
tabpfn_params = {
    "n_estimators": hp.choice("n_estimators", [4, 8, 16, 32]),
    "softmax_temperature": hp.uniform("softmax_temperature", 0.75, 1.0),
    "average_before_softmax": hp.choice("average_before_softmax", [False, True]),
    "fit_mode": hp.choice(
        "fit_mode", ["low_memory", "fit_preprocessors", "fit_with_cache"]
    ),
    "memory_saving_mode": hp.choice("memory_saving_mode", [True, False, "auto"]),
    "random_state": hp.randint("random_state", 0, 100),
    "n_jobs": hp.choice("n_jobs", [1, 4, -1]),
}

# w paperze jakies jeszcze gowno pisali trza zobaczyc
