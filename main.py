"""Main execution script for used car price project."""
from pathlib import Path

from src.config import load_config, ensure_directories
from src.logger import get_logger
from src.data_loading import load_data
from src.data_cleaning import clean_vehicle_data, add_classification_target
from src.eda import run_eda
from src.preprocessing import split_features, build_preprocessor
from src.modeling import (
    train_regression_models, train_classification_models,
    tune_best_regressor, evaluate_classification, save_model
)
from src.reporting import (
    save_results_table, plot_actual_vs_predicted, plot_residuals,
    save_feature_importance, plot_confusion_matrix, write_executive_summary,
     plot_top_feature_importance, plot_permutation_importance
)


def main():
    config = load_config()
    ensure_directories(config)
    logger = get_logger(__name__, f"{config['outputs']['logs_dir']}/project.log")

    raw_path = config["data"]["raw_path"]
    logger.info("Loading data from %s", raw_path)
    df = load_data(raw_path)
    
    '''
    print("Price (1%):", df['price'].quantile(0.01))
    print("Price (99%):", df['price'].quantile(0.99))

    print("Year (1%):", df['year'].quantile(0.01))
    print("Year (99%):", df['year'].quantile(0.99))

    print("Odometer (99%):", df['odometer'].quantile(0.99))
    '''


    logger.info("Cleaning data")
    df = clean_vehicle_data(df, config)

    if config["data"].get("sample_for_development", False) and len(df) > config["data"]["sample_size"]:
        logger.info("Sampling %s rows for faster development run", config["data"]["sample_size"])
        df = df.sample(config["data"]["sample_size"], random_state=config["project"]["random_state"])

    df = add_classification_target(df, config)
    Path(config["data"]["processed_path"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config["data"]["processed_path"], index=False)

    logger.info("Running EDA and saving plots/tables")
    run_eda(df, config["outputs"]["plots_dir"], config["outputs"]["reports_dir"])

    logger.info("Preparing regression features")
    target = config["data"]["target"]
    cls_target = config["modeling"]["classification_target_name"]
    X_reg, y_reg = split_features(df, target=target, drop_cols=[cls_target])
    preprocessor_reg, num_features, cat_features = build_preprocessor(X_reg)

    logger.info("Training regression models")
    reg_results, best_reg_model, best_reg_name, X_train, X_test, y_train, y_test, preds = train_regression_models(
        X_reg, y_reg, preprocessor_reg, config
    )
    save_results_table(reg_results, f"{config['outputs']['reports_dir']}/regression_model_results.csv")
    save_model(best_reg_model, f"{config['outputs']['models_dir']}/best_regression_model.joblib")

    best_preds = preds[best_reg_name]
    plot_actual_vs_predicted(y_test, best_preds, f"{config['outputs']['plots_dir']}/actual_vs_predicted.png")
    plot_residuals(y_test, best_preds, f"{config['outputs']['plots_dir']}/residuals.png")
    save_feature_importance(best_reg_model, num_features, cat_features, f"{config['outputs']['reports_dir']}/feature_importance.csv")

    plot_top_feature_importance(
    f"{config['outputs']['reports_dir']}/feature_importance.csv",
    f"{config['outputs']['plots_dir']}/top_feature_importance.png",
    top_n=20)

    perm_df = plot_permutation_importance(
       best_reg_model,
       X_test,
       y_test,
       f"{config['outputs']['plots_dir']}/permutation_importance.png",
       top_n=15
)

    perm_df.to_csv(
       f"{config['outputs']['reports_dir']}/permutation_importance.csv",
       index=False
)

    if config["modeling"].get("run_hyperparameter_tuning", False):
        logger.info("Running hyperparameter tuning")
        search = tune_best_regressor(X_train, y_train, preprocessor_reg, config["project"]["random_state"], config["modeling"]["cv_folds"])
        save_model(search.best_estimator_, f"{config['outputs']['models_dir']}/tuned_best_regression_model.joblib")
        Path(f"{config['outputs']['reports_dir']}/best_hyperparameters.txt").write_text(str(search.best_params_), encoding="utf-8")

    logger.info("Preparing classification features")
    X_cls, y_cls = split_features(df, target=cls_target, drop_cols=[target])
    preprocessor_cls, _, _ = build_preprocessor(X_cls)

    logger.info("Training classification models")
    cls_results, best_cls_model, best_cls_name = train_classification_models(X_cls, y_cls, preprocessor_cls, config)
    save_results_table(cls_results, f"{config['outputs']['reports_dir']}/classification_model_results.csv")
    save_model(best_cls_model, f"{config['outputs']['models_dir']}/best_classification_model.joblib")

    # Test split for confusion matrix only
    from sklearn.model_selection import train_test_split
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=config["modeling"]["test_size"],
        random_state=config["project"]["random_state"], stratify=y_cls
    )
    best_cls_model.fit(Xc_train, yc_train)
    plot_confusion_matrix(best_cls_model, Xc_test, yc_test, f"{config['outputs']['plots_dir']}/confusion_matrix.png")

    y_pred_class = evaluate_classification(best_cls_model,Xc_test,yc_test)

    write_executive_summary(reg_results, cls_results, f"{config['outputs']['reports_dir']}/executive_summary.md")
    logger.info("Project completed successfully")


if __name__ == "__main__":
    main()
