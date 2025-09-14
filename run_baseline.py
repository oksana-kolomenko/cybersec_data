from csv_saver import save_results_to_csv
from data_preps import load_features, load_labels
from helpers import (hgbc_rte, hgbc, logistic_regression, lr_rte, concat_hgbc_rte, concat_lr_rte)
from values import DatasetName


def run_models_on_table_data():
    # === LUNGDISEASE ===
    """
    dataset = DatasetName.LUNG_DISEASE.value
    y = load_labels("data/y_lung_disease_data.csv")#, n_samples=100)
    X = load_features("data/X_lung_disease_data.csv")#, n_samples=100)
    X_metr = load_features("data/X_lungdisease_metrics.csv")#, n_samples=100)

    nominal_features = [
        'Gender',
        'Smoking Status',
        'Disease Type',
        'Treatment Type'
    ]"""

    # === CYBERSECURITY ===
    dataset = DatasetName.CYBERSECURITY.value
    y = load_labels("data/y_cybersecurity_intrusion_data.csv")  # , n_samples=51)
    X = load_features("data/X_cybersecurity_intrusion_data.csv")  # ), n_samples=51)
    X_metr = load_features("data/X_cybersecurity_metrics.csv")  # , n_samples=51)

    nominal_features = [
        'encryption_used',
        'browser_type',
        'protocol_type',
        'unusual_time_access'
    ]

    # 1. logistic regression
    (log_reg_dataset, log_reg_ml_method, log_reg_emb_method, log_reg_conc, log_reg_best_params,
     log_reg_pca_comp, log_reg_train_score, log_reg_test_scores) = \
        logistic_regression(dataset_name=dataset,
                            nominal_features=nominal_features,
                            pca=None,
                            X=X, y=y)

    save_results_to_csv(
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        emb_method=log_reg_emb_method,
        pca_n_comp=log_reg_pca_comp,
        best_params=log_reg_best_params,
        concatenation=log_reg_conc,
        is_train=True,
        metrics=log_reg_train_score,
        output_file=f"train/baseline/{dataset}_log_reg_train.csv")

    save_results_to_csv(
        dataset_name=log_reg_dataset,
        ml_method=log_reg_ml_method,
        emb_method=log_reg_emb_method,
        pca_n_comp=log_reg_pca_comp,
        best_params=log_reg_best_params,
        concatenation=log_reg_conc,
        is_train=False,
        metrics=log_reg_test_scores,
        output_file=f"test/baseline/{dataset}_log_reg_test.csv")

    # 2. log reg + random trees embedding
    (lr_rt_dataset, lr_rt_ml_method, lr_rt_emb_method, lr_rt_concatenation, lr_rte_best_params, lr_rte_pca,
     lr_rte_train_score, lr_rte_test_scores) = \
        lr_rte(dataset_name=dataset,
               nominal_features=nominal_features,
               pca=False,
               X=X, y=y)

    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=True,
        metrics=lr_rte_train_score,
        ml_method=lr_rt_ml_method,
        best_params=lr_rte_best_params,
        pca_n_comp=lr_rte_pca,
        output_file=f"train/baseline/{dataset}_lr_rte_train.csv")

    save_results_to_csv(
        dataset_name=lr_rt_dataset,
        concatenation=lr_rt_concatenation,
        emb_method=lr_rt_emb_method,
        is_train=False,
        metrics=lr_rte_test_scores,
        ml_method=lr_rt_ml_method,
        best_params=lr_rte_best_params,
        pca_n_comp=lr_rte_pca,
        output_file=f"test/baseline/{dataset}_lr_rte_test.csv")

    # 3. hgbc (no embedding)
    hgbc_dataset, hgbc_ml_method, hgbc_emb_method, conc, hgbc_best_params, hgbc_train_score, hgbc_test_scores = \
        hgbc(dataset_name=dataset, X=X, y=y, nominal_features=nominal_features, pca=None)

    save_results_to_csv(
        dataset_name=hgbc_dataset,
        ml_method=hgbc_ml_method,
        emb_method=hgbc_emb_method,
        pca_n_comp="none",
        best_params=hgbc_best_params,
        concatenation=conc,
        is_train=True,
        metrics=hgbc_train_score,
        output_file=f"train/baseline/{dataset}_hgbc_train.csv")

    save_results_to_csv(
        dataset_name=hgbc_dataset,
        ml_method=hgbc_ml_method,
        emb_method=hgbc_emb_method,
        pca_n_comp="none",
        best_params=hgbc_best_params,
        concatenation=conc,
        is_train=False,
        metrics=hgbc_test_scores,
        output_file=f"test/baseline/{dataset}_hgbc_test.csv")

    # 4. random trees embedding + hgbc
    (hgbc_rt_dataset, hgbc_rt_ml_method, hgbc_rt_emb_method, hgbc_rte_conc, hgbc_rte_best_params,
     hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores) = \
        hgbc_rte(dataset_name=dataset, X=X, y=y,
                 nominal_features=nominal_features)

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=True,
        metrics=hgbc_rt_emb_train_score,
        output_file=f"train/baseline/{dataset}_HGBC_rte_train.csv")

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=False,
        metrics=hgbc_rt_emb_test_scores,
        output_file=f"test/baseline/{dataset}_HGBC_rte_test.csv")

    # 5. LR conc RTE
    (hgbc_rt_dataset, hgbc_rt_ml_method, hgbc_rt_emb_method, hgbc_rte_conc, hgbc_rte_best_params,
     hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores) = \
        concat_lr_rte(dataset_name=dataset, X_tabular=X, y=y,
                      nominal_features=nominal_features, pca=None, imp_max_iter=30, class_max_iter=10000)

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=True,
        metrics=hgbc_rt_emb_train_score,
        output_file=f"train/baseline/{dataset}_LR_rte_conc_train.csv")

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=False,
        metrics=hgbc_rt_emb_test_scores,
        output_file=f"test/baseline/{dataset}_LR_rte_conc_test.csv")


    # 6. HGBC conc RTE
    (hgbc_rt_dataset, hgbc_rt_ml_method, hgbc_rt_emb_method, hgbc_rte_conc, hgbc_rte_best_params,
     hgbc_rt_emb_train_score, hgbc_rt_emb_test_scores) = \
        concat_hgbc_rte(dataset_name=dataset, X_tabular=X, y=y,
                        nominal_features=nominal_features, pca=None, imp_max_iter=30)

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=True,
        metrics=hgbc_rt_emb_train_score,
        output_file=f"train/baseline/{dataset}_HGBC_rte_conc_train.csv")

    save_results_to_csv(
        dataset_name=hgbc_rt_dataset,
        ml_method=hgbc_rt_ml_method,
        emb_method=hgbc_rt_emb_method,
        pca_n_comp="none",
        best_params=hgbc_rte_best_params,
        concatenation=hgbc_rte_conc,
        is_train=False,
        metrics=hgbc_rt_emb_test_scores,
        output_file=f"test/baseline/{dataset}_HGBC_rte_conc_test.csv")
