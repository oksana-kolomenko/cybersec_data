import numpy as np

from csv_saver import save_results_to_csv
from data_preps import load_features, load_labels, load_summaries
from helpers import concat_lr_rte, concat_hgbc_rte, concat_lr_txt_emb, concat_hgbc_txt_emb, hgbc_txt_emb, lr_txt_emb
from models import (feature_extractor_gist_embedding_v0, feature_extractor_gte_small,
                    feature_extractor_gte_base_en_v1_5, feature_extractor_gte_base)
#, feature_extractor_gte_base, feature_extractor_bge_base_en_v1_5, \
 #   feature_extractor_gte_base_en_v1_5, feature_extractor_gte_large)

#from helpers import concat_lr_rte, concat_hgbc_rte, lr_txt_emb, hgbc_txt_emb, concat_lr_txt_emb, concat_hgbc_txt_emb
from models import (feature_extractor_bge_base_en_v1_5, feature_extractor_bge_small_en_v1_5, \
    feature_extractor_gist_small_embedding_v0, feature_extractor_e5_small_v2, \
    feature_extractor_e5_base_v2, feature_extractor_all_minilm_l6_v2)

from values import DatasetName


def run_txt_emb():
    # === POSTTRAUMA ===
    """dataset = DatasetName.POSTTRAUMA.value
    X = load_features("X_posttrauma.csv")
    y = load_labels("y_posttrauma.csv")
    summaries = load_summaries("posttrauma_summaries.txt")

    nominal_features = [
        'gender_birth',
        'ethnic_group',
        'education_age',
        'working_at_baseline',
        'penetrating_injury'
    ]"""


    # === LUNGDISEASE ===

    """
    dataset = DatasetName.LUNG_DISEASE.value
    y = load_labels("data/y_lung_disease_data.csv", n_samples=100)
    X = load_features("data/X_lung_disease_data.csv", n_samples=100)
    X_metr = load_features("data/X_lungdisease_metrics.csv", n_samples=100)
    all_summaries = load_summaries("data/_lung_disease_summaries.txt", n_samples=100)
    nom_summaries = load_summaries("data/lungdisease_nom_summaries.txt", n_samples=100)

    nominal_features = [
        'Gender',
        'Smoking Status',
        'Disease Type',
        'Treatment Type'
    ]"""


    # === CYBERSECURITY ===

    dataset = DatasetName.CYBERSECURITY.value
    y = load_labels("data/y_cybersecurity_intrusion_data.csv")#, n_samples=51)
    X = load_features("data/X_cybersecurity_intrusion_data.csv")#), n_samples=51)
    X_metr = load_features("data/X_cybersecurity_metrics.csv")#, n_samples=51)
    all_summaries = load_summaries("data/_cybersecurity_summaries.txt")#, n_samples=51)
    nom_summaries = load_summaries("data/_cybersecurity_nom_summaries.txt")#, n_samples=51)

    nominal_features = [
        'encryption_used',
        'browser_type',
        'protocol_type',
        'unusual_time_access'
    ]

    methods = {
        # all summaries, all features
        """                                     
        # all summaries, metr features
        "pca_conc2": {"X": X_metr,
                          "summaries": all_summaries,
                          "conc": "conc2",
                          "pca": True,
                          "pca_str": "pca_"},
        
        # nom summaries, metr features
        "conc3": {"X": X_metr,
                  "summaries": nom_summaries,
                  "conc": "conc3",
                  "pca": False,
                  "pca_str": ""},        
        # all summaries, metr features
        "conc2": {"X": X_metr,
                  "summaries": all_summaries,
                  "conc": "conc2",
                  "pca": False,
                  "pca_str": ""},                                
        "conc1": {"X": X,
                  "summaries": all_summaries,
                  "conc": "conc1",
                  "pca": False,
                  "pca_str": ""},
                  "pca_conc1": {"X": X,
                          "summaries": all_summaries,
                          "conc": "conc1",
                          "pca": True,
                          "pca_str": "pca_"},
        """
        # nom summaries, metr features
        "pca_conc3": {"X": X_metr,
                          "summaries": nom_summaries,
                          "conc": "conc3",
                          "pca": True,
                          "pca_str": "pca_"},
    }

    text_feature = 'text'

    feature_extractors = {
        # All MiniLM L6 v2
        #"all_miniLM_L6_v2": feature_extractor_all_minilm_l6_v2,

        # Stella en 400m v5
        #"Stella-EN-400M-v5": feature_extractor_stella_en_400M_v5,

        # GTR T5 Base
        #"GTR_T5_Base": feature_extractor_gtr_t5_base,

        # Sentence T5 Base
        #"sentence_t5_base": feature_extractor_sentence_t5_base,

        # Ember v1
        #"ember_v1": feature_extractor_ember_v1,

        # E5 Models
        #"E5-Small-V2": feature_extractor_e5_small_v2,
        #"E5-Base-V2": feature_extractor_e5_base_v2,
        #"E5-Large-V2": feature_extractor_e5_large_v2,

        # BGE Models (done)
        #"BGE-Small-EN-v1.5": feature_extractor_bge_small_en_v1_5,
        "BGE-Base-EN-v1.5": feature_extractor_bge_base_en_v1_5,
        #"BGE-Large-EN-v1.5": feature_extractor_bge_large_en_v1_5,

        # GIST Models
        #"GIST-Small-Embedding-v0": feature_extractor_gist_small_embedding_v0,
        "GIST-Embedding-v0": feature_extractor_gist_embedding_v0,
        #"GIST-Large-Embedding-v0": feature_extractor_gist_large_embedding_v0,

        # GTE Models
        "GTE-Base": feature_extractor_gte_base,
        "GTE-Base-EN-v1.5": feature_extractor_gte_base_en_v1_5,
        #"GTE-Large": feature_extractor_gte_large,
        #"GTE-Small": feature_extractor_gte_small,

        # Potion Models
        # "Potion-Base-2M": feature_extractor_potion_base_2M,
        # "Potion-Base-4M": feature_extractor_potion_base_4M,
        # "Potion-Base-8M": feature_extractor_potion_base_8M,

        ####### jetzt nicht ################
        # Clinical Longformer (done)
        # "Clinical-Longformer": feature_extractor_clinical,

        # BERT (half done)
        # "BERT-Base": feature_extractor_bert,

        # ELECTRA (half done)
        # "ELECTRA-Small": feature_extractor_electra_small,
        # "ELECTRA-Base": feature_extractor_electra_base,
        # "ELECTRA-Large": feature_extractor_electra_large,

        # SimSCE (done)
        # "SimSCE-Sup": feature_extractor_simsce_sup,
        # "SimSCE-Unsup": feature_extractor_simsce_unsup,

        # MedEmbed Models (problem)
        # "MedEmbed-Small-v0.1": feature_extractor_medembed_small_v0_1,
        # "MedEmbed-Base-v0.1": feature_extractor_medembed_base_v0_1,
        # "MedEmbed-Large-v0.1": feature_extractor_medembed_large_v0_1,

        # "GTE-Large-EN-v1.5": feature_extractor_gte_large_en_v1_5,

        # modernbert-embed-base
        # "modernbert_embed_base": feature_extractor_mbert_embed_base,

        # GTE modernbert base
        # "gte_modernbert_base": feature_extractor_gte_mbert_base,
    }

    for model_name, feature_extractor in feature_extractors.items():

        #######################
        ### no PCA, no CONC ###
        #######################

        """# Logistic Regression
        (lr_txt_dataset, lr_txt_ml_method, lr_txt_emb_method, lr_txt_concatenation, lr_txt_best_params,
         lr_txt_pca_components, lr_txt_train_score, lr_txt_test_scores) = lr_txt_emb(
            dataset_name=dataset, emb_method=model_name,
            feature_extractor=feature_extractor, max_iter=10000,
            raw_text_summaries=all_summaries, y=y, pca=False)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_train.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_test.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_test_scores, is_train=False)"""

        # HGBC
        """(hgbc_txt_dataset, hgbc_txt_ml_method, hgbc_txt_emb_method, hgbc_txt_conc, hgbc_best_params, hgbc_pca_comp,
         hgbc_txt_train_score, hgbc_txt_test_scores) \
            = hgbc_txt_emb(dataset_name=dataset,
                           emb_method=model_name,
                           feature_extractor=feature_extractor,
                           summaries=all_summaries,
                           y=y, pca=False)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_train.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_test.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_test_scores,
                            is_train=False)"""

        ####################
        ### PCA, no CONC ###
        ####################

        # Logistic Regression
        """(lr_txt_dataset, lr_txt_ml_method, lr_txt_emb_method, lr_txt_concatenation, lr_txt_best_params,
         lr_txt_pca_components, lr_txt_train_score, lr_txt_test_scores) = lr_txt_emb(
            dataset_name=dataset, emb_method=model_name,
            feature_extractor=feature_extractor, max_iter=10000,
            raw_text_summaries=all_summaries, y=y, pca=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_pca_train.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_train_score, is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_pca_test.csv", dataset_name=lr_txt_dataset,
                            ml_method=lr_txt_ml_method, emb_method=lr_txt_emb_method, concatenation=lr_txt_concatenation,
                            best_params=lr_txt_best_params, pca_n_comp=lr_txt_pca_components,
                            metrics=lr_txt_test_scores, is_train=False)

        # HGBC
        (hgbc_txt_dataset, hgbc_txt_ml_method, hgbc_txt_emb_method, hgbc_txt_conc, hgbc_best_params, hgbc_pca_comp,
         hgbc_txt_train_score, hgbc_txt_test_scores) \
            = hgbc_txt_emb(dataset_name=dataset,
                           emb_method=model_name,
                           feature_extractor=feature_extractor,
                           summaries=all_summaries,
                           y=y, pca=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_pca_train.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_train_score,
                            is_train=True)

        save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_pca_test.csv",
                            dataset_name=hgbc_txt_dataset,
                            ml_method=hgbc_txt_ml_method,
                            emb_method=hgbc_txt_emb_method,
                            concatenation=hgbc_txt_conc,
                            best_params=hgbc_best_params,
                            pca_n_comp=hgbc_pca_comp,
                            metrics=hgbc_txt_test_scores,
                            is_train=False)"""

        for method_name, attributes in methods.items():
            #################
            ### PCA, CONC ###
            #################

            conc_art = attributes.get("conc")
            X = attributes.get("X")
            summaries = attributes.get("summaries")
            pca = attributes.get("pca")
            pca_str = attributes.get("pca_str")

            # Logistic Regression conc (pca)
            """
            (lr_conc_dataset, lr_conc_ml_method, lr_conc_emb_method,
             lr_conc_yesno, lr_best_params, lr_pca_components, lr_conc_train_score,
             lr_conc_test_scores) = concat_lr_txt_emb(
                dataset_name=dataset,
                emb_method=model_name,
                feature_extractor=feature_extractor,
                raw_text_summaries=summaries,
                X_tabular=X,
                y=y,
                nominal_features=nominal_features,
                text_feature_column_name=text_feature,
                concatenation=conc_art,
                imp_max_iter=30, class_max_iter=10000, pca=pca)
                #imp_max_iter=10, class_max_iter=10, pca=True)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_{conc_art}_{pca_str}train.csv",
                                dataset_name=lr_conc_dataset,
                                ml_method=lr_conc_ml_method,
                                emb_method=lr_conc_emb_method,
                                concatenation=lr_conc_yesno,
                                best_params=lr_best_params,
                                pca_n_comp=lr_pca_components,
                                metrics=lr_conc_train_score,
                                is_train=True)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_LR_{conc_art}_{pca_str}test.csv",
                                dataset_name=lr_conc_dataset,
                                ml_method=lr_conc_ml_method,
                                emb_method=lr_conc_emb_method,
                                concatenation=lr_conc_yesno,
                                best_params=lr_best_params,
                                pca_n_comp=lr_pca_components,
                                metrics=lr_conc_test_scores,
                                is_train=False)"""

            # HGBC conc (pca)

            (concat_hgbc_dataset, concat_hgbc_ml_method, concat_hgbc_emb_method,
             hgbc_conc_yesno, hgbc_best_params, hgbc_pca_components, hgbc_conc_train_score,
             hgbc_conc_test_scores) = concat_hgbc_txt_emb(
                dataset_name=dataset,
                emb_method=model_name,
                feature_extractor=feature_extractor,
                raw_text_summaries=summaries,
                X_tabular=X, y=y,
                text_feature_column_name=text_feature,
                concatenation=conc_art, pca=pca, nominal_features=nominal_features)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_{conc_art}_{pca_str}train.csv",
                                dataset_name=concat_hgbc_dataset,
                                ml_method=concat_hgbc_ml_method,
                                emb_method=concat_hgbc_emb_method,
                                concatenation=hgbc_conc_yesno,
                                best_params=hgbc_best_params,
                                pca_n_comp=hgbc_pca_components,
                                metrics=hgbc_conc_train_score,
                                is_train=True)

            save_results_to_csv(output_file=f"{dataset}_{model_name}_HGBC_{conc_art}_{pca_str}test.csv",
                                dataset_name=concat_hgbc_dataset,
                                ml_method=concat_hgbc_ml_method,
                                emb_method=concat_hgbc_emb_method,
                                concatenation=hgbc_conc_yesno,
                                best_params=hgbc_best_params,
                                pca_n_comp=hgbc_pca_components,
                                metrics=hgbc_conc_test_scores,
                                is_train=False)
