# bayesian_optimization.py
# Set up Bayesian optimization for models with GPU support.

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import logging
import time
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import pynvml
import joblib
import json
import sklearn
import platform
import concurrent
from sklearn.impute import SimpleImputer
import optuna
from optuna.samplers import TPESampler

# Suppress warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """Bayesian Optimization for hyperparameter tuning"""
    def __init__(self, X_train, X_test, y_train, y_test, model_type, n_trials=50):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type
        self.n_trials = n_trials
        self.study = None

    def optimize(self):
        """Run Bayesian optimization"""
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        optimization_functions = {
            "XGBoost": self._objective_xgboost,
            "LightGBM": self._objective_lightgbm,
            "Random Forest": self._objective_rf,
            "SVM": self._objective_svm,
            "Gradient Boosting": self._objective_gb,
            "Decision Tree": self._objective_dt,
            "Logistic Regression": self._objective_lr
        }
        
        if self.model_type in optimization_functions:
            study.optimize(optimization_functions[self.model_type], n_trials=self.n_trials)
            self.study = study
            return study.best_params
        return None

    def _objective_xgboost(self, trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42
        }
        return self._evaluate_model(xgb.XGBClassifier(**params))

    def _objective_lightgbm(self, trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42
        }
        return self._evaluate_model(lgb.LGBMClassifier(**params))

    def _objective_rf(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'n_jobs': -1,
            'random_state': 42
        }
        return self._evaluate_model(RandomForestClassifier(**params))

    def _objective_svm(self, trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            'probability': True,
            'random_state': 42
        }
        return self._evaluate_model(SVC(**params))

    def _objective_gb(self, trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        return self._evaluate_model(GradientBoostingClassifier(**params))

    def _objective_dt(self, trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        return self._evaluate_model(DecisionTreeClassifier(**params))

    def _objective_lr(self, trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'random_state': 42
        }
        return self._evaluate_model(LogisticRegression(**params))

    def _evaluate_model(self, model):
        try:
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            return f1_score(self.y_test, y_pred)
        except Exception as e:
            return float('-inf')

def setup_gpu():
    """Initialize GPU and return device info"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            return handle, gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
        else:
            logging.warning("CUDA is not available. Using CPU.")
            return None, "CPU"
    except Exception as e:
        logging.error(f"GPU initialization failed: {str(e)}")
        return None, "CPU"

class GPUMonitor:
    def __init__(self, handle):
        self.handle = handle
        self.monitoring = True
        self.peak_memory = 0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True

    def _monitor(self):
        while self.monitoring:
            if self.handle:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    self.peak_memory = max(self.peak_memory, mem_info.used / 1024**2)
                except Exception as e:
                    logging.error(f"GPU monitoring error: {str(e)}")
            time.sleep(0.1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.monitoring = False
        self.thread.join()
        return self.peak_memory

def setup_logging(log_file='tissue_classification.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class GPUModelTrainer:
    def __init__(self, device_handle):
        self.handle = device_handle
        self.logger = self._setup_logger()
        self.trained_models = {}
        self.optimizer_results = {}
        
        # Default model parameters (fallback if optimization fails)
        self.model_params = {
            'Logistic Regression': {
                'max_iter': 1000,
                'n_jobs': -1,
                'solver': 'lbfgs',
                'multi_class': 'auto'
            },
            'Decision Tree': {
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': 16,
                'n_jobs': -1,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'XGBoost': {
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'max_depth': 8,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'LightGBM': {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': -1,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary',
                'metric': 'binary_logloss',
                'random_state': 42
            },
            'SVM': {
                'kernel': 'rbf',
                'probability': True,
                'C': 1.0,
                'random_state': 42
            },
            'Gradient Boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 8,
                'random_state': 42
            }
        }

    def _setup_logger(self):
        """Setup logger for the GPU Model Trainer"""
        logger = logging.getLogger('GPU_Training')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        if not logger.handlers:
            fh = logging.FileHandler('gpu_training.log')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def train_model(self, model_name, X_train, y_train, X_test, y_test, tissue_type, nuclei_type):
        """Train a single model with Bayesian optimization and GPU acceleration"""
        self.logger.info(f"Starting {model_name} optimization and training for {tissue_type} - {nuclei_type}")
        monitor = GPUMonitor(self.handle)
        monitor.start()
        
        start_time = time.time()
        
        try:
            # Initialize Bayesian optimizer
            optimizer = BayesianOptimizer(
                X_train, X_test, y_train, y_test,
                model_name, n_trials=50
            )
            
            # Run optimization
            best_params = optimizer.optimize()
            if best_params:
                self.logger.info(f"Best parameters found for {model_name}: {best_params}")
                current_params = {**self.model_params[model_name], **best_params}
            else:
                current_params = self.model_params[model_name]
            
            # Initialize and train model
            model_classes = {
                'Logistic Regression': LogisticRegression,
                'Decision Tree': DecisionTreeClassifier,
                'Random Forest': RandomForestClassifier,
                'XGBoost': xgb.XGBClassifier,
                'LightGBM': lgb.LGBMClassifier,
                'SVM': SVC,
                'Gradient Boosting': GradientBoostingClassifier
            }
            
            model = model_classes[model_name](**current_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store trained model
            if tissue_type not in self.trained_models:
                self.trained_models[tissue_type] = {}
            if nuclei_type not in self.trained_models[tissue_type]:
                self.trained_models[tissue_type][nuclei_type] = {}
            self.trained_models[tissue_type][nuclei_type][model_name] = model
            
            # Calculate metrics
            metrics = {
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'Accuracy': accuracy_score(y_test, y_pred),
            }
            
            training_time = time.time() - start_time
            self.logger.info(f"Completed {model_name} training successfully")
            self.logger.info(f"Training time: {training_time:.2f} seconds")
            
            return metrics, model
            
        except Exception as e:
            self.logger.error(f"Error in optimization/training {model_name}: {str(e)}")
            monitor.stop()
            return None, None
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
def process_tissue_type(args):
    """Process a single tissue type with all nuclei types"""
    tissue_type, data_dir, selected_features_df, trainer = args
    
    results = []
    tissue_logger = logging.getLogger(f'Tissue_{tissue_type}')
    handler = logging.FileHandler(f'logs/{tissue_type}_processing.log')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    tissue_logger.addHandler(handler)
    
    try:
        # Load data
        file_path = os.path.join(data_dir, f'{tissue_type}_train_nuclei_features.csv')
        df = pd.read_csv(file_path)
        tissue_logger.info(f"Loaded data for {tissue_type}: {df.shape}")
        
        # Get selected features for this tissue type
        tissue_features_all = selected_features_df[selected_features_df['Tissue'] == tissue_type]
        if tissue_features_all.empty:
            tissue_logger.error(f"No selected features found for {tissue_type}")
            return results
        
        # Combine features from all models
        all_selected_features = set()
        for _, row in tissue_features_all.iterrows():
            features = row['Top Features'].split(', ')
            all_selected_features.update(features)
        
        selected_features = list(all_selected_features)
        tissue_logger.info(f"Selected {len(selected_features)} features")
        
        # Validate features exist in dataframe
        valid_features = [f for f in selected_features if f in df.columns]
        if len(valid_features) < len(selected_features):
            tissue_logger.warning(f"Some features not found in dataset. Using {len(valid_features)} available features")
        
        if not valid_features:
            tissue_logger.error("No valid features found")
            return results
        
        # Prepare features
        X = df[valid_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        nuclei_types = {
            1: "Inflammatory",
            2: "Connective",
            3: "Dead",
            4: "Epithelial",
            5: "Neoplastic"
        }
        
        for nuclei_id, nuclei_name in nuclei_types.items():
            tissue_logger.info(f"Processing {nuclei_name} nuclei")
            y = (df['Mask'] == nuclei_id).astype(int)
            
            # Check class balance
            class_counts = y.value_counts()
            tissue_logger.info(f"Class distribution for {nuclei_name}: {class_counts.to_dict()}")
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Train models
                for model_name in [
                    'Logistic Regression', 'Decision Tree', 'Random Forest',
                    'XGBoost', 'LightGBM', 'SVM', 'Gradient Boosting'
                ]:
                    tissue_logger.info(f"Training {model_name} for {nuclei_name}")
                    metrics, _ = trainer.train_model(
                        model_name, X_train, y_train, X_test, y_test,
                        tissue_type, nuclei_name
                    )
                    
                    if metrics:
                        results.append({
                            'Tissue_Type': tissue_type,
                            'Nuclei_Type': nuclei_name,
                            'Model': model_name,
                            'Precision': metrics['Precision'],
                            'Recall': metrics['Recall'],
                            'F1': metrics['F1'],
                            'Accuracy': metrics['Accuracy'],
                            'Selected Features': ', '.join(valid_features)
                        })
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            except Exception as e:
                tissue_logger.error(f"Error processing {nuclei_name}: {str(e)}")
                continue
        
    except Exception as e:
        tissue_logger.error(f"Error processing {tissue_type}: {str(e)}")
    
    finally:
        handler.close()
        tissue_logger.removeHandler(handler)
    
    return results

class ResultsAnalyzer:
    def __init__(self, results_df, output_dir):
        self.results_df = results_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger('ResultsAnalyzer')
        
    def analyze_by_tissue(self):
        """Analyze and save results grouped by tissue type"""
        tissue_dir = os.path.join(self.output_dir, 'tissue_analysis')
        os.makedirs(tissue_dir, exist_ok=True)
        
        summary_metrics = []
        
        for tissue in self.results_df['Tissue_Type'].unique():
            self.logger.info(f"Analyzing tissue: {tissue}")
            tissue_data = self.results_df[self.results_df['Tissue_Type'] == tissue]
            
            # Save detailed tissue results
            tissue_file = os.path.join(tissue_dir, f'{tissue}_analysis.csv')
            tissue_data.to_csv(tissue_file, index=False)
            
            # Calculate average performance by model for this tissue
            model_avg = tissue_data.groupby('Model')[
                ['Precision', 'Recall', 'F1', 'Accuracy']
            ].agg(['mean', 'std'])
            
            model_avg.to_csv(os.path.join(tissue_dir, f'{tissue}_model_averages.csv'))
            
            # Find best model
            best_model = tissue_data.loc[tissue_data['F1'].idxmax()]
            
            # Add to summary
            summary_metrics.append({
                'Tissue': tissue,
                'Best_Model': best_model['Model'],
                'F1_Score': best_model['F1'],
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary_metrics)
        summary_df.to_csv(os.path.join(tissue_dir, 'tissue_summary.csv'), index=False)

    def analyze_by_nuclei(self):
        """Analyze and save results grouped by nuclei type"""
        nuclei_dir = os.path.join(self.output_dir, 'nuclei_analysis')
        os.makedirs(nuclei_dir, exist_ok=True)
        
        for nuclei in self.results_df['Nuclei_Type'].unique():
            self.logger.info(f"Analyzing nuclei type: {nuclei}")
            nuclei_data = self.results_df[self.results_df['Nuclei_Type'] == nuclei]
            
            # Save detailed results
            nuclei_file = os.path.join(nuclei_dir, f'{nuclei}_analysis.csv')
            nuclei_data.to_csv(nuclei_file, index=False)
            
            # Performance by model
            model_avg = nuclei_data.groupby('Model')[
                ['Precision', 'Recall', 'F1', 'Accuracy']
            ].agg(['mean', 'std'])
            
            model_avg.to_csv(os.path.join(nuclei_dir, f'{nuclei}_model_averages.csv'))

def main():
    # Setup logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(log_dir, 'tissue_classification.log'))
    
    try:
        # Setup GPU
        handle, gpu_name = setup_gpu()
        logger.info(f"Using device: {gpu_name}")
        
        if gpu_name != "CPU":
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        trainer = GPUModelTrainer(handle)
        
        # Paths
        data_dir = '/mnt/storage2/PanNuke/features'
        selected_features_file = '/mnt/storage2/selected_features_per_tissue.csv'
        output_dir = '/mnt/storage2/PanNuke/classification_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Load selected features
        try:
            selected_features_df = pd.read_csv(selected_features_file)
            logger.info(f"Selected features DataFrame columns: {selected_features_df.columns}")
            logger.info("\nFirst few rows of selected features:")
            logger.info(selected_features_df.head().to_string())
        except Exception as e:
            logger.error(f"Error loading selected features: {str(e)}")
            raise
        
        # Define tissue types
        tissue_types = [
            "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon",
            "Esophagus", "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian",
            "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus"
        ]
        
        # Process tissues in parallel
        args = [(tissue, data_dir, selected_features_df, trainer) for tissue in tissue_types]
        all_results = []
        
        max_workers = min(len(tissue_types), psutil.cpu_count())
        logger.info(f"Processing with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tissue = {executor.submit(process_tissue_type, arg): arg[0] for arg in args}
            
            for future in concurrent.futures.as_completed(future_to_tissue):
                tissue = future_to_tissue[future]
                try:
                    tissue_results = future.result()
                    if tissue_results:
                        all_results.extend(tissue_results)
                        logger.info(f"Completed processing {tissue}")
                except Exception as e:
                    logger.error(f"Error processing {tissue}: {str(e)}")
        
        # Check if we have any results
        if not all_results:
            logger.error("No results were generated from any tissue type")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        logger.info(f"Created DataFrame with columns: {results_df.columns}")
        
        # Check if required columns exist
        numeric_cols = ['Precision', 'Recall', 'F1', 'Accuracy']
        missing_cols = [col for col in numeric_cols if col not in results_df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return
        
        # Format numeric columns
        results_df[numeric_cols] = results_df[numeric_cols].round(4)
        
        # Save main results
        results_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
        logger.info(f"Saved results to: {os.path.join(output_dir, 'all_results.csv')}")
        
        # Perform detailed analysis
        analyzer = ResultsAnalyzer(results_df, output_dir)
        analyzer.analyze_by_tissue()
        analyzer.analyze_by_nuclei()
        
        # Display summary statistics
        logger.info("\nOverall Model Performance:")
        model_perf = results_df.groupby('Model')['F1'].mean().sort_values(ascending=False)
        logger.info("\n" + model_perf.to_string())
        
        # Display sample of results
        logger.info("\nSample of results (first 10 rows):")
        logger.info("\n" + results_df.head(10).to_string())
        
    except Exception as e:
        logger.error(f"An error occurred in main execution: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Execution completed")

if __name__ == "__main__":
    main()
