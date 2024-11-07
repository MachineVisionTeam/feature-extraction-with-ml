import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from collections import defaultdict
import logging
from datetime import datetime
import warnings
import joblib
warnings.filterwarnings('ignore')

def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with file and console handlers"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger
class NucleiClassifier:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.output_dir = 'updated_classification_results'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Add models directory
        self.models_dir = os.path.join(self.output_dir, 'trained_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Setup main logger
        self.logger = setup_logger(
            'nuclei_classifier',
            f'logs/classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        self.tissue_types = [
            "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", 
            "Colon", "Esophagus", "HeadNeck", "Kidney", "Liver", "Lung", 
            "Ovarian", "Pancreatic", "Prostate", "Skin", "Stomach", 
            "Testis", "Thyroid", "Uterus"
        ]
        
        self.nuclei_types = {
            'Inflammatory': 1,
            'Connective': 2,
            'Dead': 3,
            'Epithelial': 4,
            'Neoplastic': 5
        }
        
        # Initialize models with balanced settings
        self.models = {
            'Logistic_Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced'
            ),
            'Decision_Tree': DecisionTreeClassifier(
                class_weight='balanced'
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100, class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            'LightGBM': lgb.LGBMClassifier(
                class_weight='balanced'
            ),
            'SVM': SVC(
                probability=True,
                class_weight='balanced'
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=100
            )
        }
        
        # Track processing statistics
        self.stats = defaultdict(lambda: defaultdict(dict))
        self.missing_data = defaultdict(list)

    def process_all_tissues(self):
        """Process all tissue and nuclei combinations"""
        all_results = []
        
        try:
            for tissue_type in self.tissue_types:
                self.logger.info(f"\nProcessing tissue: {tissue_type}")
                print(f"\nProcessing tissue: {tissue_type}")
                
                try:
                    # Load tissue data
                    df = pd.read_csv(os.path.join(self.base_dir, f'{tissue_type}_train_nuclei_features.csv'))
                    self.logger.info(f"Loaded {tissue_type} data: {df.shape}")
                    print(f"Loaded {tissue_type} data: {df.shape}")
                    
                    # Get feature columns
                    feature_cols = [col for col in df.columns 
                                  if col not in ['Label', 'Name', 'Mask'] 
                                  and not col.startswith('Identifier')]
                    
                    # Process each nuclei type for this tissue
                    for nuclei_name, nuclei_id in self.nuclei_types.items():
                        results = self.process_single_combination(
                            tissue_type, nuclei_name, nuclei_id, df, feature_cols
                        )
                        all_results.extend(results)
                        
                except Exception as e:
                    self.logger.error(f"Error processing tissue {tissue_type}: {str(e)}")
                    print(f"Error processing tissue {tissue_type}: {str(e)}")
                    continue
            
            # Save and analyze results
            if all_results:
                self.save_and_analyze_results(all_results)
            else:
                self.logger.error("No results were generated!")
                print("No results were generated!")
            
        except Exception as e:
            self.logger.error(f"Error in main processing: {str(e)}")
            print(f"Error in main processing: {str(e)}")
            raise

    def process_single_combination(self, tissue_type, nuclei_name, nuclei_id, df, feature_cols):
        """Process a single tissue-nuclei combination"""
        try:
            self.logger.info(f"Processing {tissue_type} - {nuclei_name}")
            print(f"Processing {tissue_type} - {nuclei_name}")
            
            # Create binary classification target
            y = (df['Mask'] == nuclei_id).astype(int)
            
            # Get class distribution
            class_dist = dict(zip(*np.unique(y, return_counts=True)))
            self.logger.info(f"Class distribution for {tissue_type} - {nuclei_name}: {class_dist}")
            
            # Check if we have both classes
            if len(class_dist) < 2:
                self.logger.warning(f"Skipping {tissue_type} - {nuclei_name}: Only one class present")
                print(f"Skipping {tissue_type} - {nuclei_name}: Only one class present")
                return []

            X = df[feature_cols]
            X_processed = self.preprocess_data(X, tissue_type, nuclei_name)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Select features
            selected_features = self.select_features(X_train, y_train, tissue_type, nuclei_name)
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Train all models
            results = []
            trained_models = {}
            
            for model_name, model in self.models.items():
                try:
                    metrics, trained_model = self.train_evaluate_model(
                        model, X_train_selected, X_test_selected, 
                        y_train, y_test, tissue_type, nuclei_name, model_name
                    )
                    
                    # Save the trained model
                    self.save_trained_model(
                        trained_model,
                        tissue_type,
                        nuclei_name,
                        model_name,
                        selected_features
                    )
                    
                    trained_models[model_name] = trained_model
                    results.append({
                        'Tissue_Type': tissue_type,
                        'Cell_Type': nuclei_name,
                        'Machine_Learning': model_name,
                        'Precision': metrics['Precision'],
                        'Recall': metrics['Recall'],
                        'F1_score': metrics['F1'],
                        'Accuracy': metrics['Accuracy'],
                        'Selected_Features': ', '.join(selected_features)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error with {model_name}: {str(e)}")
                    print(f"Error with {model_name}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {tissue_type} - {nuclei_name}: {str(e)}")
            print(f"Error processing {tissue_type} - {nuclei_name}: {str(e)}")
            return []

    def save_and_analyze_results(self, all_results):
        """Save and analyze all results"""
        try:
            # Convert results to DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Save main results
            main_results_path = os.path.join(self.output_dir, 'complete_classification_results.csv')
            results_df.to_csv(main_results_path, index=False)
            self.logger.info(f"Saved complete results to {main_results_path}")
            print(f"Saved complete results to {main_results_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            print(f"Error saving results: {str(e)}")
            raise

    def preprocess_data(self, X, tissue_type, nuclei_type):
        """Preprocess features with logging"""
        try:
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns
            )
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_imputed),
                columns=X.columns
            )
            
            self.logger.info(f"{tissue_type} - {nuclei_type}: Preprocessing completed successfully")
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"{tissue_type} - {nuclei_type}: Preprocessing error: {str(e)}")
            raise
    
    def get_class_weights(self, y):
        """Calculate balanced class weights"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        weights = dict(zip(unique, n_samples / (n_classes * counts)))
        
        return weights
    
    def select_features(self, X, y, tissue_type, nuclei_type, n_features=20):
        """Select features with logging"""
        try:
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selector.fit(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Log feature selection results
            self.logger.info(
                f"{tissue_type} - {nuclei_type}: Selected {len(selected_features)} features"
            )
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"{tissue_type} - {nuclei_type}: Feature selection error: {str(e)}")
            raise

    def save_trained_model(self, model, tissue_type, nuclei_name, model_name, selected_features):
        """Save trained model and its selected features"""
        try:
            # Create directory for this tissue-nuclei combination
            model_dir = os.path.join(self.models_dir, f"{tissue_type}_{nuclei_name}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            
            # Save selected features
            features_path = os.path.join(model_dir, f"{model_name}_features.txt")
            with open(features_path, 'w') as f:
                f.write('\n'.join(selected_features))
            
            self.logger.info(f"Saved {model_name} model and features for {tissue_type} - {nuclei_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name} for {tissue_type} - {nuclei_name}: {str(e)}")
            raise

    def train_evaluate_model(self, model, X_train, X_test, y_train, y_test, tissue_type, nuclei_type, model_name):
        """Train and evaluate a single model with detailed logging"""
        try:
            # Get class weights
            class_weights = self.get_class_weights(y_train)
            
            # Set class weights for supported models
            if hasattr(model, 'set_params'):
                if isinstance(model, xgb.XGBClassifier):
                    # For XGBoost, use scale_pos_weight
                    scale_pos_weight = class_weights[1] / class_weights[0]
                    model.set_params(scale_pos_weight=scale_pos_weight)
                elif hasattr(model, 'class_weight'):
                    # For sklearn models
                    model.set_params(class_weight=class_weights)

            # Log training start
            self.logger.info(f"{tissue_type} - {nuclei_type} - {model_name}: Starting training")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'Accuracy': accuracy_score(y_test, y_pred)
            }
            
            # Log results
            self.logger.info(
                f"{tissue_type} - {nuclei_type} - {model_name}: "
                f"F1={metrics['F1']:.4f}, Precision={metrics['Precision']:.4f}, "
                f"Recall={metrics['Recall']:.4f}, Accuracy={metrics['Accuracy']:.4f}"
            )
            
            return metrics, model
            
        except Exception as e:
            self.logger.error(f"{tissue_type} - {nuclei_type} - {model_name}: Training error: {str(e)}")
     
            raise

def verify_saved_models(self):
    """Verify that models are saved and print their locations"""
    print("\nVerifying saved models...")
    
    for tissue_type in self.tissue_types:
        for nuclei_name in self.nuclei_types.keys():
            model_dir = os.path.join(self.models_dir, f"{tissue_type}_{nuclei_name}")
            
            if os.path.exists(model_dir):
                print(f"\nChecking {tissue_type} - {nuclei_name}:")
                models_found = os.listdir(model_dir)
                print(f"Found {len(models_found)} files:")
                for model_file in models_found:
                    file_size = os.path.getsize(os.path.join(model_dir, model_file))
                    print(f"- {model_file} ({file_size/1024:.2f} KB)")
            else:
                print(f"\nNo models found for {tissue_type} - {nuclei_name}")

def save_trained_model(self, model, tissue_type, nuclei_name, model_name, selected_features):
    """Save trained model and its selected features with verification"""
    try:
        # Create directory for this tissue-nuclei combination
        model_dir = os.path.join(self.models_dir, f"{tissue_type}_{nuclei_name}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        
        # Save selected features
        features_path = os.path.join(model_dir, f"{model_name}_features.txt")
        with open(features_path, 'w') as f:
            f.write('\n'.join(selected_features))
        
        # Verify the files were saved
        if os.path.exists(model_path) and os.path.exists(features_path):
            print(f"\nSuccessfully saved {model_name} for {tissue_type} - {nuclei_name}")
            print(f"Model path: {model_path}")
            print(f"Features path: {features_path}")
        else:
            print(f"\nError: Files not found after saving for {model_name}")
        
        self.logger.info(f"Saved {model_name} model and features for {tissue_type} - {nuclei_name}")
        
    except Exception as e:
        print(f"\nError saving model {model_name} for {tissue_type} - {nuclei_name}: {str(e)}")
        self.logger.error(f"Error saving model {model_name} for {tissue_type} - {nuclei_name}: {str(e)}")
        raise

# Modified main function with verification
def main():
    # Set the base directory containing the tissue data files
    base_dir = '/mnt/storage2/PanNuke/features'  # Modify this path to your actual data location
    
    # Initialize and run classifier
    classifier = NucleiClassifier(base_dir)
    
    try:
        print("\nStarting model training and saving process...")
        classifier.process_all_tissues()
        
        print("\nVerifying all saved models...")
        classifier.verify_saved_models()
        
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nMain execution error: {str(e)}")
        classifier.logger.error(f"Main execution error: {str(e)}")
        raise

if __name__ == "__main__":
    main()        