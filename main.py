import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os


def load_and_preprocess_data(filepath="credit_risk.csv"):
    """Loads data and handles basic missing values to ensure rule evaluation doesn't crash."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] '{filepath}' not found. Please ensure the dataset is in the same folder.")
    
    df = pd.read_csv(filepath)
    
    # Basic preprocessing for robustness
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].fillna('N')
    df['loan_grade'] = df['loan_grade'].fillna('G') # Worst case assumption
    df['debt_to_income_ratio'] = df['debt_to_income_ratio'].fillna(0)
    df['past_delinquencies'] = df['past_delinquencies'].fillna(0)
    df['credit_utilization_ratio'] = df['credit_utilization_ratio'].fillna(0)
    
    return df

def algo_1_deterministic_knockout(row):
    """
    Algorithm 1: Deterministic Threshold Expert System.
    Logic: If ANY single severe risk factor is present, predict Default (1). 
    Otherwise, predict Non-Default (0).
    """
    # Rule 1: Prior default on file is an immediate knockout
    if row['cb_person_default_on_file'] == 'Y':
        return 1
    # Rule 2: Very poor credit grade (F or G)
    if str(row['loan_grade']).upper() in ['F', 'G']:
        return 1
    # Rule 3: Debt-to-income ratio exceeds 50%
    if row['debt_to_income_ratio'] > 0.50:
        return 1
    # Rule 4: Excessive past delinquencies
    if row['past_delinquencies'] >= 3:
        return 1
        
    return 0

def algo_2_weighted_scoring(row):
    """
    Algorithm 2: Weighted Scoring Rule Engine.
    Logic: Accumulates risk points across multiple dimensions. 
    Allows for 'trade-offs' (e.g., a 'C' grade can be offset by low DTI).
    Threshold: >= 45 points predicts Default (1).
    """
    score = 0
    
    # Prior default is the heaviest weighting
    if row['cb_person_default_on_file'] == 'Y':
        score += 40
        
    # Loan grade tiered scoring
    grade = str(row['loan_grade']).upper()
    if grade in ['F', 'G']:
        score += 30
    elif grade in ['D', 'E']:
        score += 15
        
    # Debt-to-income ratio tiered scoring
    if row['debt_to_income_ratio'] > 0.50:
        score += 25
    elif row['debt_to_income_ratio'] > 0.35:
        score += 15
        
    # Past delinquencies scoring
    if row['past_delinquencies'] >= 3:
        score += 20
    elif row['past_delinquencies'] >= 1:
        score += 10
        
    # Credit utilization scoring
    if row['credit_utilization_ratio'] > 0.75:
        score += 15
        
    # Decision threshold
    return 1 if score >= 45 else 0

def evaluate_algorithm(df, algo_func, algo_name):
    """Applies the algorithm and calculates standard classification metrics."""
    # Apply the rule-based function to every row
    predictions = df.apply(algo_func, axis=1)
    actuals = df['loan_status']
    
    # Calculate metrics (using zero_division=0 to handle edge cases safely)
    acc = accuracy_score(actuals, predictions)
    prec = precision_score(actuals, predictions, zero_division=0)
    rec = recall_score(actuals, predictions, zero_division=0)
    f1 = f1_score(actuals, predictions, zero_division=0)
    
    results = {
        'Algorithm': algo_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    }
    
    return results, predictions

if __name__ == "__main__":
    print("="*70)
    print("CPU5006: Rule-Based AI Experiment Execution")
    print("="*70)
    
    # 1. Load Data
    print("\n[1/4] Loading and preprocessing data...")
    df = load_and_preprocess_data("credit_risk.csv")
    print(f"      Loaded {len(df)} records successfully.")
    
    # 2. Execute Algorithm 1
    print("\n[2/4] Executing Algorithm 1: Deterministic Knockout System...")
    results_1, preds_1 = evaluate_algorithm(df, algo_1_deterministic_knockout, "Algo 1: Deterministic Knockout")
    
    # 3. Execute Algorithm 2
    print("\n[3/4] Executing Algorithm 2: Weighted Scoring Engine...")
    results_2, preds_2 = evaluate_algorithm(df, algo_2_weighted_scoring, "Algo 2: Weighted Scoring")
    
    # 4. Display Results
    print("\n[4/4] EXPERIMENTAL RESULTS")
    print("-" * 70)
    
    results_df = pd.DataFrame([results_1, results_2])
    print(results_df.to_string(index=False))
    print("-" * 70)
    
    # Save results to a CSV for easy inclusion in the report
    results_df.to_csv("experiment_results.csv", index=False)
    print("\n[SUCCESS] Results saved to 'experiment_results.csv'")
    print("="*70)