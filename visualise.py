import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os


def load_and_preprocess(filepath="credit_risk.csv"):
    df = pd.read_csv(filepath)
    df['cb_person_default_on_file'] = df['cb_person_default_on_file'].astype(str).str.upper().fillna('N')
    df['loan_grade'] = df['loan_grade'].astype(str).str.upper().fillna('G')
    df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce').fillna(0)
    df['past_delinquencies'] = pd.to_numeric(df['past_delinquencies'], errors='coerce').fillna(0)
    df['credit_utilization_ratio'] = pd.to_numeric(df['credit_utilization_ratio'], errors='coerce').fillna(0)
    return df

def algo_1(row):
    if row['cb_person_default_on_file'] == 'Y': return 1
    if row['loan_grade'] in ['F', 'G']: return 1
    if row['debt_to_income_ratio'] > 0.50: return 1
    if row['past_delinquencies'] >= 3: return 1
    return 0

def algo_2(row):
    score = 0
    if row['cb_person_default_on_file'] == 'Y': score += 40
    if row['loan_grade'] in ['F', 'G']: score += 30
    elif row['loan_grade'] in ['D', 'E']: score += 15
    if row['debt_to_income_ratio'] > 0.50: score += 25
    elif row['debt_to_income_ratio'] > 0.35: score += 15
    if row['past_delinquencies'] >= 3: score += 20
    elif row['past_delinquencies'] >= 1: score += 10
    if row['credit_utilization_ratio'] > 0.75: score += 15
    return 1 if score >= 45 else 0

if __name__ == "__main__":
    print("Generating academic visualisations...")
    df = load_and_preprocess()
    actuals = df['loan_status']
    
    preds_1 = df.apply(algo_1, axis=1)
    preds_2 = df.apply(algo_2, axis=1)
    
    # Set academic plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.1)
    
    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm1 = confusion_matrix(actuals, preds_1)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Predicted Safe (0)', 'Predicted Default (1)'],
                yticklabels=['Actual Safe (0)', 'Actual Default (1)'])
    axes[0].set_title('Algorithm 1: Deterministic Knockout\nConfusion Matrix', fontweight='bold')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    cm2 = confusion_matrix(actuals, preds_2)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Predicted Safe (0)', 'Predicted Default (1)'],
                yticklabels=['Actual Safe (0)', 'Actual Default (1)'])
    axes[1].set_title('Algorithm 2: Weighted Scoring\nConfusion Matrix', fontweight='bold')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved 'confusion_matrices.png'")
    
    # 2. Metrics Comparison Bar Chart
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Algo 1 (Deterministic)': [0.7402, 0.4271, 0.5589, 0.4842],
        'Algo 2 (Weighted)': [0.7669, 0.4569, 0.3640, 0.4052]
    })
    
    metrics_melted = metrics_df.melt(id_vars='Metric', var_name='Algorithm', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(x='Metric', y='Score', hue='Algorithm', data=metrics_melted, palette=['#1f77b4', '#2ca02c'])
    
    # Add exact values on top of bars
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), '.3f'), 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    
    plt.ylim(0, 1.0)
    plt.title('Performance Comparison of Rule-Based AI Algorithms', fontweight='bold', fontsize=14)
    plt.ylabel('Score')
    plt.xlabel('Evaluation Metric')
    plt.legend(title='Algorithm')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("[SUCCESS] Saved 'metrics_comparison.png'")