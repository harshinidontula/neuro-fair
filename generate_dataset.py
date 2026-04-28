import pandas as pd
import numpy as np
import os

def generate_biased_dataset(num_samples=1000):
    np.random.seed(42)
    
    # 0 = Female, 1 = Male
    gender = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    
    # Age between 22 and 65
    age = np.random.randint(22, 65, size=num_samples)
    
    # Credit Score between 300 and 850
    credit_score = np.random.normal(650, 50, size=num_samples)
    credit_score = np.clip(credit_score, 300, 850).astype(int)
    
    # Income mapping: let's introduce a slightly higher income for males
    base_income = 40000
    income = base_income + (credit_score - 500) * 100 + np.random.normal(0, 10000, size=num_samples)
    income = np.where(gender == 1, income + 15000, income)  # Bias: Male income is artificially higher
    income = np.clip(income, 20000, 200000).astype(int)
    
    # Target: Loan Approved (0 = No, 1 = Yes)
    # We will make Approval highly dependent on Income and Credit Score, which inherently passes down the gender bias
    score = (credit_score / 850 * 0.4) + (income / 200000 * 0.4) + np.random.normal(0, 0.1, size=num_samples)
    
    # Additionally, let's explicitly add direct bias where males have a slightly higher chance
    score = np.where(gender == 1, score + 0.05, score)
    
    # Convert score to hard binary labels based on a threshold
    approved = (score > 0.6).astype(int)
    
    df = pd.DataFrame({
        'Gender': np.where(gender == 1, 'Male', 'Female'),
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Approved': approved
    })
    
    return df

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df = generate_biased_dataset(1000)
    df.to_csv('data/dataset.csv', index=False)
    print("Dataset generated successfully at data/dataset.csv")
