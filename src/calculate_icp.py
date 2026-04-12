import pandas as pd
import numpy as np

def calculate_icp(df):
    # α and β represent weights for the dependency components
    alpha = 0.5
    beta = 0.5
    
    # 1. Clean Commute Time (Friction)
    # Filter out invalid commute hours (valid range is 0 to 9) and minutes (valid range is 0 to 59). Exclude rows with -8 or -88.
    mask_invalid_hr = (df['o28a_hr'].notna()) & ((df['o28a_hr'] < 0) | (df['o28a_hr'] > 9))
    mask_invalid_min = (df['o28a_min'].notna()) & ((df['o28a_min'] < 0) | (df['o28a_min'] > 59))
    df = df[~(mask_invalid_hr | mask_invalid_min)].copy()
    
    # Calculate DAILY ROUND-TRIP time (Ida y Vuelta = (Hours + Minutes/60) * 2)
    daily_commute_hrs = (df['o28a_hr'].fillna(0) + (df['o28a_min'].fillna(0) / 60)) * 2
    
    # Calculate MONTHLY commute hours (assuming 20 working days)
    df['commute_total_hrs'] = daily_commute_hrs * 20
    
    # 2. Imputed Cost of Isolation (C_friction)
    # Opportunity cost of time = (Autonomous Income / 160 hrs month) * Monthly Commute Time
    hourly_rate = df['yautcorh'] / 160
    df['c_friction'] = df['commute_total_hrs'] * hourly_rate.clip(lower=0)
    
    # 3. Handle ratios safely (prevent division by zero)
    # S / I_total
    df['subsidy_ratio'] = (df['ysubh'] / df['ytotcorh'].replace(0, np.nan)).fillna(0)
    
    # C_friction / A_potential 
    # (Since C_friction is now monthly, this naturally compares monthly friction against monthly income)
    df['isolation_ratio'] = (df['c_friction'] / df['yautcorh'].replace(0, np.nan)).fillna(0)
    
    # 4. FINAL ICP CALCULATION (Scaled 0-100 for public audience)
    # Now that the math is balanced, expanding to a 0-100 scale makes it an intuitive grade.
    df['icp'] = (alpha * df['subsidy_ratio'] + beta * df['isolation_ratio']) * 100
    
    return df

if __name__ == "__main__":
    df = pd.read_parquet("data/processed/casen_2024_base.parquet")
    df_icp = calculate_icp(df)
    
    # Aggregate by Region and Housing Type
    # Housing Type 1: Subsidized (v15 == 1)
    # Housing Type 0: Others
    df_icp['is_subsidized'] = (df_icp['v15'] == 1.0).astype(int)
    
    # Grouping to verify the thesis
    summary = df_icp.groupby(['region', 'is_subsidized']).agg({
        'icp': 'mean',
        'subsidy_ratio': 'mean',
        'isolation_ratio': 'mean',
        'yautcorh': 'mean'
    }).reset_index()
    
    print("--- ICP Summary (Thesis Check) ---")
    print(summary.head(10))
    
    df_icp.to_parquet("data/processed/icp_results.parquet")
    print("\nSaved ICP results to data/processed/icp_results.parquet")
