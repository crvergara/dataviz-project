import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/master_dataset.parquet')

print("=== CHEQUEO DE CÓDIGOS CASEN (v36b y v36c) ===")
print("Valores de v36b (Balaceras):")
print(df['v36b'].value_counts(dropna=False).sort_index())

print("\nValores de v36c (Narco):")
print(df['v36c'].value_counts(dropna=False).sort_index())

print("\n=== CHEQUEO DE SUBSIDY RATIO ===")
print(df['subsidy_ratio'].describe())

print("\n=== CHEQUEO DE INGRESO AUTONOMO ===")
print(df['yautcorh'].describe())

print("\n=== CHEQUEO POBREZA ===")
print(df['pobreza'].value_counts(dropna=False).sort_index())
