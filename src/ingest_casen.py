import pandas as pd
from pathlib import Path

def ingest_casen_stata(file_path: str, output_path: str):
    print(f"Ingesting STATA file from {file_path}...")
    
    # We only load the columns we need for the Shift-Share and validation.
    # WARNING: Verify these exact names in your Excel codebook!
    target_columns = [
        'id_vivienda', 'folio', 'id_persona', 'region', 'expr',
        'v13',          # Housing tenure
        'v15',          # Purchased with subsidy (1=Yes, 2=No)
        'sexo',         # Gender (1=H, 2=M)
        'activ',        # Labor force status
        'o9a',          # Occupation status
        'oficio1_08',   # CIUO 08 Group
        'oficio4_08',   # CIUO 08 Detailed
        'o28a_hr',      # Commute time (hours)
        'o28a_min',     # Commute time (minutes)
        'yautcorh',     # Autonomous household income
        'ysubh',        # Household subsidies
        'ytotcorh',     # Total household income
        'pobreza'       # Poverty status
    ]
    
    # read_stata preserves the native data types defined by the Ministry.
    # convert_categoricals=False keeps the raw codes (e.g., '1', '2') instead of 
    # converting them to text labels like 'Región de Tarapacá'. 
    # We want raw codes for programmatic merging later.
    df = pd.read_stata(
        file_path, 
        columns=target_columns,
        convert_categoricals=False 
    )
    
    # Save immediately to Parquet for fast, compressed downstream access
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    print(f"Successfully saved to {output_path} | Rows: {len(df)}")

if __name__ == "__main__":
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    ingest_casen_stata(
        file_path="data/raw/casen_2024.dta", # Update to the actual extracted filename
        output_path="data/processed/casen_2024_base.parquet"
    )