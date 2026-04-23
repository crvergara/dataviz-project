import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_vulnerability_boxplot(df):
    """
    Boxplot nacional: Acumulación de vulnerabilidades.
    """
    # 1. CALCULAR ÍNDICE (0-5)
    df['flag_commute'] = np.where(df['o28a_hr'] >= 1, 1, 0)
    df['flag_crime'] = np.where(df['v36e'] == 1, 1, 0)
    df['flag_hacinamiento'] = np.where(df['ind_hacina'] >= 2, 1, 0)
    df['flag_pobreza_multi'] = np.where(df['pobreza_multi'] == 1, 1, 0)
    df['flag_salud'] = np.where(df['s13'].isin([1, 2]), 1, 0)

    df['total_vulnerabilities'] = (
        df['flag_commute'] + df['flag_crime'] + 
        df['flag_hacinamiento'] + df['flag_pobreza_multi'] + 
        df['flag_salud']
    )

    # 2. CONFIGURACIÓN VISUAL
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Boxplot con strip plot (puntos) para ver densidad
    sns.boxplot(
        data=df, x='grupo_vivienda', y='total_vulnerabilities',
        palette=['#4575B4', '#D73027'], ax=ax, width=0.5,
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}
    )

    # 3. ESTÉTICA
    ax.set_title("Distribución Nacional de Vulnerabilidades Acumuladas\n(0 = Ninguna | 5 = Crítica)", 
                 fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel("")
    ax.set_ylabel("Cantidad de Vulnerabilidades (Índice 0-5)")
    ax.set_yticks(range(6))
    sns.despine()

    plt.tight_layout()
    
    output_path = "test/outputs/vulnerability_boxplot_national.png"
    os.makedirs("test/outputs", exist_ok=True)
    plt.savefig(output_path)
    
    # ESTADÍSTICAS
    stats = df.groupby('grupo_vivienda')['total_vulnerabilities'].describe()
    print(f"Gráfico: {output_path}")
    print("\nEstadísticas Descriptivas:")
    print(stats[['mean', '50%', '75%', 'max']])

if __name__ == "__main__":
    df_master = pd.read_parquet("data/processed/master_dataset.parquet")
    generate_vulnerability_boxplot(df_master)
