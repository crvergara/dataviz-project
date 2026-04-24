import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/master_dataset.parquet')
df_re = df[df['region'] != 13]

df_sub = df_re[df_re['grupo_vivienda'] == 'Subsidiada']
df_priv = df_re[df_re['grupo_vivienda'] == 'No Subsidiada / Otro']

def wpct(d, mask):
    w = d['expr'].sum()
    return (d[mask(d)]['expr'].sum() / w * 100) if w > 0 else 0

pob_sub = wpct(df_sub, lambda x: x['pobreza'].isin([1,2]))
pob_priv = wpct(df_priv, lambda x: x['pobreza'].isin([1,2]))

dep_sub = wpct(df_sub, lambda x: x['subsidy_ratio'] > 0.20)
dep_priv = wpct(df_priv, lambda x: x['subsidy_ratio'] > 0.20)

ing_sub = wpct(df_sub, lambda x: x['yautcorh'] < 600000)
ing_priv = wpct(df_priv, lambda x: x['yautcorh'] < 600000)

print('--- ZOOM REGIONES (Sub vs Priv) ---')
print(f'Pobreza por ingresos - Sub:  {pob_sub:.1f}%')
print(f'Pobreza por ingresos - Priv: {pob_priv:.1f}%')
print(f'Dependencia Estatal >20% - Sub:  {dep_sub:.1f}%')
print(f'Dependencia Estatal >20% - Priv: {dep_priv:.1f}%')
print(f'Ingreso Autonomo <$600k - Sub:  {ing_sub:.1f}%')
print(f'Ingreso Autonomo <$600k - Priv: {ing_priv:.1f}%')
