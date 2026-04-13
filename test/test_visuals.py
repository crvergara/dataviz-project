import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_gender_trap_plot(df):
    """Generates a horizontal dumbbell chart showing labor inactivity by gender and housing type."""
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATION 1: THE GENDER TRAP (REFINED) ")
    print("="*80)
    
def generate_gender_trap_plot(df):
    """Generates a horizontal dumbbell chart showing labor inactivity by gender and housing type."""
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATION 1: THE GENDER TRAP (ENGLISH & PASTEL) ")
    print("="*80)
    
    df_plot = df.copy()
    # Map to English
    df_plot['genero'] = df_plot['sexo'].map({1: 'Men', 2: 'Women'})
    
    inact = df_plot.groupby(['grupo_vivienda', 'genero'])['activ'].apply(lambda x: (x == 3).mean() * 100).reset_index()
    inact.rename(columns={'activ': 'pct_inactivo'}, inplace=True)
    
    # Premium aesthetics
    plt.figure(figsize=(10, 5), facecolor='#F8F9FA') # Soft off-white global background
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA') # Plot area background
    
    # Add a white grid
    ax.grid(color='white', linestyle='-', linewidth=2, axis='x', zorder=0)
    
    y_positions = {'Men': 0, 'Women': 1}
    
    # Pastel Colors
    color_priv = '#8DA9C4' # Soft grayish blue (Pastel)
    color_sub = '#F28F8C'  # Soft coral pink (Pastel)
    color_line = '#DDE2E5' # Soft gray for the connecting line
    
    for genero in ['Men', 'Women']:
        val_priv = inact[(inact['genero'] == genero) & (inact['grupo_vivienda'] == 'No Subsidiada / Otro')]['pct_inactivo'].values[0]
        val_sub = inact[(inact['genero'] == genero) & (inact['grupo_vivienda'] == 'Subsidiada')]['pct_inactivo'].values[0]
        delta = val_sub - val_priv
        y = y_positions[genero]
        
        # Line connecting
        ax.plot([val_priv, val_sub], [y, y], color=color_line, linewidth=6, zorder=1)
        
        # Points
        ax.scatter(val_priv, y, color=color_priv, s=350, edgecolor='white', linewidth=2, zorder=2)
        ax.scatter(val_sub, y, color=color_sub, s=350, edgecolor='white', linewidth=2, zorder=2)
        
        # Text for points (Offset slightly above)
        ax.text(val_priv, y - 0.20, f"{val_priv:.1f}%", va='top', ha='center', fontsize=11, fontweight='bold', color='#5D6D7E')
        ax.text(val_sub, y - 0.20, f"{val_sub:.1f}%", va='top', ha='center', fontsize=11, fontweight='bold', color='#B03A2E')
        
        # Text for delta (above line)
        ax.text((val_priv + val_sub) / 2, y + 0.15, f"+{delta:.2f} pp", va='bottom', ha='center', fontsize=12, fontweight='bold', color='#34495E')

    # Y-axis categorical labels
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Hombres', 'Mujeres'], fontsize=13, fontweight='bold', color='#2C3E50')
    
    # X-axis label
    ax.set_xlabel("Tasa de Inactividad Laboral (%)", fontsize=12, fontweight='bold', color='#5D6D7E', labelpad=15)
    
    # Clean spines
    sns.despine(left=False, bottom=False)
    ax.spines['left'].set_color('#E5E7E9')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_color('#E5E7E9')
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='x', colors='#5D6D7E', labelsize=11)
    ax.tick_params(axis='y', colors='#2C3E50', length=5, width=2) # Add short y ticks back
    ax.set_ylabel("Género", fontsize=12, fontweight='bold', color='#5D6D7E', labelpad=15)
    
    # Set X and Y limits to give some breathing room for the text
    min_x = inact['pct_inactivo'].min()
    max_x = inact['pct_inactivo'].max()
    ax.set_xlim(min_x - 3, max_x + 3)
    ax.set_ylim(-0.6, 1.6)
    
    # Centered Minimalist Title
    title_text = "Tasa de Inactividad Laboral por Género y Tipo de Vivienda"
    plt.title(title_text, fontsize=16, fontweight='bold', pad=30, loc='center', color='#2C3E50')

    # Custom Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color_priv, marker='o', linestyle='None', markersize=12),
                    Line2D([0], [0], color=color_sub, marker='o', linestyle='None', markersize=12)]
    ax.legend(custom_lines, ['Vivienda Privada', 'Vivienda Subsidiada'], 
              loc='lower right', frameon=True, facecolor='white', edgecolor='#E5E7E9', fontsize=10)
    
    out_dir = "data_explorer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viz1_gender_trap.png")
    
    plt.tight_layout() 
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
    plt.close()
    
    print(f"[OK] Visualization 1 successfully saved to: {out_path}")

def generate_regional_gaps_plot(df):
    """Generates two horizontal bar charts showing poverty and income gaps by region."""
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATION 2: REGIONAL POVERTY & INCOME GAPS ")
    print("="*80)
    
    df_plot = df.dropna(subset=['region', 'grupo_vivienda', 'yautcorh', 'pobreza']).copy()
    
    region_map = {
        1: 'Tarapacá', 2: 'Antofagasta', 3: 'Atacama', 4: 'Coquimbo',
        5: 'Valparaíso', 6: "O'Higgins", 7: 'Maule', 8: 'Biobío',
        9: 'Araucanía', 10: 'Los Lagos', 11: 'Aysén', 12: 'Magallanes',
        13: 'Metropolitana', 14: 'Los Ríos', 15: 'Arica y Parinacota', 16: 'Ñuble'
    }
    df_plot['region_name'] = df_plot['region'].map(region_map)
    df_plot['is_poor'] = df_plot['pobreza'].isin([1.0, 2.0])
    
    # Calculate metrics
    grp = df_plot.groupby(['region_name', 'grupo_vivienda']).apply(
        lambda x: pd.Series({
            'poverty_rate': np.average(x['is_poor'], weights=x['expr']) * 100 if x['expr'].sum() > 0 else 0,
            'median_income': x['yautcorh'].median()
        })
    ).reset_index()
    
    # Pivot to get sorting order based on Subsidy poverty rate (highest to lowest)
    pivot_pov = grp.pivot(index='region_name', columns='grupo_vivienda', values='poverty_rate')
    sorted_regions = pivot_pov.sort_values('Subsidiada', ascending=False).index.tolist()
    
    # Rename group items to match the legend
    grp['grupo_vivienda'] = grp['grupo_vivienda'].replace({
        'Subsidiada': 'CON Subsidio',
        'No Subsidiada / Otro': 'SIN Subsidio'
    })
    
    # Apply ordering to categorical mapping
    grp['region_name'] = pd.Categorical(grp['region_name'], categories=sorted_regions, ordered=True)
    grp = grp.sort_values(['region_name', 'grupo_vivienda'])
    
    # Make the plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor='white')
    
    hue_order = ['CON Subsidio', 'SIN Subsidio']
    
    # Plot 1: Poverty Gap
    colors_pov = {'CON Subsidio': '#5975A4', 'SIN Subsidio': '#CC8963'} # Matches user's Blue/Orange
    sns.barplot(
        data=grp, y='region_name', x='poverty_rate', hue='grupo_vivienda',
        palette=colors_pov, hue_order=hue_order, ax=axes[0], orient='h', width=0.6,
        edgecolor='none'
    )
    axes[0].set_title('Brecha de Pobreza Multidimensional por Región', fontsize=15, fontweight='bold', pad=15)
    axes[0].set_xlabel('Tasa de Pobreza (%)', fontsize=13, labelpad=10)
    axes[0].set_ylabel('')
    axes[0].grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=1, zorder=0)
    axes[0].set_axisbelow(True) # Put grid behind bars
    
    # Legend formatting
    axes[0].legend(title='Vivienda', loc='lower right', frameon=True, 
                   facecolor='white', edgecolor='#D0D0D0', fontsize=11, title_fontsize=12)
    
    # Plot 2: Income Gap
    colors_inc = {'CON Subsidio': '#5F9E6E', 'SIN Subsidio': '#B55D60'} # Matches user's Green/Red
    sns.barplot(
        data=grp, y='region_name', x='median_income', hue='grupo_vivienda',
        palette=colors_inc, hue_order=hue_order, ax=axes[1], orient='h', width=0.6,
        edgecolor='none'
    )
    axes[1].set_title('Brecha de Ingresos por Región', fontsize=15, fontweight='bold', pad=15)
    axes[1].set_xlabel('Ingreso Mediano del Hogar ($)', fontsize=13, labelpad=10)
    axes[1].set_ylabel('')
    axes[1].grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=1, zorder=0)
    axes[1].set_axisbelow(True)
    
    axes[1].legend(title='Vivienda', loc='lower right', frameon=True, 
                   facecolor='white', edgecolor='#D0D0D0', fontsize=11, title_fontsize=12)
    
    # Format X-axis for monetary values (clp dot format)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', '.')))
    
    # Clean up both axes
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E0E0E0')
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_color('#E0E0E0')
        ax.spines['bottom'].set_linewidth(1)
        ax.tick_params(axis='y', length=0, labelsize=11)
        ax.tick_params(axis='x', labelsize=11)
        
    plt.tight_layout(pad=3.0)
    
    out_dir = "data_explorer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viz2_regional_gaps.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[OK] Visualization 2 successfully saved to: {out_path}")

def generate_income_composition_plot(df):
    """Generates a 100% horizontal stacked bar chart showing income composition."""
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATION 3: COMPOSICION DE INGRESOS ")
    print("="*80)

    df_sub = df.dropna(subset=['region', 'grupo_vivienda', 'ysubh', 'yautcorh', 'ytotcorh']).copy()
    region_map = {1: 'Tarapacá', 2: 'Antofagasta', 3: 'Atacama', 4: 'Coquimbo',
                  5: 'Valparaíso', 6: "O'Higgins", 7: 'Maule', 8: 'Biobío',
                  9: 'Araucanía', 10: 'Los Lagos', 11: 'Aysén', 12: 'Magallanes',
                  13: 'Metropolitana', 14: 'Los Ríos', 15: 'Arica y Parinacota', 16: 'Ñuble'}
    df_sub['region_name'] = df_sub['region'].map(region_map)
    
    grp_sub = df_sub.groupby(['region_name', 'grupo_vivienda']).apply(
        lambda x: pd.Series({
            'Subsidio': (np.average(x['ysubh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0,
            'Autónomo': (np.average(x['yautcorh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0
        })
    ).reset_index()
    
    grp_sub['Otros'] = 100 - (grp_sub['Subsidio'] + grp_sub['Autónomo'])
    
    # Format and pivot for plotting
    pivot_sub = grp_sub.pivot(index='region_name', columns='grupo_vivienda', values='Subsidio')
    sorted_regions = pivot_sub.sort_values('Subsidiada', ascending=True).index.tolist()
    
    grp_sub['region_name'] = pd.Categorical(grp_sub['region_name'], categories=sorted_regions, ordered=True)
    grp_sub = grp_sub.sort_values(['region_name', 'grupo_vivienda'], ascending=[True, False])
    
    # Plot definitions
    fig, ax = plt.subplots(figsize=(10, 12), facecolor='white')
    
    # Colors
    c_sub = '#F28F8C'    # Rojo pastel
    c_aut = '#8DA9C4'    # Azul pastel
    c_otr = '#E5E7E9'    # Gris claro
    
    y_labels = []
    y_ticks = []
    
    y_base = 0
    for region in sorted_regions:
        reg_data = grp_sub[grp_sub['region_name'] == region]
        
        # Pull records
        s_data = reg_data[reg_data['grupo_vivienda'] == 'Subsidiada'].iloc[0]
        p_data = reg_data[reg_data['grupo_vivienda'] == 'No Subsidiada / Otro'].iloc[0]
        
        # Plot No Subsidiada (Top bar of the group)
        ax.barh(y_base + 0.4, p_data['Subsidio'], color=c_sub, height=0.35, edgecolor='white', linewidth=0.5, zorder=3)
        ax.barh(y_base + 0.4, p_data['Autónomo'], left=p_data['Subsidio'], color=c_aut, height=0.35, edgecolor='white', linewidth=0.5, zorder=3)
        ax.barh(y_base + 0.4, p_data['Otros'], left=p_data['Subsidio'] + p_data['Autónomo'], color=c_otr, height=0.35, edgecolor='white', linewidth=0.5, zorder=3)
        
        # Plot Subsidiada (Bottom bar of the group)
        ax.barh(y_base, s_data['Subsidio'], color=c_sub, height=0.35, edgecolor='white', linewidth=0.5, zorder=3)
        ax.barh(y_base, s_data['Autónomo'], left=s_data['Subsidio'], color=c_aut, height=0.35, edgecolor='white', linewidth=0.5, zorder=3)
        ax.barh(y_base, s_data['Otros'], left=s_data['Subsidio'] + s_data['Autónomo'], color=c_otr, height=0.35, edgecolor='white', linewidth=0.5, zorder=3)
        
        # Annotate percentages if large enough
        if s_data['Subsidio'] > 2:
            ax.text(s_data['Subsidio']/2, y_base, f"{s_data['Subsidio']:.1f}%", va='center', ha='center', color='#B03A2E', fontsize=8, fontweight='bold', zorder=4)
        if p_data['Subsidio'] > 2:
            ax.text(p_data['Subsidio']/2, y_base + 0.4, f"{p_data['Subsidio']:.1f}%", va='center', ha='center', color='#5D6D7E', fontsize=8, fontweight='bold', zorder=4)
            
        ax.text(s_data['Subsidio'] + s_data['Autónomo']/2, y_base, f"{s_data['Autónomo']:.1f}%", va='center', ha='center', color='white', fontsize=8, fontweight='bold', zorder=4)
        ax.text(p_data['Subsidio'] + p_data['Autónomo']/2, y_base + 0.4, f"{p_data['Autónomo']:.1f}%", va='center', ha='center', color='white', fontsize=8, fontweight='bold', zorder=4)

        # Annotate group labels a la derecha para no pisar el eje Y
        ax.text(101, y_base, "Subsidiada", va='center', ha='left', color='#B03A2E', fontsize=9, fontweight='bold')
        ax.text(101, y_base + 0.4, "Sin Subsidio", va='center', ha='left', color='#5D6D7E', fontsize=9)
        
        y_ticks.append(y_base + 0.2)
        y_labels.append(region)
        
        y_base += 1.5

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 115)
    
    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=c_sub, lw=8),
                    Line2D([0], [0], color=c_aut, lw=8),
                    Line2D([0], [0], color=c_otr, lw=8)]
    ax.legend(custom_lines, ['Ingreso por Subsidio', 'Ingreso Autónomo', 'Otros Ingresos (Residual)'], 
              loc='upper center', bbox_to_anchor=(0.5, -0.05), borderaxespad=0., ncol=3, frameon=False, fontsize=11)
              
    ax.set_title("Estructura del Bolsillo Familiar\nDependencia Estatal vs Autonomía Financiera", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Proporción del Ingreso Total (%)", fontsize=12, labelpad=10)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E0E0E0')
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.xaxis.grid(color='#F2F3F4', linestyle='-', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    out_dir = "data_explorer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viz3_income_composition.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Visualization 3 successfully saved to: {out_path}")

def generate_dependency_multiplier_plot(df):
    """Generates a horizontal lollipop chart showing the multiplier of state dependency."""
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATION 4: DEPENDENCY MULTIPLIER ")
    print("="*80)

    df_sub = df.dropna(subset=['region', 'grupo_vivienda', 'ysubh', 'ytotcorh']).copy()
    region_map = {1: 'Tarapacá', 2: 'Antofagasta', 3: 'Atacama', 4: 'Coquimbo',
                  5: 'Valparaíso', 6: "O'Higgins", 7: 'Maule', 8: 'Biobío',
                  9: 'Araucanía', 10: 'Los Lagos', 11: 'Aysén', 12: 'Magallanes',
                  13: 'Metropolitana', 14: 'Los Ríos', 15: 'Arica y Parinacota', 16: 'Ñuble'}
    df_sub['region_name'] = df_sub['region'].map(region_map)
    
    grp_sub = df_sub.groupby(['region_name', 'grupo_vivienda']).apply(
        lambda x: pd.Series({
            'pct_subsidio': (np.average(x['ysubh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0
        })
    ).reset_index()
    
    pivot = grp_sub.pivot(index='region_name', columns='grupo_vivienda', values='pct_subsidio')
    pivot['multiplier'] = pivot['Subsidiada'] / pivot['No Subsidiada / Otro']
    
    # Sort ascending for bottom-to-top plotting
    pivot = pivot.sort_values('multiplier', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    y = np.arange(len(pivot.index))
    multipliers = pivot['multiplier'].values
    regions = pivot.index.tolist()
    
    # Colors
    c_line = '#F28F8C'    # Pastel red for the stick
    c_dot = '#B03A2E'     # Dark red for the head
    c_base = '#8DA9C4'    # Pastel blue for the 1x baseline points
    
    # Plot geometry
    ax.hlines(y=y, xmin=1, xmax=multipliers, color=c_line, linewidth=4, zorder=2)
    ax.scatter(multipliers, y, color=c_dot, s=120, zorder=3)
    ax.scatter(np.ones(len(y)), y, color=c_base, s=80, zorder=3)
    
    # Baseline line at 1x
    ax.axvline(x=1, color='#AAB7B8', linestyle='--', linewidth=1.5, zorder=1)
    
    # Annotations on the dots
    for i, mult in enumerate(multipliers):
        ax.text(mult + 0.08, i, f"{mult:.1f}x", va='center', ha='left', color=c_dot, fontsize=10, fontweight='bold')
    
    ax.set_yticks(y)
    ax.set_yticklabels(regions, fontsize=11, fontweight='bold', color='#2C3E50')
    
    # Set x limits to give space for the text
    max_mult = pivot['multiplier'].max()
    ax.set_xlim(0.8, max_mult + 0.4)
    
    # Titles and labels
    ax.set_xlabel("Multiplicador de Dependencia Estatal (Tasa Relativa)", fontsize=12, labelpad=10, color='#5D6D7E', fontweight='bold')
    ax.set_title("Sobrerrepresentación de Asistencia Estatal en Viviendas Subsidiadas", 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # Floating legend box near the 1x line
    ax.text(1.1, len(y) - 1.5, "1.0x = Nivel de\nDependencia\nBase (Sector Privado)", 
            color='#5D6D7E', fontsize=9, bbox=dict(facecolor='white', edgecolor='#E0E0E0', boxstyle='round,pad=0.5'))
            
    # Cleanup spines
    import seaborn as sns
    sns.despine(left=True, bottom=False)
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.xaxis.grid(color='#F2F3F4', linestyle='-', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    out_dir = "data_explorer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viz4_dependency_multiplier.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualization 4 successfully saved to: {out_path}")

def generate_crowding_dependency_plot(df):
    """Generates a line chart showing dependency across crowding levels."""
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATION 5: HACINAMIENTO Y DEPENDENCIA ")
    print("="*80)

    if 'ind_hacina' not in df.columns:
        print("[!] 'ind_hacina' no encontrado. Saltando...")
        return

    df_hrel = df.dropna(subset=['ind_hacina', 'ysubh', 'ytotcorh', 'grupo_vivienda']).copy()
    
    hacin_map = {1.0: 'Sin hacinamiento', 2.0: 'Hacinamiento\nmedio bajo', 
                 3.0: 'Hacinamiento\nmedio alto', 4.0: 'Hacinamiento\ncrítico', 
                 5.0: 'Hacinamiento\ncrítico'}
    df_hrel['nivel_hacinamiento'] = df_hrel['ind_hacina'].map(hacin_map).fillna('Otro')
    df_hrel = df_hrel[df_hrel['nivel_hacinamiento'] != 'Otro']
    
    grp = df_hrel.groupby(['grupo_vivienda', 'nivel_hacinamiento']).apply(
        lambda x: pd.Series({
            'pct_ingreso_subsidio': (np.average(x['ysubh'], weights=x['expr']) / np.average(x['ytotcorh'], weights=x['expr'])) * 100 if np.average(x['ytotcorh'], weights=x['expr']) > 0 else 0
        })
    ).reset_index()
    
    cat_order = ['Sin hacinamiento', 'Hacinamiento\nmedio bajo', 'Hacinamiento\nmedio alto', 'Hacinamiento\ncrítico']
    grp['nivel_hacinamiento'] = pd.Categorical(grp['nivel_hacinamiento'], categories=cat_order, ordered=True)
    grp = grp.sort_values('nivel_hacinamiento')
    
    sub_data = grp[grp['grupo_vivienda'] == 'Subsidiada']
    priv_data = grp[grp['grupo_vivienda'] == 'No Subsidiada / Otro']
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    x = np.arange(len(cat_order))
    
    # Lines
    ax.plot(x, sub_data['pct_ingreso_subsidio'], color='#B03A2E', linewidth=4, marker='o', markersize=10, label='Vivienda Subsidiada', zorder=3)
    ax.plot(x, priv_data['pct_ingreso_subsidio'], color='#8DA9C4', linewidth=3, marker='o', markersize=8, label='Sin Subsidio / Privada', zorder=3)
    
    # Fill between lines
    ax.fill_between(x, priv_data['pct_ingreso_subsidio'], sub_data['pct_ingreso_subsidio'], color='#F28F8C', alpha=0.15, zorder=2)
    
    # Annotations
    for i, (_, row) in enumerate(sub_data.iterrows()):
        ax.text(i, row['pct_ingreso_subsidio'] + 0.6, f"{row['pct_ingreso_subsidio']:.1f}%", ha='center', color='#B03A2E', fontweight='bold', fontsize=10)
    for i, (_, row) in enumerate(priv_data.iterrows()):
        ax.text(i, row['pct_ingreso_subsidio'] - 1.2, f"{row['pct_ingreso_subsidio']:.1f}%", ha='center', color='#5D6D7E', fontweight='bold', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_ylim(0, max(sub_data['pct_ingreso_subsidio']) + 3)
    ax.set_ylabel("Dependencia del Estado (% del Ingreso Mantenido)", fontsize=11, labelpad=10, fontweight='bold', color='#5D6D7E')
    ax.set_title("La Capa Ineludible:\nLa dependencia estatal es estructuralmente mayor sin importar el grado de hacinamiento", 
                 fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
                 
    import seaborn as sns
    sns.despine(left=True, bottom=False)
    ax.spines['bottom'].set_color('#E0E0E0')
    ax.yaxis.grid(color='#F2F3F4', linestyle='-', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(frameon=False, fontsize=11, loc='lower right')
    
    plt.tight_layout()
    out_dir = "data_explorer"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viz5_crowding_dependency.png")
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization 5 successfully saved to: {out_path}")

if __name__ == '__main__':
    df = pd.read_parquet('data/processed/icp_results.parquet')
    df = df.replace([np.inf, -np.inf], np.nan)
    df['grupo_vivienda'] = np.where(df['v15'] == 1, 'Subsidiada', 'No Subsidiada / Otro')
    generate_gender_trap_plot(df)
    generate_regional_gaps_plot(df)
    generate_income_composition_plot(df)
    generate_dependency_multiplier_plot(df)
    generate_crowding_dependency_plot(df)
