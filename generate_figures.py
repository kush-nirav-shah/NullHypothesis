#!/usr/bin/env python3
"""
generate_figures.py
Reads results/tables/all_results.csv and produces 10 publication-quality figures.
Usage: python generate_figures.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, 'results', 'tables', 'all_results.csv')
OUT_DIR = os.path.join(BASE, 'results', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──
df = pd.read_csv(CSV_PATH)

# ── Style constants ──
DPI = 300
METHOD_ORDER = ['Plug-in', 'RC-OOS', 'Exp.Simulation', 'Two-Stage Bayes', 'GPR', 'RF', 'GBR']
METHOD_COLORS = {
    'Plug-in': '#722F37',
    'RC-OOS': '#3A6B8F',
    'Exp.Simulation': '#8B7355',
    'Two-Stage Bayes': '#6B5B8B',
    'GPR': '#5A8A6E',
    'RF': '#B85C5C',
    'GBR': '#C49A3C',
}
METHOD_SHORT = {
    'Plug-in': 'Plug-in',
    'RC-OOS': 'RC-OOS',
    'Exp.Simulation': 'Exp.Sim',
    'Two-Stage Bayes': 'Bayes',
    'GPR': 'GPR',
    'RF': 'RF',
    'GBR': 'GBR',
}

plt.rcParams.update({
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#222222',
    'axes.labelcolor': '#222222',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#E0E0E0',
    'grid.linewidth': 0.5,
})


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, format='png', dpi=DPI)
    plt.close(fig)
    print(f'  {name}')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: replication_mse.png
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
rep = df[df.Study == 'Replication']
scenarios = ['A', 'B', 'C', 'D']
x = np.arange(len(scenarios))
w = 0.11
for i, m in enumerate(METHOD_ORDER):
    vals = [rep[(rep.Scenario == s) & (rep.Method == m)]['MSE'].values[0] for s in scenarios]
    ax.bar(x + i * w - 3 * w, vals, w, label=METHOD_SHORT[m], color=METHOD_COLORS[m], edgecolor='white', linewidth=0.3)
ax.set_xticks(x)
ax.set_xticklabels([f'Scenario {s}' for s in scenarios])
ax.set_ylabel('Mean Squared Error')
ax.set_title('Study 1: Replication MSE by Scenario', fontweight='bold', pad=10)
ax.legend(ncol=4, frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper left')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.4)
save(fig, 'replication_mse.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: replication_coverage.png
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, m in enumerate(METHOD_ORDER):
    vals = [rep[(rep.Scenario == s) & (rep.Method == m)]['Coverage'].values[0] for s in scenarios]
    ax.plot(scenarios, vals, 'o-', label=METHOD_SHORT[m], color=METHOD_COLORS[m],
            linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.axhline(y=95, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.6, label='Nominal 95%')
ax.set_xticks(scenarios)
ax.set_xticklabels([f'Scenario {s}' for s in scenarios])
ax.set_ylabel('Coverage (%)')
ax.set_title('Study 1: 95% CI Coverage by Scenario', fontweight='bold', pad=10)
ax.legend(ncol=4, frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='lower left')
ax.set_ylim(-5, 105)
ax.grid(alpha=0.4)
save(fig, 'replication_coverage.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: density_mse.png  (Epidemiology density)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
dens = df[(df.Study == 'Density') & (df.Domain == 'Epidemiology')].sort_values('N_Monitors')
monitors = sorted(dens['N_Monitors'].unique())
for m in METHOD_ORDER:
    vals = [dens[(dens.N_Monitors == nd) & (dens.Method == m)]['MSE'].values[0] for nd in monitors]
    ax.plot(monitors, vals, 'o-', label=METHOD_SHORT[m], color=METHOD_COLORS[m],
            linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xlabel('Number of Monitors')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Study 2: MSE vs Monitor Density (Epidemiology, Scenario C)', fontweight='bold', pad=10)
ax.legend(ncol=4, frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper right')
ax.set_xscale('log')
ax.set_xticks(monitors)
ax.set_xticklabels([str(m) for m in monitors])
ax.grid(alpha=0.4)
save(fig, 'density_mse.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: density_coverage.png  (Epidemiology density)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in METHOD_ORDER:
    vals = [dens[(dens.N_Monitors == nd) & (dens.Method == m)]['Coverage'].values[0] for nd in monitors]
    ax.plot(monitors, vals, 's-', label=METHOD_SHORT[m], color=METHOD_COLORS[m],
            linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.axhline(y=95, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.6)
ax.set_xlabel('Number of Monitors')
ax.set_ylabel('Coverage (%)')
ax.set_title('Study 2: Coverage vs Monitor Density (Epidemiology)', fontweight='bold', pad=10)
ax.legend(ncol=4, frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='lower left')
ax.set_xscale('log')
ax.set_xticks(monitors)
ax.set_xticklabels([str(m) for m in monitors])
ax.set_ylim(-5, 105)
ax.grid(alpha=0.4)
save(fig, 'density_coverage.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: modern_methods.png
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
mod = df[(df.Study == 'Modern') & (df.N_Monitors == 40)]
mse_vals = [mod[mod.Method == m]['MSE'].values[0] for m in METHOD_ORDER]
bars = ax.bar(range(len(METHOD_ORDER)), mse_vals,
              color=[METHOD_COLORS[m] for m in METHOD_ORDER],
              edgecolor='white', linewidth=0.5)
best_idx = int(np.argmin(mse_vals))
bars[best_idx].set_edgecolor('#2E7D32')
bars[best_idx].set_linewidth(2)
ax.set_xticks(range(len(METHOD_ORDER)))
ax.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER])
ax.set_ylabel('Mean Squared Error')
ax.set_title('Study 3: Modern Method Comparison (40 Monitors, Scenario C)', fontweight='bold', pad=10)
for i, v in enumerate(mse_vals):
    ax.text(i, v + max(mse_vals) * 0.02, f'{v:.3f}', ha='center', va='bottom',
            fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.4)
save(fig, 'modern_methods.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: agriculture_density.png
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
agri = df[(df.Study == 'Agriculture')].sort_values('N_Monitors')
agri_mon = sorted(agri['N_Monitors'].unique())
for m in METHOD_ORDER:
    vals = [agri[(agri.N_Monitors == nd) & (agri.Method == m)]['MSE'].values[0] for nd in agri_mon]
    ax.plot(agri_mon, vals, 'o-', label=METHOD_SHORT[m], color=METHOD_COLORS[m],
            linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.set_xlabel('Number of Sensors')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Study 4: MSE vs Sensor Density (Agriculture, Scenario C)', fontweight='bold', pad=10)
ax.legend(ncol=4, frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper right')
ax.set_xscale('log')
ax.set_xticks(agri_mon)
ax.set_xticklabels([str(m) for m in agri_mon])
ax.grid(alpha=0.4)
save(fig, 'agriculture_density.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: agriculture_coverage.png
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
for m in METHOD_ORDER:
    vals = [agri[(agri.N_Monitors == nd) & (agri.Method == m)]['Coverage'].values[0] for nd in agri_mon]
    ax.plot(agri_mon, vals, 's-', label=METHOD_SHORT[m], color=METHOD_COLORS[m],
            linewidth=1.8, markersize=5, markeredgecolor='white', markeredgewidth=0.5)
ax.axhline(y=95, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.6)
ax.set_xlabel('Number of Sensors')
ax.set_ylabel('Coverage (%)')
ax.set_title('Study 4: Coverage vs Sensor Density (Agriculture)', fontweight='bold', pad=10)
ax.legend(ncol=4, frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='lower left')
ax.set_xscale('log')
ax.set_xticks(agri_mon)
ax.set_xticklabels([str(m) for m in agri_mon])
ax.set_ylim(-5, 105)
ax.grid(alpha=0.4)
save(fig, 'agriculture_coverage.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8: cross_domain_comparison.png
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
common_densities = [80, 60, 40, 20, 10]
epi_d = df[(df.Study == 'Density') & (df.Domain == 'Epidemiology')]
agri_d = df[df.Study == 'Agriculture']

# Left: MSE comparison at each density
x_pos = np.arange(len(common_densities))
w = 0.35
for m in METHOD_ORDER:
    epi_vals = [epi_d[(epi_d.N_Monitors == nd) & (epi_d.Method == m)]['MSE'].values[0]
                for nd in common_densities]
    agri_vals = [agri_d[(agri_d.N_Monitors == nd) & (agri_d.Method == m)]['MSE'].values[0]
                 for nd in common_densities]
    axes[0].plot(common_densities, epi_vals, 'o-', color=METHOD_COLORS[m],
                 linewidth=1.2, markersize=4, alpha=0.5)
    axes[0].plot(common_densities, agri_vals, 's--', color=METHOD_COLORS[m],
                 linewidth=1.2, markersize=4, alpha=0.9)

# Add mean trendlines
epi_mean = [np.mean([epi_d[(epi_d.N_Monitors == nd) & (epi_d.Method == m)]['MSE'].values[0]
                     for m in METHOD_ORDER]) for nd in common_densities]
agri_mean = [np.mean([agri_d[(agri_d.N_Monitors == nd) & (agri_d.Method == m)]['MSE'].values[0]
                      for m in METHOD_ORDER]) for nd in common_densities]
axes[0].plot(common_densities, epi_mean, 'o-', color='#3A6B8F', linewidth=2.5,
             markersize=7, label='Epidemiology (mean)', zorder=5)
axes[0].plot(common_densities, agri_mean, 's--', color='#5A8A6E', linewidth=2.5,
             markersize=7, label='Agriculture (mean)', zorder=5)

axes[0].set_xlabel('Number of Monitors / Sensors')
axes[0].set_ylabel('Mean MSE')
axes[0].set_title('Mean MSE Across Domains', fontweight='bold')
axes[0].set_xscale('log')
axes[0].set_xticks(common_densities)
axes[0].set_xticklabels([str(d) for d in common_densities])
axes[0].legend(frameon=True, fancybox=False, edgecolor='#CCCCCC')
axes[0].grid(alpha=0.4)

# Right: method ranking heatmap at 40 monitors
epi40 = epi_d[epi_d.N_Monitors == 40].set_index('Method').loc[METHOD_ORDER]
agri40 = agri_d[agri_d.N_Monitors == 40].set_index('Method').loc[METHOD_ORDER]
rank_data = np.zeros((len(METHOD_ORDER), 2))
epi_ranked = epi40['MSE'].rank().values
agri_ranked = agri40['MSE'].rank().values
rank_data[:, 0] = epi_ranked
rank_data[:, 1] = agri_ranked

im = axes[1].imshow(rank_data, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=7)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Epidemiology', 'Agriculture'])
axes[1].set_yticks(range(len(METHOD_ORDER)))
axes[1].set_yticklabels([METHOD_SHORT[m] for m in METHOD_ORDER])
axes[1].set_title('MSE Rank at 40 Monitors\n(1=best, 7=worst)', fontweight='bold')
for i in range(len(METHOD_ORDER)):
    for j in range(2):
        val = int(rank_data[i, j])
        color = 'white' if val >= 5 else 'black'
        axes[1].text(j, i, str(val), ha='center', va='center', fontsize=11,
                     fontweight='bold', color=color)
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.suptitle('Cross-Domain Comparison', fontweight='bold', fontsize=14, y=1.02)
save(fig, 'cross_domain_comparison.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 9: runtime_comparison.png
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
# Use Scenario C replication as representative
rt = df[(df.Study == 'Replication') & (df.Scenario == 'C')]
runtimes = [rt[rt.Method == m]['Runtime_ms'].values[0] for m in METHOD_ORDER]
bars = ax.bar(range(len(METHOD_ORDER)), runtimes,
              color=[METHOD_COLORS[m] for m in METHOD_ORDER],
              edgecolor='white', linewidth=0.5)
ax.set_xticks(range(len(METHOD_ORDER)))
ax.set_xticklabels([METHOD_SHORT[m] for m in METHOD_ORDER])
ax.set_ylabel('Runtime (ms)')
ax.set_title('Runtime per Replication (Scenario C, 82 Monitors)', fontweight='bold', pad=10)
ax.set_yscale('log')
for i, v in enumerate(runtimes):
    ax.text(i, v * 1.3, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
ax.grid(axis='y', alpha=0.4)
save(fig, 'runtime_comparison.png')


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10: final_method_ranking.png
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Combine all studies: average MSE per method
all_mse = df.groupby('Method')['MSE'].median().reindex(METHOD_ORDER)
all_cov = df.groupby('Method')['Coverage'].median().reindex(METHOD_ORDER)

# Left: median MSE ranking
y_pos = np.arange(len(METHOD_ORDER))
axes[0].barh(y_pos, all_mse.values, color=[METHOD_COLORS[m] for m in METHOD_ORDER],
             edgecolor='white', linewidth=0.5)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels([METHOD_SHORT[m] for m in METHOD_ORDER])
axes[0].invert_yaxis()
axes[0].set_xlabel('Median MSE (all studies)')
axes[0].set_title('MSE Ranking', fontweight='bold')
for i, v in enumerate(all_mse.values):
    axes[0].text(v + max(all_mse.values) * 0.02, i, f'{v:.3f}',
                 va='center', fontsize=8, fontweight='bold')
axes[0].grid(axis='x', alpha=0.4)

# Right: median Coverage ranking
axes[1].barh(y_pos, all_cov.values, color=[METHOD_COLORS[m] for m in METHOD_ORDER],
             edgecolor='white', linewidth=0.5)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([METHOD_SHORT[m] for m in METHOD_ORDER])
axes[1].invert_yaxis()
axes[1].set_xlabel('Median Coverage % (all studies)')
axes[1].set_title('Coverage Ranking', fontweight='bold')
axes[1].set_xlim(0, 105)
for i, v in enumerate(all_cov.values):
    axes[1].text(v + 2, i, f'{v:.0f}%', va='center', fontsize=8, fontweight='bold')
axes[1].axvline(x=95, color='#2E7D32', linestyle='--', linewidth=1.2, alpha=0.6)
axes[1].grid(axis='x', alpha=0.4)

fig.suptitle('Overall Method Performance Ranking', fontweight='bold', fontsize=14, y=1.02)
save(fig, 'final_method_ranking.png')


print(f'\nAll 10 figures saved to {OUT_DIR}/')
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f'  {f:40s} {sz:>7.1f} KB')
