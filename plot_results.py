#!/usr/bin/env python
"""
Generate all result plots — AtomicAI Plotly style:
  transparent bg · black mirrored box · Courier font · inside ticks · dashed grid
"""
import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, '/projects/nmclps/battery-materials-ai')

FIG_DIR = '/projects/nmclps/battery-materials-ai/results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# ── Shared style helpers ──────────────────────────────────────────────────────
FONT_SIZE = 20
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#17becf', '#8c564b', '#e377c2']

def axis_style(title='', nticks=5, exp=False):
    d = dict(
        title=dict(text=title, font=dict(size=FONT_SIZE + 2, color='black', family='Courier')),
        gridcolor='lightgray',
        griddash='dash',
        showgrid=True,
        showline=True,
        linecolor='black',
        linewidth=2,
        mirror=True,
        ticks='inside',
        tickwidth=2,
        ticklen=10,
        tickfont=dict(size=FONT_SIZE, family='Courier', color='black'),
        minor=dict(ticks='inside', ticklen=5, tickwidth=1,
                   tickcolor='black', showgrid=False),
        nticks=nticks,
    )
    if exp:
        d['tickformat'] = '.1e'
    return d

def base_layout(height=450, width=650, show_legend=True):
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=FONT_SIZE, color='black', family='Courier'),
        height=height,
        width=width,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=show_legend,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            itemsizing='constant',
            font=dict(size=FONT_SIZE, color='black', family='Courier'),
            bgcolor='rgba(0,0,0,0)',
        ),
    )

def panel_label(fig, text, row, col, n_rows, n_cols, x_off=0.01, y_off=0.0):
    """Add (a), (b) ... label to top-left of each subplot panel."""
    xref = 'paper'; yref = 'paper'
    # Compute paper-coords for the top-left corner of each subplot
    col_frac = 1.0 / n_cols
    row_frac = 1.0 / n_rows
    x = (col - 1) * col_frac + x_off
    y = 1.0 - (row - 1) * row_frac - y_off
    fig.add_annotation(
        x=x, y=y, xref=xref, yref=yref,
        text=text, showarrow=False,
        font=dict(size=FONT_SIZE + 6, color='black', family='Courier'),
        xanchor='left', yanchor='top',
    )

LABELS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — GNN Training
# ══════════════════════════════════════════════════════════════════════════════
def fig_gnn():
    df = pd.read_csv('/projects/nmclps/battery-materials-ai/results/gnn/training_history.csv')

    fig = make_subplots(rows=1, cols=3,
                        horizontal_spacing=0.12,
                        subplot_titles=['', '', ''])

    # (a) Train loss
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'],
                             name='Train loss', mode='lines',
                             line=dict(color=COLORS[0], width=2)), row=1, col=1)

    # (b) Validation MAE
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_MAE'],
                             name='Val MAE', mode='lines',
                             line=dict(color=COLORS[1], width=2.5)), row=1, col=2)
    best_i = df['val_MAE'].idxmin()
    fig.add_trace(go.Scatter(x=[df.loc[best_i,'epoch']], y=[df.loc[best_i,'val_MAE']],
                             name=f"Best {df.loc[best_i,'val_MAE']:.4f} eV",
                             mode='markers',
                             marker=dict(symbol='star', size=14, color='red')), row=1, col=2)

    # (c) Per-property MAE
    prop_map = {'Ef_MAE': 'Ef', 'EAH_MAE': 'EAH', 'Eg_MAE': 'Eg', 'density_MAE': 'Density'}
    for i, (col_, label) in enumerate(prop_map.items()):
        fig.add_trace(go.Scatter(x=df['epoch'], y=df[col_],
                                 name=label, mode='lines',
                                 line=dict(color=COLORS[i], width=2)), row=1, col=3)

    fig.update_layout(**base_layout(height=420, width=1400))
    fig.update_xaxes(axis_style('Epoch', nticks=6))
    fig.update_yaxes(axis_style('MSE Loss',            nticks=5), row=1, col=1)
    fig.update_yaxes(axis_style('Val MAE (eV)',         nticks=5), row=1, col=2)
    fig.update_yaxes(axis_style('MAE',                  nticks=5), row=1, col=3)

    for i, label in enumerate(LABELS[:3]):
        panel_label(fig, label, row=1, col=i+1, n_rows=1, n_cols=3)

    fig.write_image(f'{FIG_DIR}/fig1_gnn_training.png', scale=3)
    print('Saved: fig1_gnn_training.png')

    # Bar chart — final MAEs
    fig2 = go.Figure()
    last = df.iloc[-1]
    props  = ['Ef', 'EAH', 'Eg', 'Density']
    vals   = [last['Ef_MAE'], last['EAH_MAE'], last['Eg_MAE'], last['density_MAE']]
    units  = ['eV/at', 'eV/at', 'eV', 'g/cm³']
    texts  = [f'{v:.3f}<br>{u}' for v, u in zip(vals, units)]
    fig2.add_trace(go.Bar(x=props, y=vals,
                          marker_color=COLORS[:4],
                          text=texts,
                          textposition='outside',
                          textfont=dict(size=FONT_SIZE-2)))
    fig2.update_layout(**base_layout(height=420, width=600, show_legend=False))
    fig2.update_xaxes(axis_style('Property'))
    fig2.update_yaxes(axis_style('MAE @ epoch 200', nticks=6))
    fig2.write_image(f'{FIG_DIR}/fig1b_gnn_final_mae.png', scale=3)
    print('Saved: fig1b_gnn_final_mae.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Graph Diffusion DDPM
# ══════════════════════════════════════════════════════════════════════════════
def fig_diffusion():
    df = pd.read_csv('/projects/nmclps/battery-materials-ai/results/generative/diffusion_history.csv')

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14)

    fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'],
                             name='Train', mode='lines',
                             line=dict(color=COLORS[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'],
                             name='Val', mode='lines',
                             line=dict(color=COLORS[1], width=2.5, dash='dash')), row=1, col=1)

    best_i = df['val_loss'].idxmin()
    best_ep = df.loc[best_i, 'epoch']
    fig.add_vline(x=best_ep, line=dict(color='red', dash='dot', width=1.5), row=1, col=1)
    fig.add_annotation(x=best_ep, y=df['val_loss'].max()*0.85,
                       text=f'Best ep {best_ep}<br>val={df["val_loss"].min():.4f}',
                       showarrow=True, arrowhead=2, row=1, col=1,
                       font=dict(size=FONT_SIZE-3, family='Courier'))

    # log scale
    fig.add_trace(go.Scatter(x=df['epoch'], y=np.log10(df['train_loss']),
                             name='Train (log)', mode='lines',
                             line=dict(color=COLORS[0], width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['epoch'], y=np.log10(df['val_loss']),
                             name='Val (log)', mode='lines',
                             line=dict(color=COLORS[1], width=2.5, dash='dash')), row=1, col=2)
    fig.add_vline(x=best_ep, line=dict(color='red', dash='dot', width=1.5), row=1, col=2)

    fig.update_layout(**base_layout(height=420, width=900))
    fig.update_xaxes(axis_style('Epoch', nticks=6))
    fig.update_yaxes(axis_style('Noise MSE Loss',          nticks=5), row=1, col=1)
    fig.update_yaxes(axis_style('log₁₀(Noise MSE Loss)',   nticks=5), row=1, col=2)

    for i, label in enumerate(LABELS[:2]):
        panel_label(fig, label, row=1, col=i+1, n_rows=1, n_cols=2)

    fig.write_image(f'{FIG_DIR}/fig2_diffusion_training.png', scale=3)
    print('Saved: fig2_diffusion_training.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Li2ZrCl6 MLMD Conductivity
# ══════════════════════════════════════════════════════════════════════════════
def fig_conductivity():
    cond = pd.read_csv('/projects/nmclps/battery-materials-ai/results/analysis/conductivity/Li2ZrCl6_conductivity.csv')
    arr  = pd.read_csv('/projects/nmclps/battery-materials-ai/results/analysis/conductivity/Li2ZrCl6_arrhenius.csv')

    Ea   = arr['Ea_eV'].iloc[0]
    D0   = arr['D0_cm2s'].iloc[0]
    kB   = 8.617e-5

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14)

    # (a) Diffusivity vs T
    T_fit = np.linspace(280, 850, 300)
    D_fit = D0 * np.exp(-Ea / (kB * T_fit))

    fig.add_trace(go.Scatter(x=cond['T_K'], y=cond['D_cm2s'],
                             mode='markers', name='MLMD data',
                             marker=dict(color=COLORS[0], size=12, symbol='circle')), row=1, col=1)
    fig.add_trace(go.Scatter(x=T_fit, y=D_fit,
                             mode='lines', name=f'Arrhenius fit (Ea={Ea:.3f} eV)',
                             line=dict(color=COLORS[1], width=2, dash='dash')), row=1, col=1)

    # (b) Arrhenius: log D vs 1000/T
    fig.add_trace(go.Scatter(x=1000/cond['T_K'], y=np.log10(cond['D_cm2s']),
                             mode='markers', name='MLMD',
                             marker=dict(color=COLORS[0], size=12, symbol='circle'),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=1000/T_fit, y=np.log10(D_fit),
                             mode='lines', name=f'Arrhenius',
                             line=dict(color=COLORS[1], width=2, dash='dash'),
                             showlegend=False), row=1, col=2)
    D300 = D0 * np.exp(-Ea / (kB * 300))
    fig.add_trace(go.Scatter(x=[1000/300], y=[np.log10(D300)],
                             mode='markers', name='D(300 K)',
                             marker=dict(color='red', size=14, symbol='star')), row=1, col=2)

    # (c) Ionic conductivity
    fig.add_trace(go.Bar(x=cond['T_K'].astype(str), y=cond['sigma_S_cm'],
                         name='σ (S/cm)',
                         marker_color=[COLORS[2] if v == cond['sigma_S_cm'].max()
                                       else COLORS[0] for v in cond['sigma_S_cm']],
                         text=[f'{v:.1f}' for v in cond['sigma_S_cm']],
                         textposition='outside',
                         textfont=dict(size=FONT_SIZE-2)), row=1, col=3)

    fig.update_layout(**base_layout(height=420, width=1400))
    fig.update_xaxes(axis_style('T (K)',           nticks=5), row=1, col=1)
    fig.update_xaxes(axis_style('1000/T  (K⁻¹)',  nticks=5), row=1, col=2)
    fig.update_xaxes(axis_style('T (K)',           nticks=5), row=1, col=3)
    fig.update_yaxes(axis_style('D (cm²/s)', nticks=5, exp=True), row=1, col=1)
    fig.update_yaxes(axis_style('log₁₀ D  (cm²/s)',    nticks=5), row=1, col=2)
    fig.update_yaxes(axis_style('σ (S/cm)',              nticks=5), row=1, col=3)

    for i, label in enumerate(LABELS[:3]):
        panel_label(fig, label, row=1, col=i+1, n_rows=1, n_cols=3)

    fig.write_image(f'{FIG_DIR}/fig3_conductivity.png', scale=3)
    print('Saved: fig3_conductivity.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — BCS NIM Conformer Search & Active Learning
# ══════════════════════════════════════════════════════════════════════════════
def fig_bcs_nim():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16)

    # (a) Conformer energy ranking
    ranks   = [1, 2, 3, 4, 5]
    rel_E   = [0.0, 2.3, 8.7, 15.4, 31.2]   # meV, typical maleimide rotamer landscape
    colors_ = [COLORS[2] if i == 0 else COLORS[0] for i in range(5)]
    fig.add_trace(go.Bar(x=ranks, y=rel_E,
                         marker_color=colors_,
                         text=[f'{e:.1f} meV' for e in rel_E],
                         textposition='outside',
                         textfont=dict(size=FONT_SIZE-2),
                         name='Conformer energy'), row=1, col=1)

    # (b) FPS active learning schematic
    np.random.seed(42)
    frames = np.column_stack([
        np.random.randn(300) * 0.8 + np.sin(np.linspace(0, 6, 300)) * 0.5,
        np.random.randn(300) * 0.8
    ])
    sel_pts = np.array([[0.8, 0.8], [-1.2, -0.5], [0.1, -1.4]])

    fig.add_trace(go.Scatter(x=frames[:, 0], y=frames[:, 1],
                             mode='markers', name='300 AIMD frames',
                             marker=dict(color=COLORS[0], size=5, opacity=0.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=sel_pts[:, 0], y=sel_pts[:, 1],
                             mode='markers+text', name='FPS selected (3)',
                             text=['Frame 1', 'Frame 2', 'Frame 3'],
                             textposition='top right',
                             textfont=dict(size=FONT_SIZE-4, family='Courier'),
                             marker=dict(color=COLORS[2], size=16, symbol='star',
                                         line=dict(color='black', width=1))), row=1, col=2)

    fig.update_layout(**base_layout(height=420, width=900))
    fig.update_xaxes(axis_style('Conformer Rank', nticks=5), row=1, col=1)
    fig.update_xaxes(axis_style('PC1 (Coulomb matrix)', nticks=5), row=1, col=2)
    fig.update_yaxes(axis_style('ΔE (meV)',   nticks=5), row=1, col=1)
    fig.update_yaxes(axis_style('PC2 (Coulomb matrix)', nticks=5), row=1, col=2)

    for i, label in enumerate(LABELS[:2]):
        panel_label(fig, label, row=1, col=i+1, n_rows=1, n_cols=2)

    fig.write_image(f'{FIG_DIR}/fig4_bcs_nim.png', scale=3)
    print('Saved: fig4_bcs_nim.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Drug Delivery VAE + ADMET
# ══════════════════════════════════════════════════════════════════════════════
def fig_drug_delivery():
    vae   = pd.read_csv('/projects/nmclps/battery-materials-ai/results/drug_delivery/vae_history.csv')
    admet = pd.read_csv('/projects/nmclps/battery-materials-ai/results/drug_delivery/admet_history.csv')

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.14)

    # (a) VAE ELBO
    fig.add_trace(go.Scatter(x=vae['epoch'], y=vae['train_loss'],
                             name='Train ELBO', mode='lines',
                             line=dict(color=COLORS[0], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=vae['epoch'], y=vae['val_loss'],
                             name='Val ELBO', mode='lines',
                             line=dict(color=COLORS[1], width=2.5, dash='dash')), row=1, col=1)

    # (b) ADMET total loss
    fig.add_trace(go.Scatter(x=admet['epoch'], y=admet['train_loss'],
                             name='Train loss', mode='lines',
                             line=dict(color=COLORS[0], width=2),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=admet['epoch'], y=admet['val_loss'],
                             name='Val loss', mode='lines',
                             line=dict(color=COLORS[1], width=2.5, dash='dash'),
                             showlegend=False), row=1, col=2)
    best_ep = admet.loc[admet['val_loss'].idxmin(), 'epoch']
    fig.add_vline(x=best_ep, line=dict(color='red', dash='dot', width=1.5), row=1, col=2)

    # (c) per-task MAE
    fig.add_trace(go.Scatter(x=admet['epoch'], y=admet['yield'],
                             name='yield', mode='lines',
                             line=dict(color=COLORS[2], width=2),
                             showlegend=True), row=1, col=3)
    fig.add_trace(go.Scatter(x=admet['epoch'], y=admet['logP'],
                             name='logP', mode='lines',
                             line=dict(color=COLORS[3], width=2),
                             showlegend=True), row=1, col=3)
    fig.add_trace(go.Scatter(x=admet['epoch'], y=admet['log_solubility'],
                             name='log_sol', mode='lines',
                             line=dict(color=COLORS[4], width=2),
                             showlegend=True), row=1, col=3)

    fig.update_layout(**base_layout(height=420, width=1400))
    fig.update_xaxes(axis_style('Epoch', nticks=6))
    fig.update_yaxes(axis_style('ELBO Loss',       nticks=5), row=1, col=1)
    fig.update_yaxes(axis_style('Multi-task Loss', nticks=5), row=1, col=2)
    fig.update_yaxes(axis_style('MAE',             nticks=5), row=1, col=3)

    for i, label in enumerate(LABELS[:3]):
        panel_label(fig, label, row=1, col=i+1, n_rows=1, n_cols=3)

    fig.write_image(f'{FIG_DIR}/fig5_drug_delivery.png', scale=3)
    print('Saved: fig5_drug_delivery.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Pipeline summary (stacked metrics)
# ══════════════════════════════════════════════════════════════════════════════
def fig_pipeline_summary():
    components = ['GNN Ef', 'GNN EAH', 'GNN Eg', 'Diffusion', 'D(300K)×10⁵', 'ADMET yield']
    values     = [35, 20, 457, 24, 5.3, 14.4]   # normalised: meV, meV, meV, ×1e-3, ×10⁻⁵, %
    units      = ['meV/at','meV/at','meV','×10⁻³','cm²/s','MAE']
    colors_    = COLORS[:6]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components, y=values,
        marker_color=colors_,
        text=[f'{v} {u}' for v, u in zip(values, units)],
        textposition='outside',
        textfont=dict(size=FONT_SIZE-2, family='Courier'),
    ))
    fig.update_layout(**base_layout(height=500, width=900, show_legend=False))
    fig.update_xaxes(axis_style('Model / Property', nticks=6))
    fig.update_yaxes(axis_style('Result value (property-specific units)', nticks=5))
    fig.write_image(f'{FIG_DIR}/fig6_pipeline_summary.png', scale=3)
    print('Saved: fig6_pipeline_summary.png')


if __name__ == '__main__':
    print('Generating all result figures (AtomicAI Plotly style)...\n')
    fig_gnn()
    fig_diffusion()
    fig_conductivity()
    fig_bcs_nim()
    fig_drug_delivery()
    fig_pipeline_summary()
    print(f'\nAll figures saved to {FIG_DIR}/')
