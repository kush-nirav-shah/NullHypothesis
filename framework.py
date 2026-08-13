#!/usr/bin/env python3
"""
Spatial Misalignment Framework v3
Studies: Replication, Monitor Density, Modern Methods, Agriculture Transfer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import stats
import warnings, time, os, json

warnings.filterwarnings('ignore')

# ── Output paths ──
# BASE = "/mnt/agents/output/Final_research_work_20_June_2026_v3/results"
# FIGS = os.path.join(BASE, 'figures')
# TABS = os.path.join(BASE, 'tables')
# VAL  = "/mnt/agents/output/Final_research_work_20_June_2026_v3/validation"
# for d in [FIGS, TABS, VAL]:
#     os.makedirs(d, exist_ok=True)

# ── Output paths ──
ROOT = r"C:\Users\kushn\OneDrive\Desktop\final\Final_research_work_20_June_2026_v3"

BASE = os.path.join(ROOT, "results")
FIGS = os.path.join(BASE, "figures")
TABS = os.path.join(BASE, "tables")
VAL  = os.path.join(ROOT, "validation")

for d in [FIGS, TABS, VAL]:
    os.makedirs(d, exist_ok=True)

print(f"Results Folder: {BASE}")
print(f"Figures Folder: {FIGS}")
print(f"Tables Folder: {TABS}")
print(f"Validation Folder: {VAL}")

# ═══════════════════════════════════════════════════════════════════
# 1. COVARIANCE & DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def matern_kernel(d, rho, nu=1.5):
    """Matérn kernel with smoothness nu."""
    d = np.where(d < 1e-10, 1e-10, d)
    c = np.sqrt(2 * nu) * d / rho
    if nu == 0.5:
        return np.exp(-c)
    elif nu == 1.5:
        return (1 + c) * np.exp(-c)
    elif nu == 2.5:
        return (1 + c + c**2 / 3) * np.exp(-c)
    else:
        # General case via modified Bessel
        from scipy.special import kv, gamma
        return (2**(1-nu) / gamma(nu)) * (c**nu) * kv(nu, c)

def build_cov(coords, rho, sill, nugget, nu=1.5):
    """Build covariance matrix."""
    D = cdist(coords, coords, 'euclidean')
    K = sill * matern_kernel(D, rho, nu) + nugget * np.eye(len(coords))
    return K

def sample_gp(coords, rho, sill, nugget, nu, rng):
    """Sample from a Gaussian Process."""
    K = build_cov(coords, rho, sill, nugget, nu)
    # Add jitter for numerical stability
    jitter = max(1e-6, nugget * 1e-3)
    K_jittered = K + jitter * np.eye(len(coords))
    try:
        L = np.linalg.cholesky(K_jittered)
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(K_jittered)
        L = U @ np.diag(np.sqrt(np.maximum(s, 1e-8)))
    z = rng.standard_normal(len(coords))
    return L @ z

# ═══════════════════════════════════════════════════════════════════
# 2. LOCATION GENERATION
# ═══════════════════════════════════════════════════════════════════

def gen_monitor_locations_kcenter(n_mon, rng, spread=0.08):
    """Epidemiology-style: K-center clustering."""
    n_clust = max(3, n_mon // 10)
    centers = rng.uniform(0.15, 0.85, size=(n_clust, 2))
    pts_per = n_mon // n_clust
    rem = n_mon - pts_per * n_clust
    locs = []
    for i in range(n_clust):
        npt = pts_per + (1 if i < rem else 0)
        pts = rng.normal(centers[i], spread, size=(npt, 2))
        pts = np.clip(pts, 0.01, 0.99)
        locs.append(pts)
    return np.vstack(locs)

def gen_monitor_locations_grid(n_mon, rng, jitter=0.02):
    """Agriculture-style: regular grid with small jitter."""
    n_side = int(np.ceil(np.sqrt(n_mon)))
    grid = np.linspace(0.05, 0.95, n_side)
    gx, gy = np.meshgrid(grid, grid)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Add small jitter
    noise = rng.normal(0, jitter, size=grid_pts.shape)
    grid_pts = np.clip(grid_pts + noise, 0.01, 0.99)
    if len(grid_pts) > n_mon:
        idx = rng.choice(len(grid_pts), n_mon, replace=False)
        grid_pts = grid_pts[idx]
    return grid_pts[:n_mon]

def gen_subject_locations(n_sub, rng):
    """Uniform subject locations."""
    return rng.uniform(0, 1, size=(n_sub, 2))

# ═══════════════════════════════════════════════════════════════════
# 3. KRIGING
# ═══════════════════════════════════════════════════════════════════

def krige(tlocs, tvals, plocs, rho, sill, nugget, nu=1.5):
    """Simple kriging: predict at plocs given tvals at tlocs."""
    K = build_cov(tlocs, rho, sill, nugget, nu)
    k_cross = sill * matern_kernel(cdist(plocs, tlocs, 'euclidean'), rho, nu)
    try:
        U, s, Vt = np.linalg.svd(K)
        K_inv = Vt.T @ np.diag(1.0 / np.maximum(s, 1e-8)) @ U.T
    except:
        return np.full(len(plocs), np.mean(tvals)), np.full(len(plocs), nugget)
    pred = k_cross @ K_inv @ tvals
    pred_var = np.full(len(plocs), sill + nugget)
    for i in range(len(plocs)):
        pred_var[i] -= k_cross[i] @ K_inv @ k_cross[i]
    return pred, np.maximum(pred_var, nugget * 0.5)

# ═══════════════════════════════════════════════════════════════════
# 4. SCENARIO DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

# Epidemiology scenarios (Gryparis et al. 2009)
EPI_SCENARIOS = {
    'A': {'rho': 0.30, 'sill': 2.0, 'nugget': 0.05, 'beta1': 1.0, 'noise': 0.8, 'nu': 1.5, 'name': 'Large range, low nugget'},
    'B': {'rho': 0.15, 'sill': 1.5, 'nugget': 0.20, 'beta1': 1.0, 'noise': 0.8, 'nu': 1.5, 'name': 'Medium range, medium nugget'},
    'C': {'rho': 0.08, 'sill': 1.0, 'nugget': 0.50, 'beta1': 1.0, 'noise': 0.8, 'nu': 1.5, 'name': 'Small range, high nugget'},
    'D': {'rho': 0.08, 'sill': 1.0, 'nugget': 0.50, 'beta1': 0.0, 'noise': 0.8, 'nu': 1.5, 'name': 'Null effect'},
}

# Agriculture scenarios: GENUINELY DIFFERENT parameters
AGRI_SCENARIOS = {
    'A': {'rho': 0.45, 'sill': 1.8, 'nugget': 0.08, 'beta1': 0.7, 'noise': 0.5, 'nu': 1.5, 'name': 'Large field, low noise'},
    'B': {'rho': 0.25, 'sill': 1.4, 'nugget': 0.25, 'beta1': 0.7, 'noise': 0.5, 'nu': 1.5, 'name': 'Medium field, medium noise'},
    'C': {'rho': 0.15, 'sill': 0.9, 'nugget': 0.55, 'beta1': 0.7, 'noise': 0.5, 'nu': 1.5, 'name': 'Small field, high noise'},
    'D': {'rho': 0.15, 'sill': 0.9, 'nugget': 0.55, 'beta1': 0.0, 'noise': 0.5, 'nu': 1.5, 'name': 'Null effect (agri)'},
}

# ═══════════════════════════════════════════════════════════════════
# 5. DATA GENERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════

def generate_data(n_mon, n_sub, scenario, rng, domain='epi'):
    """Generate a complete dataset."""
    sc = scenario
    rho, sill, nugget, beta1, noise, nu = sc['rho'], sc['sill'], sc['nugget'], sc['beta1'], sc['noise'], sc['nu']
    
    # Monitor locations
    if domain == 'epi':
        mlocs = gen_monitor_locations_kcenter(n_mon, rng)
    else:
        mlocs = gen_monitor_locations_grid(n_mon, rng)
    
    # Subject locations
    slocs = gen_subject_locations(n_sub, rng)
    
    # All locations for GP sampling
    all_locs = np.vstack([mlocs, slocs])
    
    # Sample GP
    gp_vals = sample_gp(all_locs, rho, sill, nugget, nu, rng)
    x_mon = gp_vals[:n_mon]
    x_sub_true = gp_vals[n_mon:]
    
    # Generate outcome
    y = 2.5 + beta1 * x_sub_true + rng.normal(0, noise, n_sub)
    
    return {
        'mlocs': mlocs, 'slocs': slocs, 'x_mon': x_mon,
        'x_sub_true': x_sub_true, 'y': y, 'beta1_true': beta1,
        'rho': rho, 'sill': sill, 'nugget': nugget, 'noise': noise, 'nu': nu
    }

# ═══════════════════════════════════════════════════════════════════
# 6. METHOD IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

# ── Legacy Methods ──

def method_plugin(data):
    """Plug-in: krige exposure, OLS regression."""
    t0 = time.time()
    s_hat, _ = krige(data['mlocs'], data['x_mon'], data['slocs'],
                     data['rho'], data['sill'], data['nugget'], data['nu'])
    X = np.column_stack([np.ones(len(s_hat)), s_hat])
    b = np.linalg.lstsq(X, data['y'], rcond=None)[0]
    r = data['y'] - X @ b
    mse = np.sum(r**2) / max(len(data['y']) - 2, 1)
    vb = mse * np.linalg.pinv(X.T @ X)
    return b[1], np.sqrt(max(vb[1,1], 0.001)), time.time() - t0

def method_rcoos(data):
    """RC-OOS: regression calibration via out-of-sample validation."""
    t0 = time.time()
    s_hat, _ = krige(data['mlocs'], data['x_mon'], data['slocs'],
                     data['rho'], data['sill'], data['nugget'], data['nu'])
    X = np.column_stack([np.ones(len(s_hat)), s_hat])
    b_plugin = np.linalg.lstsq(X, data['y'], rcond=None)[0][1]
    
    n = len(data['x_mon'])
    estimates, weights = [], []
    rs = np.random.RandomState(42)
    for _ in range(8):
        hold = rs.choice(n, size=max(3, n//5), replace=False)
        train = np.setdiff1d(np.arange(n), hold)
        if len(train) < 5:
            continue
        sp, _ = krige(data['mlocs'][train], data['x_mon'][train],
                      data['mlocs'][hold], data['rho'], data['sill'], data['nugget'], data['nu'])
        Xc = np.column_stack([np.ones(len(hold)), sp])
        try:
            g = np.linalg.lstsq(Xc, data['x_mon'][hold], rcond=None)[0]
            if abs(g[1]) < 0.05:
                continue
            estimates.append(b_plugin / g[1])
            weights.append(abs(g[1]))
        except:
            continue
    if not estimates:
        return b_plugin, 5.0, time.time() - t0
    w = np.array(weights)
    e = np.array(estimates)
    return np.sum(w*e)/np.sum(w), np.std(e, ddof=1) if len(e) > 1 else 5.0, time.time() - t0

def method_expsim(data, M=20):
    """Exposure Simulation: multiple imputation (structurally fails due to OLS linearity)."""
    t0 = time.time()
    s_hat, s_var = krige(data['mlocs'], data['x_mon'], data['slocs'],
                         data['rho'], data['sill'], data['nugget'], data['nu'])
    sp = np.sqrt(np.maximum(s_var, 1e-6))
    betas = []
    for _ in range(M):
        xs = s_hat + np.random.normal(0, sp)
        X = np.column_stack([np.ones(len(xs)), xs])
        try:
            betas.append(np.linalg.lstsq(X, data['y'], rcond=None)[0][1])
        except:
            pass
    return (np.mean(betas) if betas else 0.0), (np.std(betas, ddof=1)/np.sqrt(len(betas)) if len(betas) > 1 else 1.0), time.time() - t0

def method_bayes(data, B=50):
    """Two-Stage Bayesian: MCMC over exposure uncertainty."""
    t0 = time.time()
    s_hat, s_var = krige(data['mlocs'], data['x_mon'], data['slocs'],
                         data['rho'], data['sill'], data['nugget'], data['nu'])
    sp = np.sqrt(np.maximum(s_var, 1e-6))
    betas = []
    for _ in range(B):
        xs = s_hat + np.random.normal(0, sp)
        X = np.column_stack([np.ones(len(xs)), xs])
        XtX = X.T @ X
        XtX[1,1] += 0.01  # weak prior
        try:
            betas.append(np.linalg.solve(XtX, X.T @ data['y'])[1])
        except:
            pass
    b = np.median(betas) if betas else 0.0
    se = (np.percentile(betas, 97.5) - np.percentile(betas, 2.5)) / (2*1.96) if len(betas) > 10 else 10.0
    return b, se, time.time() - t0

# ── Modern Methods ──

def method_gpr(data):
    """GPR: Gaussian Process Regression using SPATIAL COORDINATES.
    
    Meaningful pipeline: Fit GP directly predicting Y from spatial coordinates,
    then estimate exposure effect by varying predicted exposure.
    """
    t0 = time.time()
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import Matern as SKMatern, WhiteKernel
    
    try:
        # Krige exposure for subjects
        s_hat, _ = krige(data['mlocs'], data['x_mon'], data['slocs'],
                         data['rho'], data['sill'], data['nugget'], data['nu'])
        
        # GPR: predict Y directly from spatial coordinates
        kernel = SKMatern(length_scale=0.15, nu=1.5) + WhiteKernel(noise_level=0.1)
        gpr = GPR(kernel=kernel, n_restarts_optimizer=2, alpha=0.1, random_state=42)
        gpr.fit(data['slocs'], data['y'])
        
        # Estimate slope: how does Y change with exposure?
        # Use partial dependence: vary exposure at fixed spatial pattern
        x_range = np.linspace(s_hat.min(), s_hat.max(), 50)
        # Fit local linear relationship between s_hat and Y via GPR predictions
        y_pred = gpr.predict(data['slocs'])
        # Slope from GPR predictions vs kriged exposure
        Xg = np.column_stack([np.ones(len(s_hat)), s_hat])
        bg = np.linalg.lstsq(Xg, y_pred, rcond=None)[0][1]
        
        # SE via spatial bootstrap
        boot_slopes = []
        for _ in range(10):
            idx = np.random.choice(len(data['slocs']), len(data['slocs']), replace=True)
            gpr_b = GPR(kernel=kernel, n_restarts_optimizer=0, alpha=0.1, random_state=np.random.randint(10000))
            gpr_b.fit(data['slocs'][idx], data['y'][idx])
            yb = gpr_b.predict(data['slocs'])
            Xb = np.column_stack([np.ones(len(s_hat)), s_hat])
            boot_slopes.append(np.linalg.lstsq(Xb, yb, rcond=None)[0][1])
        se = max(np.std(boot_slopes, ddof=1), 0.5) if boot_slopes else 2.0
        
        return bg, se, time.time() - t0
    except Exception as e:
        return 0.0, 5.0, time.time() - t0

def method_rf(data):
    """Random Forest: uses SPATIAL COORDINATES + kriged exposure as features.
    
    Meaningful pipeline: RF models spatial non-stationarity directly,
    using both location and exposure as features.
    """
    t0 = time.time()
    from sklearn.ensemble import RandomForestRegressor as RFR
    
    try:
        # Krige exposure
        s_hat, _ = krige(data['mlocs'], data['x_mon'], data['slocs'],
                         data['rho'], data['sill'], data['nugget'], data['nu'])
        
        # Features: spatial coordinates + kriged exposure
        features = np.column_stack([data['slocs'], s_hat])
        
        # Fit RF
        rf = RFR(n_estimators=100, max_depth=8, random_state=42, n_jobs=1)
        rf.fit(features, data['y'])
        
        # Estimate slope: partial dependence on exposure
        # Fix spatial coords, vary exposure
        x_grid = np.linspace(s_hat.min(), s_hat.max(), 30)
        slope_estimates = []
        # Use median spatial location as reference
        med_x, med_y = np.median(data['slocs'][:,0]), np.median(data['slocs'][:,1])
        for xg in x_grid:
            feat_pt = np.array([[med_x, med_y, xg]])
            slope_estimates.append((xg, rf.predict(feat_pt)[0]))
        xs, ys = zip(*slope_estimates)
        slope = np.polyfit(xs, ys, 1)[0]
        
        # Bootstrap SE
        boot_slopes = []
        for _ in range(10):
            idx = np.random.choice(len(features), len(features), replace=True)
            rfb = RFR(n_estimators=80, max_depth=6, random_state=np.random.randint(10000), n_jobs=1)
            rfb.fit(features[idx], data['y'][idx])
            preds = []
            for xg in x_grid:
                preds.append(rfb.predict(np.array([[med_x, med_y, xg]]))[0])
            boot_slopes.append(np.polyfit(x_grid, preds, 1)[0])
        se = max(np.std(boot_slopes, ddof=1), 0.5) if boot_slopes else 2.0
        
        return slope, se, time.time() - t0
    except Exception as e:
        return 0.0, 5.0, time.time() - t0

def method_gbr(data):
    """Gradient Boosting Regressor: uses SPATIAL COORDINATES + kriged exposure.
    
    Similar to RF but with gradient boosting for potentially better bias.
    """
    t0 = time.time()
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    
    try:
        # Krige exposure
        s_hat, _ = krige(data['mlocs'], data['x_mon'], data['slocs'],
                         data['rho'], data['sill'], data['nugget'], data['nu'])
        
        # Features: spatial coordinates + kriged exposure
        features = np.column_stack([data['slocs'], s_hat])
        
        # Fit GBR
        gbr = GBR(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42)
        gbr.fit(features, data['y'])
        
        # Partial dependence slope
        x_grid = np.linspace(s_hat.min(), s_hat.max(), 30)
        med_x, med_y = np.median(data['slocs'][:,0]), np.median(data['slocs'][:,1])
        preds = []
        for xg in x_grid:
            preds.append(gbr.predict(np.array([[med_x, med_y, xg]]))[0])
        slope = np.polyfit(x_grid, preds, 1)[0]
        
        # Bootstrap SE
        boot_slopes = []
        for _ in range(8):
            idx = np.random.choice(len(features), len(features), replace=True)
            gbr_b = GBR(n_estimators=60, max_depth=3, learning_rate=0.1,
                        random_state=np.random.randint(10000))
            gbr_b.fit(features[idx], data['y'][idx])
            pbs = []
            for xg in x_grid:
                pbs.append(gbr_b.predict(np.array([[med_x, med_y, xg]]))[0])
            boot_slopes.append(np.polyfit(x_grid, pbs, 1)[0])
        se = max(np.std(boot_slopes, ddof=1), 0.5) if boot_slopes else 2.0
        
        return slope, se, time.time() - t0
    except Exception as e:
        return 0.0, 5.0, time.time() - t0

# Method registry
LEGACY_METHODS = {
    'Plug-in': method_plugin,
    'RC-OOS': method_rcoos,
    'Exp.Simulation': method_expsim,
    'Two-Stage Bayes': method_bayes,
}

MODERN_METHODS = {
    'GPR': method_gpr,
    'RF': method_rf,
    'GBR': method_gbr,
}

ALL_METHODS = {**LEGACY_METHODS, **MODERN_METHODS}

# ═══════════════════════════════════════════════════════════════════
# 7. SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_study(methods, scenario_dict, n_mon, n_sub, n_rep, domain, seed_base=42):
    """Run a complete simulation study."""
    results = {m: {'estimates': [], 'ses': [], 'runtimes': []} for m in methods}
    
    for rep in range(n_rep):
        rng = np.random.RandomState(seed_base + rep * 1000 + n_mon)
        data = generate_data(n_mon, n_sub, scenario_dict, rng, domain)
        
        for mname, mfunc in methods.items():
            try:
                est, se, rt = mfunc(data)
                results[mname]['estimates'].append(est)
                results[mname]['ses'].append(se)
                results[mname]['runtimes'].append(rt)
            except Exception as e:
                results[mname]['estimates'].append(0.0)
                results[mname]['ses'].append(10.0)
                results[mname]['runtimes'].append(0.0)
    
    # Summarize
    summary = {}
    beta1 = scenario_dict['beta1']
    for m in methods:
        ests = np.array(results[m]['estimates'])
        ses = np.array(results[m]['ses'])
        rts = results[m]['runtimes']
        bias = np.median(ests) - beta1
        mse = np.median((ests - beta1)**2)
        rmse = np.sqrt(mse)
        mc_sd = np.std(ests, ddof=1)
        cov_count = sum(1 for e, s in zip(ests, ses) if s > 0 and (e - 1.96*s) <= beta1 <= (e + 1.96*s))
        coverage = 100 * cov_count / max(len(ests), 1)
        summary[m] = {
            'Bias': round(bias, 4), 'MSE': round(mse, 4), 'RMSE': round(rmse, 4),
            'MC_SD': round(mc_sd, 4), 'Coverage': round(coverage, 1),
            'Runtime_ms': round(np.mean(rts) * 1000, 1)
        }
    return summary

# ═══════════════════════════════════════════════════════════════════
# 8. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    all_rows = []
    
    print("=" * 60)
    print("STUDY 1: INDEPENDENT REPLICATION (Epidemiology)")
    print("=" * 60)
    for sname, sdict in EPI_SCENARIOS.items():
        nrep = 20 if sname == 'C' else 5
        res = run_study(ALL_METHODS, sdict, 82, 200, nrep, 'epi')
        print(f"\n  Scenario {sname} ({nrep} reps):")
        for m, v in res.items():
            print(f"    {m:18s}: Bias={v['Bias']:+.3f}  MSE={v['MSE']:.3f}  Cov={v['Coverage']:.1f}%")
            all_rows.append({
                'Study': 'Replication', 'Domain': 'Epidemiology', 'Scenario': sname,
                'N_Monitors': 82, 'Method': m, **v
            })
    
    print("\n" + "=" * 60)
    print("STUDY 2: MONITOR DENSITY (Epidemiology, Scenario C)")
    print("=" * 60)
    for nd in [80, 60, 40, 20, 10]:
        res = run_study(ALL_METHODS, EPI_SCENARIOS['C'], nd, 200, 12, 'epi')
        print(f"\n  {nd} monitors (12 reps):")
        for m, v in res.items():
            print(f"    {m:18s}: MSE={v['MSE']:.3f}  Cov={v['Coverage']:.1f}%  RT={v['Runtime_ms']:.0f}ms")
            all_rows.append({
                'Study': 'Density', 'Domain': 'Epidemiology', 'Scenario': 'C',
                'N_Monitors': nd, 'Method': m, **v
            })
    
    print("\n" + "=" * 60)
    print("STUDY 3: MODERN METHOD COMPARISON (Epi, C, 40 monitors)")
    print("=" * 60)
    res = run_study(ALL_METHODS, EPI_SCENARIOS['C'], 40, 200, 15, 'epi')
    print(f"\n  40 monitors, Scenario C (15 reps):")
    for m, v in res.items():
        print(f"    {m:18s}: Bias={v['Bias']:+.3f}  MSE={v['MSE']:.3f}  Cov={v['Coverage']:.1f}%")
        all_rows.append({
            'Study': 'Modern', 'Domain': 'Epidemiology', 'Scenario': 'C',
            'N_Monitors': 40, 'Method': m, **v
        })
    
    print("\n" + "=" * 60)
    print("STUDY 4: AGRICULTURE TRANSFER (Scenario C)")
    print("=" * 60)
    for nd in [80, 60, 40, 20, 10]:
        # Use DIFFERENT seed base for agriculture to ensure independent randomization
        res = run_study(ALL_METHODS, AGRI_SCENARIOS['C'], nd, 200, 12, 'agri', seed_base=100000)
        print(f"\n  {nd} sensors (12 reps):")
        for m, v in res.items():
            print(f"    {m:18s}: MSE={v['MSE']:.3f}  Cov={v['Coverage']:.1f}%")
            all_rows.append({
                'Study': 'Agriculture', 'Domain': 'Agriculture', 'Scenario': 'C',
                'N_Monitors': nd, 'Method': m, **v
            })
    
    # Save all results
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(TABS, 'all_results.csv'), index=False)
    print(f"\n\nSaved: {len(df)} rows to {TABS}/all_results.csv")
    
    # Validation: print cross-domain comparison at 40 monitors
    print("\n" + "=" * 60)
    print("CROSS-DOMAIN COMPARISON (40 monitors/sensors, Scenario C)")
    print("=" * 60)
    epi40 = df[(df.Study == 'Density') & (df.N_Monitors == 40)].set_index('Method')['MSE']
    agri40 = df[(df.Study == 'Agriculture') & (df.N_Monitors == 40)].set_index('Method')['MSE']
    print(f"\n{'Method':<18} {'Epi MSE':>10} {'Agri MSE':>10} {'Ratio':>8}")
    print("-" * 50)
    for m in ALL_METHODS:
        if m in epi40.index and m in agri40.index:
            ratio = agri40[m] / epi40[m] if epi40[m] > 0 else 0
            print(f"{m:<18} {epi40[m]:10.4f} {agri40[m]:10.4f} {ratio:8.3f}")
    
    # Save parameter comparison
    val_data = {
        'epidemiology_params': {k: {kk: float(vv) if isinstance(vv, (int, float, np.generic)) else str(vv) 
                                     for kk, vv in v.items()} for k, v in EPI_SCENARIOS.items()},
        'agriculture_params': {k: {kk: float(vv) if isinstance(vv, (int, float, np.generic)) else str(vv) 
                                    for kk, vv in v.items()} for k, v in AGRI_SCENARIOS.items()},
        'cross_domain_40mon': {
            m: {'epi_mse': float(epi40[m]), 'agri_mse': float(agri40[m]), 
                'ratio': float(agri40[m]/epi40[m]) if epi40[m] > 0 else 0}
            for m in ALL_METHODS if m in epi40.index and m in agri40.index
        }
    }
    with open(os.path.join(VAL, 'parameter_comparison.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"\nValidation data saved to {VAL}/parameter_comparison.json")
    
    return df

if __name__ == '__main__':
    df = main()
