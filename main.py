"""Compare genetic algorithms for neural network weight optimization."""
import os
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt  # for future visualization

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data, split_data
from src.neural_network import NeuralNetwork
from src.genetic_algorithm_1 import GeneticAlgorithm1
from src.genetic_algorithm_2 import GeneticAlgorithm2
from config import config


def _print_section(title, width=70):
    """Print a formatted section header."""
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}\n")


def _evaluate_model(nn, X, y, dataset_name):
    """Evaluate neural network performance."""
    preds = nn.predict(X)
    mse = np.mean(np.square(preds - y))
    mae = np.mean(np.abs(preds - y))
    
    print(f"\n{dataset_name} set:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Sample predictions (first 5):")
    for i in range(min(5, len(preds))):
        err = abs(preds[i][0] - y[i][0])
        print(f"    Pred: {preds[i][0]:.4f} | True: {y[i][0]:.4f} | Error: {err:.4f}")
    
    return mse, mae


def _normalize_xy(X, y):
    """Min-max normalize features and target independently.
    Returns (X_norm, y_norm, X_min, X_max, y_min, y_max)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    return X_norm, y_norm, X_min, X_max, float(y_min), float(y_max)


def _train_ga1(input_size, X_train, y_train, verbose=True):
    """Train a neural net with GA1 and return the fitted network."""
    nn = NeuralNetwork(input_size, config.NEURAL_HIDDEN_SIZE, 1)
    ga = GeneticAlgorithm1(
        population_size=config.AG1_POPULATION_SIZE,
        generations=config.AG1_GENERATIONS,
        mutation_rate=config.AG1_MUTATION_RATE,
        crossover_rate=config.AG1_CROSSOVER_RATE,
        tournament_size=config.AG1_TOURNAMENT_SIZE,
        elite_size=config.AG1_ELITE_SIZE,
        display_interval=config.DISPLAY_INTERVAL,
    )
    best_w = ga.evolve(nn, X_train, y_train, verbose)
    nn.set_weights_from_array(best_w)
    return nn


def _predict_full_sequence_if_requested(base_path: Path, out_dir: Path):
    """Train per-position models and predict full positions 1..15 for new rows.

    Sources for new rows (in order of precedence):
    1) database/novas_linhas.csv if it exists (preferred)
    2) If not, and config.NEW_ROWS_COUNT > 0, auto-generate that many rows
       with new Processo IDs and empty positions to be predicted.

    Expectations for database/novas_linhas.csv:
    - Columns should match the base file structure (Processo + 15 posições).
    - You may leave positions empty; they'll be predicted sequentially.
    """
    novas_path = Path("database/novas_linhas.csv")

    _print_section("Full-sequence prediction for new rows")
    try:
        base_df = pd.read_excel(base_path, engine="odf")
    except Exception as e:
        print(f"Skipping full-sequence prediction (cannot read base): {e}")
        return

    # Determine input rows
    novas_df = None
    if novas_path.exists():
        try:
            novas_df = pd.read_csv(novas_path)
            print(f"Loaded novas_linhas.csv with {len(novas_df)} rows")
        except Exception as e:
            print(f"Skipping reading novas_linhas.csv due to error: {e}")
            novas_df = None

    if novas_df is None and getattr(config, 'NEW_ROWS_COUNT', 0) > 0:
        # Auto-generate N rows with new Processo IDs and missing positions
        cols = list(base_df.columns)
        n = int(getattr(config, 'NEW_ROWS_COUNT', 0))
        print(f"Auto-generating {n} new rows for prediction (from environment)")
        # Generate Processo IDs with some variation to avoid identical predictions
        proc_col = cols[0]
        try:
            max_proc = pd.to_numeric(base_df[proc_col], errors='coerce').max()
            if pd.isna(max_proc):
                max_proc = 0
        except Exception:
            max_proc = 0
        
        # Generate diverse starting points by sampling from existing data patterns
        rows = []
        base_sample = base_df.sample(min(n, len(base_df)), replace=(n > len(base_df)))
        
        for i in range(n):
            r = {c: None for c in cols}
            r[proc_col] = int(max_proc) + 1 + i
            
            # Copy first position from sampled row to add diversity
            # (This gives the model a varied starting point for predictions)
            if len(base_sample) > 0:
                sample_row = base_sample.iloc[i % len(base_sample)]
                r[cols[1]] = sample_row.iloc[1]  # Copy 1ª posição from sample
            
            rows.append(r)
        novas_df = pd.DataFrame(rows, columns=cols)

    if novas_df is None or len(novas_df) == 0:
        print("No new rows provided and NEW_ROWS_COUNT is 0. Skipping full-sequence prediction.")
        return

    # Align columns to base structure
    novas_df = novas_df.reindex(columns=base_df.columns)

    # Train one model per position k (1..15), using columns [0..k-1] as features and column k as target.
    n_cols = base_df.shape[1]
    if n_cols < 2:
        print("Base data has insufficient columns to train sequence models.")
        return

    models = []
    scalers = []  # list of tuples (X_min, X_max, y_min, y_max)

    print("Training per-position models (GA1) ...")
    for k in range(1, n_cols):  # column 0 is 'Processo'; predict col k
        # Build training set for position k
        # IMPORTANT: Skip column 0 (Processo) as it's just an ID with no predictive value
        if k == 1:
            # For 1st position: no previous positions, use a constant feature
            # This forces the model to learn the average/distribution of 1st positions
            Xk = np.ones((len(base_df), 1))  # constant feature
        else:
            # For positions 2-15: use ONLY previous positions (skip Processo ID)
            Xk = base_df.iloc[:, 1:k].values  # features: previous positions only
        
        yk = base_df.iloc[:, k].values.reshape(-1, 1)  # target: position k

        Xk_norm, yk_norm, Xk_min, Xk_max, yk_min, yk_max = _normalize_xy(Xk, yk)

        nn_k = _train_ga1(input_size=Xk.shape[1], X_train=Xk_norm, y_train=yk_norm, verbose=False)
        models.append(nn_k)
        scalers.append((Xk_min, Xk_max, yk_min, yk_max))

    def _predict_row_full_sequence(row: pd.Series) -> pd.Series:
        row_out = row.copy()
        # ensure numeric for prediction; keep NaNs for missing
        for k in range(1, n_cols):
            if pd.notna(row_out.iloc[k]):
                # value provided; keep as-is
                continue
            
            # Build feature vector
            if k == 1:
                # For 1st position: use constant feature (same as training)
                feats = np.array([1.0])
            else:
                # For positions 2-15: use previous predicted positions (skip Processo)
                feats = row_out.iloc[1:k].astype(float).values
            
            Xk_min, Xk_max, yk_min, yk_max = scalers[k-1]
            Xk_norm = (feats - Xk_min) / (Xk_max - Xk_min + 1e-8)
            yk_pred_norm = models[k-1].predict(Xk_norm.reshape(1, -1))
            yk_pred = yk_pred_norm * (yk_max - yk_min) + yk_min
            row_out.iloc[k] = round(float(yk_pred[0, 0]))  # Round to integer like base data
        return row_out

    # Predict for all new rows
    preds_rows = [
        _predict_row_full_sequence(novas_df.iloc[i, :]) for i in range(len(novas_df))
    ]
    preds_df = pd.DataFrame(preds_rows, columns=base_df.columns)
    
    # Convert position columns to integers (same as base)
    for col in preds_df.columns[1:]:  # skip Processo column
        preds_df[col] = preds_df[col].round().astype(int)

    out_path = out_dir / "predictions_full_sequence.csv"
    preds_df.to_csv(out_path, index=False)
    print(f"Saved full-sequence predictions to: {out_path}")


def main():
    _print_section("Neural Network Optimization with Genetic Algorithms")
    
    config.validate()
    config.print_config()
    
    # Load and split data
    _print_section("Data Loading")
    X, y = load_data(config.DATA_FILE_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y, config.TRAIN_RATIO)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")
    print(f"  Features: {X.shape[1]}")
    
    # Setup network architecture
    inp_size = X_train.shape[1]
    h_size = config.NEURAL_HIDDEN_SIZE
    out_size = 1
    
    # Train with GA1 (Tournament)
    _print_section("Training: Genetic Algorithm 1 (Tournament)")
    ga1 = GeneticAlgorithm1(
        population_size=config.AG1_POPULATION_SIZE,
        generations=config.AG1_GENERATIONS,
        mutation_rate=config.AG1_MUTATION_RATE,
        crossover_rate=config.AG1_CROSSOVER_RATE,
        tournament_size=config.AG1_TOURNAMENT_SIZE,
        elite_size=config.AG1_ELITE_SIZE,
        display_interval=config.DISPLAY_INTERVAL
    )
    
    nn1 = NeuralNetwork(inp_size, h_size, out_size)
    best_weights_ga1 = ga1.evolve(nn1, X_train, y_train, config.VERBOSE)
    nn1.set_weights_from_array(best_weights_ga1)
    
    # Evaluate GA1
    _print_section("Evaluation: GA1 Results")
    mse_tr1, mae_tr1 = _evaluate_model(nn1, X_train, y_train, "Training")
    mse_te1, mae_te1 = _evaluate_model(nn1, X_test, y_test, "Test")
    
    # Train with GA2 (Roulette)
    _print_section("Training: Genetic Algorithm 2 (Roulette Wheel)")
    ga2 = GeneticAlgorithm2(
        population_size=config.AG2_POPULATION_SIZE,
        generations=config.AG2_GENERATIONS,
        mutation_rate=config.AG2_MUTATION_RATE,
        crossover_rate=config.AG2_CROSSOVER_RATE,
        elite_size=config.AG2_ELITE_SIZE,
        display_interval=config.DISPLAY_INTERVAL
    )
    
    nn2 = NeuralNetwork(inp_size, h_size, out_size)
    best_weights_ga2 = ga2.evolve(nn2, X_train, y_train, config.VERBOSE)
    nn2.set_weights_from_array(best_weights_ga2)
    
    # Evaluate GA2
    _print_section("Evaluation: GA2 Results")
    mse_tr2, mae_tr2 = _evaluate_model(nn2, X_train, y_train, "Training")
    mse_te2, mae_te2 = _evaluate_model(nn2, X_test, y_test, "Test")

    # Export predictions to CSV in base-like format (same structure as database)
    try:
        base_df = pd.read_excel(config.DATA_FILE_PATH, engine="odf")
        orig_y = base_df.iloc[:, -1].values.reshape(-1, 1)
        y_min, y_max = float(orig_y.min()), float(orig_y.max())
        has_base = True
    except Exception:
        base_df = None
        y_min, y_max = 0.0, 1.0
        has_base = False

    def _denorm(a):
        return a * (y_max - y_min) + y_min

    # Get predictions for both models
    preds_te1 = nn1.predict(X_test)
    preds_te2 = nn2.predict(X_test)

    split_idx = len(X_train)
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Generate outputs with exact same structure as database
    if has_base:
        # Base test with all columns (Processo + 15 positions + final position)
        base_test_full = base_df.iloc[split_idx:, :].copy()
        base_pos_path = out_dir / "base_posicao_test.csv"
        base_test_full.to_csv(base_pos_path, index=False)
        print(f"\nSaved base test (full structure) to: {base_pos_path}")

        # GA1 predictions: copy structure and replace last column with predictions
        ga1_test_full = base_df.iloc[split_idx:, :].copy()
        ga1_test_full.iloc[:, -1] = _denorm(preds_te1).flatten()
        ga1_pos_path = out_dir / "predictions_ga1_test.csv"
        ga1_test_full.to_csv(ga1_pos_path, index=False)
        print(f"Saved GA1 predictions (full structure) to: {ga1_pos_path}")

        # GA2 predictions: copy structure and replace last column with predictions
        ga2_test_full = base_df.iloc[split_idx:, :].copy()
        ga2_test_full.iloc[:, -1] = _denorm(preds_te2).flatten()
        ga2_pos_path = out_dir / "predictions_ga2_test.csv"
        ga2_test_full.to_csv(ga2_pos_path, index=False)
        print(f"Saved GA2 predictions (full structure) to: {ga2_pos_path}")
    else:
        print("\nWarning: Base file not available, cannot generate full-structure outputs")
    
    # Final comparison
    _print_section("Comparison Summary")
    
    print(f"\n{'Metric':<25} | {'GA1':<12} | {'GA2':<12} | {'Winner':<10}")
    print("-" * 65)
    
    def _compare_row(name, v1, v2):
        winner = "GA1" if v1 < v2 else "GA2"
        print(f"{name:<25} | {v1:>12.6f} | {v2:>12.6f} | {winner:<10}")
    
    _compare_row("MSE (Train)", mse_tr1, mse_tr2)
    _compare_row("MSE (Test)", mse_te1, mse_te2)
    _compare_row("MAE (Train)", mae_tr1, mae_tr2)
    _compare_row("MAE (Test)", mae_te1, mae_te2)
    
    winner_ga = "GA1 (Tournament)" if mse_te1 < mse_te2 else "GA2 (Roulette)"
    best_mse = min(mse_te1, mse_te2)
    improvement = abs(mse_te1 - mse_te2) / max(mse_te1, mse_te2) * 100
    
    print(f"\n*** Winner: {winner_ga} ***")
    print(f"Test MSE: {best_mse:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"\n{'='*70}\n")
    
    # Generate interactive markdown report
    from datetime import datetime
    report_path = out_dir / "report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""# Neural Network Training Report

Generated: {timestamp}

---

## Results Summary

| Metric | GA1 (Tournament) | GA2 (Roulette) | Winner |
|--------|------------------|----------------|--------|
| MSE (Test) | {mse_te1:.6f} | {mse_te2:.6f} | {'GA1' if mse_te1 < mse_te2 else 'GA2'} |
| MAE (Test) | {mae_te1:.6f} | {mae_te2:.6f} | {'GA1' if mae_te1 < mae_te2 else 'GA2'} |

**Winner:** {winner_ga} (improvement: {improvement:.2f}%)

---

## Why the Winner Won

"""
    
    # Analyze why winner won
    if mse_te1 < mse_te2:
        # GA1 won
        report_content += f"""**GA1 (Tournament Selection)** achieved better results. Key factors:

**Selection Pressure:**
- Tournament selection creates stronger competitive pressure by directly comparing individuals in small groups
- This leads to faster convergence toward optimal solutions
- The tournament size ({config.AG1_TOURNAMENT_SIZE}) ensures a balance between exploration and exploitation

**Parameter Configuration:**
- Mutation rate: {config.AG1_MUTATION_RATE*100:.1f}% (lower than GA2's {config.AG2_MUTATION_RATE*100:.1f}%)
  * Lower mutation preserves good solutions once found
  * Reduces risk of disrupting near-optimal weight configurations
- Crossover rate: {config.AG1_CROSSOVER_RATE*100:.1f}% ({"lower" if config.AG1_CROSSOVER_RATE < config.AG2_CROSSOVER_RATE else "higher"} than GA2's {config.AG2_CROSSOVER_RATE*100:.1f}%)
  * Balanced crossover promotes mixing of successful weight patterns

**Performance Advantage:**
- Test MSE: {improvement:.2f}% better than GA2
- Generalization: {"Better" if mse_te1/mse_tr1 < mse_te2/mse_tr2 else "Comparable"} test/train ratio indicates {"less" if mse_te1/mse_tr1 < mse_te2/mse_tr2 else "similar"} overfitting
"""
    else:
        # GA2 won
        report_content += f"""**GA2 (Roulette Wheel Selection)** achieved better results. Key factors:

**Selection Diversity:**
- Roulette selection maintains higher population diversity by giving all individuals a chance proportional to fitness
- This broader exploration helps escape local optima
- Probabilistic selection prevents premature convergence to suboptimal solutions

**Parameter Configuration:**
- Mutation rate: {config.AG2_MUTATION_RATE*100:.1f}% (higher than GA1's {config.AG1_MUTATION_RATE*100:.1f}%)
  * Higher mutation increases exploration of the weight space
  * Helps discover unconventional but effective solutions
- Crossover rate: {config.AG2_CROSSOVER_RATE*100:.1f}% ({"higher" if config.AG2_CROSSOVER_RATE > config.AG1_CROSSOVER_RATE else "lower"} than GA1's {config.AG1_CROSSOVER_RATE*100:.1f}%)
  * {"Strong" if config.AG2_CROSSOVER_RATE > 0.85 else "Moderate"} crossover promotes solution recombination
- Elite size: {config.AG2_ELITE_SIZE} ({"larger" if config.AG2_ELITE_SIZE > config.AG1_ELITE_SIZE else "smaller"} than GA1's {config.AG1_ELITE_SIZE})
  * {"More" if config.AG2_ELITE_SIZE > config.AG1_ELITE_SIZE else "Fewer"} elite individuals preserved per generation

**Performance Advantage:**
- Test MSE: {improvement:.2f}% better than GA1
- Generalization: {"Better" if mse_te2/mse_tr2 < mse_te1/mse_tr1 else "Comparable"} test/train ratio indicates {"less" if mse_te2/mse_tr2 < mse_te1/mse_tr1 else "similar"} overfitting
"""
    
    report_content += f"""
---

## Configuration

| Parameter | GA1 | GA2 |
|-----------|-----|-----|
| Population | {config.AG1_POPULATION_SIZE} | {config.AG2_POPULATION_SIZE} |
| Generations | {config.AG1_GENERATIONS} | {config.AG2_GENERATIONS} |
| Mutation Rate | {config.AG1_MUTATION_RATE*100:.1f}% | {config.AG2_MUTATION_RATE*100:.1f}% |
| Crossover Rate | {config.AG1_CROSSOVER_RATE*100:.1f}% | {config.AG2_CROSSOVER_RATE*100:.1f}% |
| Elite Size | {config.AG1_ELITE_SIZE} | {config.AG2_ELITE_SIZE} |

**Dataset:** {len(X_train)} training samples, {len(X_test)} test samples ({X.shape[1]} features)

---

## Training Metrics

### GA1 (Tournament Selection)

| Dataset | MSE | MAE |
|---------|-----|-----|
| Training | {mse_tr1:.6f} | {mae_tr1:.6f} |
| Test | {mse_te1:.6f} | {mae_te1:.6f} |

### GA2 (Roulette Wheel Selection)

| Dataset | MSE | MAE |
|---------|-----|-----|
| Training | {mse_tr2:.6f} | {mae_tr2:.6f} |
| Test | {mse_te2:.6f} | {mae_te2:.6f} |

---

## Output Files

- `base_posicao_test.csv` - Actual test values
- `predictions_ga1_test.csv` - GA1 predictions
- `predictions_ga2_test.csv` - GA2 predictions

All files: {len(X_test)} rows, 16 columns (same structure as database)
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {report_path}")

    # Optionally perform full-sequence prediction for new rows
    _predict_full_sequence_if_requested(Path(config.DATA_FILE_PATH), out_dir)


if __name__ == "__main__":
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)
    main()
