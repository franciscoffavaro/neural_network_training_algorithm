# Neural Network Optimization with Genetic Algorithms

A minimal project that compares two genetic algorithms (Tournament and Roulette) to optimize a simple neural network using data from an ODS file.

## Requirements

- Python 3.8+
- `pip install -r requirements.txt`

## How to Run

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Configuration (via .env)

- DATA_FILE_PATH: path to the ODS file (last column is target)
- TRAIN_RATIO: train/test split (e.g., 0.8)
- NEURAL_HIDDEN_SIZE: hidden layer size
- GA1/GA2 parameters: population, generations, mutation/crossover, elite
- NEW_ROWS_COUNT: when > 0, auto-generate this many new rows to predict full sequence (1..15)

## Output

Three CSV files with the exact same structure as the database (Processo + 15 positions + final position):

- `outputs/base_posicao_test.csv` - actual values from the test set (all columns)
- `outputs/predictions_ga1_test.csv` - GA1 predictions (same structure, predicted values in last column)
- `outputs/predictions_ga2_test.csv` - GA2 predictions (same structure, predicted values in last column)

### Full-sequence predictions for new rows

You can generate complete position sequences (1..15) for new lines in two ways:

1) Provide `database/novas_linhas.csv` with the exact same columns as the base (Processo + 15 positions). You may leave positions empty — the system predicts missing ones sequentially.

2) Set `NEW_ROWS_COUNT` in `.env` (e.g., `NEW_ROWS_COUNT=5`) and run `python main.py`. The system will auto-generate 5 blank rows (new Processo IDs) and output predictions for all 15 positions.

## Data

- Input: ODS file at `database/dadosParaRede.ods`
- Assumption: the last column is the target. The first column is treated as an identifier if present.

## License

Educational use.
