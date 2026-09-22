# IoT Predictive Maintenance System for Aircraft Engines

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor data, using the NASA C-MAPSS dataset, and turning those predictions into maintenance alerts.

## Description

Unplanned engine failures are expensive and dangerous, while replacing parts too early wastes money. This project uses sensor readings from aircraft engines to estimate how many more operating cycles each engine can run before it fails. Those estimates then feed a maintenance recommendation system that ranks engines by urgency.

The project uses the **FD001** subset of NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. It contains 100 training engines that were run until failure and 100 test engines whose data stops at some point before failure.

## Key Features

**Data exploration:** Plots sensor trends against engine life and uses correlation analysis to find the sensors most strongly linked to wear (sensors 12, 7, 21, and 20).

**Feature engineering:** Converts each engine's sensor history into 10-cycle rolling windows. For each of the 24 inputs (21 sensors and 3 operating settings), it calculates the mean, standard deviation, and latest value, giving 72 features per time step.

**Leak-free validation:** Splits the data 80/20 **by engine** rather than by row, so no engine appears in both training and validation. This gives a realistic picture of how the model performs on engines it has never seen.

**Model:** A Gradient Boosting Regressor (scikit-learn) trained on the scaled features.

**Maintenance alert system:** Maps each predicted RUL to one of four priority levels, each with a recommended action and timeline:

| Level | Meaning | Recommended action |
|---|---|---|
| 🔴 Critical | Failure imminent | Shut down and perform emergency maintenance within 24 hours |
| 🟠 High | High failure risk | Schedule urgent maintenance within 1 week |
| 🟡 Medium | Monitor closely | Maintain during next planned downtime (within 1 month) |
| 🟢 Low | Normal operation | Continue routine monitoring |

**Dashboard:** Four Matplotlib charts showing the priority distribution, predicted vs. actual RUL, a maintenance timeline, and prediction error by priority level.

## Results

Evaluated on NASA's held-out test set of 100 engines:

| Metric | Validation (20% of training engines) | Test (100 engines) |
|---|---|---|
| MAE | 32.64 cycles | **21.97 cycles** |
| R² | 0.547 | **0.470** |

The alert system sorted the 100 test engines as follows: **11 critical, 9 high, 10 medium, 70 low**, flagging 20 engines for urgent attention.

Sample predictions:

| Engine | Predicted RUL | Actual RUL | Error |
|---|---|---|---|
| 3 | 63.9 | 69 | 5.1 |
| 8 | 110.9 | 95 | -15.9 |
| 10 | 99.1 | 96 | -3.1 |
| 4 | 110.2 | 82 | -28.2 |

## Tech Stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```
3. Download the C-MAPSS dataset from the NASA Prognostics Data Repository (also available on Kaggle), and place these files in the project folder:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`
4. Open and run the notebook:
   ```bash
   jupyter notebook IotMaintenance.ipynb
   ```

## Limitations

- **Confidence scores:** The confidence level attached to each alert is currently calculated from the engine's actual RUL, which would not be known in a real deployment. The alert tiers should be treated as a proof of concept.
- **Early-life predictions:** The model tends to overestimate RUL for healthy engines, where sensor readings change very little.
- **Run-to-run variation:** The engine split is not seeded, so results may vary slightly between runs.
- **Single subset:** Only FD001 (one operating condition, one fault mode) is used so far.

## Future Improvements

- Cap training RUL values (a piecewise-linear target), a common approach for C-MAPSS that reduces early-life overestimation
- Compare against LSTM and 1D-CNN models on the raw sensor sequences
- Derive confidence from prediction uncertainty, such as quantile regression or model ensembles
- Extend to the FD002–FD004 subsets with multiple operating conditions and fault modes
- Report RMSE and the NASA scoring function for comparison with published benchmarks

## Dataset Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," *International Conference on Prognostics and Health Management*, 2008.
