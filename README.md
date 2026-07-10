# RAMBO

Official implementation for the camera-ready version of our ICML 2026 paper:

**Regime-Adaptive Bayesian Optimization via Dirichlet Process Mixtures of Gaussian Processes**  
Yan Zhang, Xuefeng Liu, Sipeng Chen, Sascha Ranftl, Chong Liu, and Shibo Li  
*Proceedings of the 43rd International Conference on Machine Learning (ICML 2026), PMLR 306, 2026.*

[Paper](https://openreview.net/pdf?id=bSUMVAYoMq) | [Official OpenReview](https://openreview.net/forum?id=bSUMVAYoMq) | [Official ICML Page](https://icml.cc/virtual/2026/poster/62957)

This repository provides the implementation of **RAMBO**, a regime-adaptive Bayesian optimization framework based on Dirichlet Process Mixtures of Gaussian Processes. The code is intended to reproduce the experiments reported in the camera-ready paper, including synthetic benchmarks and real-world optimization tasks.

## Repository Structure

* `run_baselines.py`: Unified entry point for all experiments.
* `models/`: Implementations of RAMBO (DPMM-GPR) and SGP (Single Gaussian Process) models.
* `functions/`: Core modules for molecular encoding, energy calculation, and dataset loading.
* `utils.py`: Common experiment runners, visualization utilities, and default configuration.

## Getting Started

### 1. Requirements
* Python 3.11+
* Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation

For submission anonymity and storage efficiency, the molecular dataset is provided as a compressed archive:

* Navigate to the `functions/` directory.
* Unzip `smiles_data.zip` within that folder.
* Ensure the file `functions/smiles_data.csv` is present before executing experiments.

### 3. Usage

To replicate experiments (e.g., the Molecular Torsion Energy benchmark):

```bash
python run_baselines.py --baselines RAMBO --fn torsion_energy --T 200 --N_INIT 20 --R 5 --start_R 1 --end_R 5 --resume 1
```
**Key Features:**

* **Resume Capability:** The code includes a resume option. Set `--resume 1` to pick up from the last completed round.
* **Selective Execution:** Use `--start_R` and `--end_R` to run specific independent rounds.
* **Output:** Logs, CSV results, and summary plots (mean ± std) will be saved in the `results/` directory.

## Supported Benchmarks

**Synthetic:** Levy (6D/10D), Schwefel (6D/10D)

**Real-world:** Molecular Torsion Energy, Cancer Data, ConStellaration

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{zhang2026rambo,
  title     = {Regime-Adaptive Bayesian Optimization via Dirichlet Process Mixtures of Gaussian Processes},
  author    = {Zhang, Yan and Liu, Xuefeng and Chen, Sipeng and Ranftl, Sascha and Liu, Chong and Li, Shibo},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026},
  url       = {https://openreview.net/pdf?id=bSUMVAYoMq}
}
```
