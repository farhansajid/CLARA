# CLARA: Criticality- and Channel-Aware Resource Allocation for Healthcare 5.0 WBANs

Official repository for **CLARA** (Criticality- and Channel-Aware Resource Allocation), a hub-side scheduler for IEEE 802.15.6 scheduled-access Wireless Body Area Networks (WBANs).

Targeted for the **IEEE Journal of Biomedical and Health Informatics (JBHI)** Special Issue on *"Advanced Wireless Networks for Healthcare 5.0"*.

---

## 📌 Overview

Wireless Body Area Networks (WBANs) form the foundational sensing tier in Healthcare 5.0, streaming both life-critical alarms (e.g., cardiac arrhythmias, fall detection, hypoglycemic shock) and routine physiological streams (e.g., ECG, EEG, SpO₂, blood pressure).

### Core Challenges
1. **Dynamic Body Shadowing:** Posture transitions and limb movements introduce bursty $15\text{--}30\text{ dB}$ attenuation across on-body propagation paths (well modeled by Gilbert–Elliott dynamics).
2. **Resource Constraints & Stale CSI:** Ultralow-power sensor nodes operate on miniature coin cells. Because the central hub only samples Channel State Information (CSI) during scheduled polls or beacon intervals, standard schedulers either misallocate transmit power into deep fades or miss critical alarm deadlines.

### The CLARA Solution
CLARA provides joint decision-making in every slot:
- **Which node transmits**
- **At what transmit power** ($P \in \{-20, -10, -5, 0, +4\}\text{ dBm}$)
- **Whether to defer transmission** to avoid wasting energy during severe shadowing fades

Built on Lyapunov drift-plus-penalty optimization with virtual defect queues, packet urgency surrogates, and a strict criticality guard, CLARA protects high-priority alarm deadlines under heavy network loads while significantly reducing energy consumption.

---

## 🚀 Key Results

In extensive simulations across standard IEEE 802.15.6 CM3 channel models with Nakagami-$m$ fading and human gait dynamics:
- **36% reduction** in critical-alarm deadline misses compared to Earliest-Deadline-First with Power Control (EDF-PC, $p = 0.016$).
- **55%–82% reduction** in misses compared to standard EDF, MaxWeight (MW), Priority (PRIO), and Round-Robin (RR) policies.
- **5.5% lower energy consumption** than EDF-PC and **16%–18% lower** than fixed-power baselines.
- **Enhanced ECG Fidelity:** When replaying realistic delivery patterns on MIT-BIH Arrhythmia Database recordings under network stress, QRS-detection sensitivity increases from $96.4\%\text{--}99.3\%$ (baselines) to **$99.9\%$** with CLARA.

---

## 📁 Repository Structure

```
.
├── wban/                           # Core WBAN simulation & algorithmic modules
│   ├── channel.py                  # IEEE 802.15.6 CM3 channel model & Gilbert-Elliott fading
│   ├── common.py                   # Lookup tables, predictors, and trace utilities
│   ├── ecg_app.py                  # MIT-BIH ECG processing & QRS detection pipeline
│   ├── predictor.py                # Channel loss forecasting model & calibration
│   └── sim.py                      # Discrete-event slot-level simulator & scheduling policies
├── data/                           # Evaluation physiological datasets
│   ├── 100.dat / 100.hea / 100.atr # MIT-BIH Arrhythmia Database record 100
│   └── scipy_ecg_mitdb208.dat      # 5-minute excerpt of MIT-BIH record 208
├── make_figures.py                 # Generates visualization figures and summary tables
├── paired_tests.py                 # Computes paired Wilcoxon signed-rank tests
├── run_experiments.py              # Experiment suite driver (reproduces all paper benchmarks)
├── requirements.txt                # Python environment dependencies
└── .gitignore                      # Git ignore patterns (excluding LaTeX/PDF binaries)
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Git

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/farhansajid/CLARA.git
   cd CLARA
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔬 Reproducing Experiments

The experiment suite executes parallel Monte Carlo runs across multiple random seeds and channel configurations.

### 1. Run Complete Benchmark Suite
```bash
python run_experiments.py
```
> **Quick test mode:** To verify the pipeline with shorter simulation runs, append `--quick`:
> ```bash
> python run_experiments.py --quick
> ```
> To run specific experiment tags:
> ```bash
> python run_experiments.py --only train,pred,e1,e2,e3,e4,e2ng,ecg
> ```

### 2. Statistical Significance Testing
Compute paired Wilcoxon signed-rank test statistics across all paired seeds:
```bash
python paired_tests.py
```

### 3. Generate Evaluation Figures
```bash
python make_figures.py
```

---

## 📊 Implemented Schedulers & Baselines

| Policy | Description | Channel Aware? | Power Adaptive? | Deadline Aware? |
|---|---|:---:|:---:|:---:|
| **RR** | Round-Robin cyclic allocation | ❌ | ❌ (0 dBm) | ❌ |
| **PRIO** | Strict static priority (Alarms > Vital > Routine) | ❌ | ❌ (0 dBm) | ❌ |
| **EDF** | Earliest Deadline First | ❌ | ❌ (0 dBm) | ✅ |
| **MW** | Queue- and channel-state MaxWeight scheduler | ✅ | ❌ (0 dBm) | ❌ |
| **EDF-PC** | Earliest Deadline First with transmit power control | ✅ | ✅ | ✅ |
| **CLARA** *(Proposed)* | Drift-plus-penalty with virtual queues, urgency & guard | ✅ | ✅ | ✅ |

---

## 📚 Data Sources & Attribution

- **MIT-BIH Arrhythmia Database (Record 100):** Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001). [PhysioNet](https://physionet.org/content/mitdb/1.0.0/), Open Data Commons Attribution License (ODC-By).
- **MIT-BIH Record 208 Excerpt:** Distributed via `scipy.datasets.electrocardiogram`.

---

## 📜 Citation

If you use this codebase or scheduler in your research, please cite:

```bibtex
@article{clara2026jbhi,
  title={CLARA: Criticality- and Channel-Aware Resource Allocation for Healthcare 5.0 WBANs},
  author={Farhan, Muhammad and collaborators},
  journal={IEEE Journal of Biomedical and Health Informatics (Special Issue on Advanced Wireless Networks for Healthcare 5.0)},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
