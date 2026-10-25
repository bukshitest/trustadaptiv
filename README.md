# Trust-adaptive consensus coupled with Q-learning of evaluation modes — reproduction code

Requirements: Python 3.10+, numpy, scipy, pandas, matplotlib, openpyxl.

## Institutional data (Tables 2–3, Figs. 2–4)
The Uniwork KPI exports are not distributed (personal data). With the two exports placed in `real/`:

    cd real
    python -c "from load import load; a,_=load('kpi_2025.xlsx',2025); b,_=load('kpi_2026.xlsx',2026); a.to_pickle('kpi2025.pkl'); b.to_pickle('kpi2026.pkl')"
    python v65_real.py     # multi-agent methods and weighted-sum BI (Table 2), consensus trace (Fig. 3)
    python v65_extra.py    # corrupted agent, K = 3 (Table 3), weight sensitivity
    python v7_extra.py     # coordinate-wise median, K = 6 variant
    python v8_bicheb.py    # centralized BI with selection rule (6)
    python figs_real.py    # Figs. 2–4

`load.py` drops teacher names and keeps only anonymous ids.

## Synthetic experiments (Figs. 5–6, Table 4)

    cd synthetic
    python run_consensus.py && python plots.py   # Figs. 5–6
    python rl_sim.py                             # Table 4

All random seeds are fixed in the code.

## Citation
If you use this code, please cite the accompanying article (Computers, Materials & Continua, under review).

## License
MIT (see `LICENSE`).
