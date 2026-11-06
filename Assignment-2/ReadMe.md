# Instructions

Requirements: Python 3 with `scipy` and `matplotlib`.
Run the files from the `Assignment-2` folder, in this order:

1. `python amrSimulation.py` – the base model of the hub: one simulation run, printed report.
2. `python amrPilotSimulation.py` – pilot experiment: an ensemble of replications without warm-up,
   plotted as average curves with 95% CIs to determine the warm-up length T0.
   Output: `pilot_raw_data.csv`, `pilot_mean_plot.png`.
3. `python replicationMethodSimulation.py` – main experiment: R = 25 independent replications
   with T0 = 120 min; prints the average, sample variance and 95% CI half-width of each measure.
   Output: `experiment_results/experiment_raw_data.csv`.
4. `python validation.py` – OFAT validation: runs the main experiment's model at 40, 45 and
   50 AMRs per hour with the same seeds.
   Output: `validation_results/ofat_raw_data.csv`.