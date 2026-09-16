# Linear Control with Decentralized Gain Learning

## Experiment

This project evaluates a linear voltage controller with decentralized gain
learning on the IEEE 33-bus radial distribution system. All electrical
quantities are expressed in per unit on a 100 MVA, 12.66 kV base, and the
voltage reference is 1 p.u. The controller is designed using the LinDistFlow
model, while the simulations use the original nonlinear DistFlow equations to
test closed-loop robustness.

The proposed controller is compared with a fixed-gain linear feedback
controller. Both use the same incremental local control law and start from the
same diagonal gain matrix. The fixed-gain controller keeps its gains unchanged;
the proposed controller adapts them from local voltage measurements.

The time-varying active-power profiles are constructed from the Caltech SoCal
distribution-grid digital-twin dataset, collected from an operational
California distribution system. The data capture heterogeneous generation and
demand associated with solar photovoltaics, data centers, electric-vehicle
charging, and cooling facilities. The profiles are normalized, aggregated, and
mapped to the 32 non-slack IEEE-33 buses. Voltage measurements and reactive-power
commands are updated once per minute.

## Code structure

- `controllers/decentralized_linear.py`: fixed-gain and decentralized
  gain-adaptation controllers.
- `models/PF_models.py`: nonlinear DistFlow simulator.
- `cases/pypower/case33/`: IEEE 33-bus network data.
- `experiments/time_varying_profiles.py`: power-profile loading, processing, and
  IEEE-33 mapping.
- `experiments/experiment_common.py`: case and controller initialization.
- `experiments/run_time_varying_controller_comparison.py`: complete experiment
  entry point and numerical-result storage.
- `experiments/plot_time_varying_dql_figs.py`: voltage and reactive-action
  figures.
- `experiments/plot_time_varying_gain_dql_figs.py`: adaptive-gain figure.
- `utils/summarize_time_varying_results.py`: CSV summary of voltage violations,
  voltage deviation, and reactive-action magnitude.

The included `time_varying_power_profile` dataset is stored under
`data/inputs/`. Place input files with the same structure in the `inputs/`
subdirectory of `--data-root` (which defaults to `data/`). The experiment reads
the local date from the input
timestamps and uses the 9am-9pm window. If a file contains several local dates,
select one with `--date YYYY-MM-DD`.

## Run

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the `time_varying_power_profile` experiment and generate all figures:

```powershell
python experiments/run_time_varying_controller_comparison.py
```

Generate the numerical comparison table:

```powershell
python utils/summarize_time_varying_results.py `
  --results outputs/results/time_varying_results.npz
```

Numerical results are saved in `outputs/results/`; PDF figures are
saved in its `figures/` subdirectory.
