# Data

## `example_synthetic_knee_torque.csv` — SYNTHETIC

This file is **not** measured data. It is the quasi-static gravity knee torque
$m_2 g c_2 \sin\varphi$ for a 75 kg subject with the default anthropometry, lunar gravity
(1.62 m/s²), and the hip fixed at 0°. It was generated with
`exosuit_sim.biomechanics.quasi_static_knee_torque`. Its only purpose is to demonstrate the
tabulated-baseline code path (`config/example_csv_baseline.json`).

## Supplying your own measured baseline

To use a measured knee-torque profile:

1. Create a CSV with a header row and these columns. Lines starting with `#` are ignored.

   | column | unit | convention |
   |---|---|---|
   | `knee_flexion_deg` | degrees | anatomical flexion, strictly increasing |
   | `knee_torque_Nm` | N·m | net joint torque, **positive = flexion moment** |

2. Set the following in your config file:

   ```json
   "baseline_torque": {"source": "csv", "csv_path": "data/your_file.csv"}
   ```

3. Make sure the table spans the full knee range of the trajectory. Extrapolation is refused.

The profile is evaluated with monotone PCHIP interpolation (Fritsch & Carlson, 1980).
An angle-indexed table is only appropriate for quasi-static or monotonic movements. For a
flexion–extension cycle the same angle occurs twice and may need two different torques.

**Provenance.** Record the source of any data you add: publication, subject group, speed, and
whether the values are net moments from inverse dynamics. Do not commit data you do not have
the right to redistribute.
