# Validation Report

**Scope.** This report records what was checked, how it was checked, and what remains
unverified. Every result below came from code that was **actually executed**. Nothing was
estimated or assumed to pass.

| Item | Value |
|---|---|
| Date | 2026-10-28 |
| Platform | macOS (Darwin 24.6), Apple Silicon |
| Interpreter | CPython 3.9.6 |
| Libraries | NumPy 2.0.2 (Apple Accelerate BLAS), SciPy 1.13.1, Matplotlib 3.9.4, pytest 8.4.2 |
| Test result | `python -m pytest`: **73 passed**, 0 failed, 0 skipped. Warnings are treated as errors, except one filtered third-party deprecation explained in §4. |
| Clean install | the repository was copied without `.venv` or outputs; `pip install ".[test]"` into a fresh venv; pytest gave **73 passed**; `exosuit-sim` ran all three bundled configs and exited with code 0 |
| Full run | `python -m exosuit_sim`: 32 sensitivity runs, 26 figure files + GIF, about 21–29 s |
| Not tested | Python 3.10–3.12, Linux, Windows |

---

## 1. Mathematically verified

These results are derived analytically, and then a test checks the code against the derivation.

| Claim | Derivation | Test(s) | Observed agreement |
|---|---|---|---|
| Unconstrained optimum, μ = 0: $\tau_a^*=\frac{w}{w+\lambda}\tau_{suit}$ | $\partial J/\partial\tau_i=0$ for the separable quadratic | `test_closed_form_shrinkage`, `test_numerical_matches_analytic_unconstrained` | SLSQP vs closed form: max \|Δ\| = 1.1×10⁻¹⁴ N·m (default scenario) |
| Box-constrained optimum, μ = 0, is the clipped closed form | per-sample 1-D convex QP; projection onto an interval | `test_numerical_matches_clipped_analytic_with_bounds` | SLSQP 3.9×10⁻⁷ N·m; L-BFGS-B and projected gradient < 10⁻⁶ N·m |
| Unconstrained optimum, μ > 0, solves $H\tau=-g$ | stationarity of a strictly convex quadratic | `test_numerical_matches_analytic_unconstrained[0.01-0.001]` | < 10⁻⁵ N·m (SLSQP stopping tolerance; condition number ≈ 160) |
| Baseline torque cancels from $J$ | $\tau_{h,s}-\tau_{h,b}=\tau_{suit}-\tau_a$ | `test_torque_balance_identity`; CSV-baseline run | swapping the baseline source leaves $J$ and the optimised profile unchanged (85.456 % in both runs) |
| Rate-limited optimum ≥ bounds-only optimum | the rate-limited problem is a restriction of the bounds-only one | `test_rate_limit_is_a_restriction_of_box_problem` | J = 0.010195 ≥ 0.010005 |
| More actuator capacity never increases the optimal $J$ | the feasible sets are nested | `test_more_actuator_capacity_never_hurts` | monotone over 7 torque ratings and 5 rate limits |
| Trapezoidal $\int\tau\,d\varphi$ is exact for constant or linear torque | the trapezoid rule is exact for linear integrands | `test_constant_torque_work_is_exact_*`, `test_linear_torque_work_matches_integral` | relative error < 10⁻¹² |
| Hysteresis-loop area → $2\int(hE+\tau_f)\,d\varphi$ as $\varphi_t\to0$ | branch state switches ±1 | `test_suit_resists_flexion_and_has_hysteresis` | within 0.1 % for φ_t = 0.01°; strictly smaller for the default 3° |
| The branch-state update does not depend on discretisation | exact solution of $ds/d\varphi$ over each increment | `test_branch_state_memory_and_discretisation_invariance` | 61 vs 1201 samples agree to 10⁻¹⁰ |
| Inverse dynamics conserve energy | $\int\boldsymbol\tau^T\dot{\mathbf q}\,dt=\Delta(KE+PE)$ | `test_inverse_dynamics_energy_balance` | relative error < 10⁻⁵ (4001 samples, moving hip and knee) |
| Static and zero-g limits of the inverse dynamics | $m_2gc_2\sin\varphi$ and $I_k\ddot\varphi$ | `test_inverse_dynamics_static_and_inertial_limits` | 10⁻¹² |
| Forward kinematics at four reference poses | hand-computed geometry | `test_ankle_position_known_configurations` | 10⁻¹² m |
| Homogeneous chain equals the planar formula | the planar case is SE(3) with z-axis joints | `test_homogeneous_chain_matches_planar_kinematics` | 10⁻¹² m over 100 random poses |

## 2. Numerically tested (properties, not closed forms)

| Property | Test(s) |
|---|---|
| Rotations preserve length, are orthonormal with det +1, and compose additively | `test_rotation_2d_*`, `test_rotation_z_*` |
| Segment lengths are preserved in 200 random poses | `test_segment_lengths_are_preserved_for_random_configurations` |
| $J$ is finite, non-negative, and equal to $\tfrac12\tau^TH\tau+g^T\tau+c$ for random profiles | `test_objective_is_finite_nonnegative_and_matches_quadratic_form` |
| The analytic gradient matches central differences | `test_gradient_matches_finite_differences` |
| The optimised profile never exceeds torque, force, or rate limits (4 actuator designs, including force-limited and one-directional) | `test_optimised_profile_never_exceeds_limits` |
| The optimum is ≤ zero assistance, ≤ 100 random feasible profiles, and ≤ the clipped ideal where that is feasible | `test_optimum_beats_feasible_alternatives` |
| Larger λ gives strictly less assistance | `test_larger_effort_penalty_reduces_assistance` |
| The sign convention holds: the suit resists flexion, the actuator assists, and actuator power ≥ 0 in pure flexion | `test_sign_convention_in_flexion` |
| Cycle: actuator negative work < 0 during extension | `test_full_pipeline_writes_all_outputs` |
| PCHIP interpolation accuracy, and refusal to extrapolate | `test_tabulated_profile_interpolation_and_bounds` |
| Invalid configuration values and unknown keys are rejected | `test_invalid_configs_raise`, `test_load_config_errors` |
| The CLI writes every artefact and returns exit codes 0 and 2 correctly | `test_full_pipeline_writes_all_outputs`, `test_cli_exit_codes` |

**Runtime checks recorded with every run** (`summary.json → validation_checks`), default scenario:

| Check | Value |
|---|---|
| SLSQP status | success in 12 iterations |
| max torque / force violation | 0 / 0 |
| max torque-rate violation | 2.0×10⁻¹¹ N·m/s |
| J optimised | 0.010195 |
| J with zero assistance | 0.351 |
| J of the best of 200 random feasible profiles | 0.226 |
| J of the bounds-only relaxation (lower bound) | 0.010005 |
| Solver comparison (bounds only, μ = 10⁻³) | SLSQP, L-BFGS-B, and projected gradient agree to J − J* ≤ 9×10⁻¹⁴ |

## 3. Dependent on simplifying assumptions (NOT validated)

These results are only as good as the assumptions behind them. No experimental validation was
performed.

* **Suit-resistance magnitudes and shape.** The law and its coefficients are illustrative, so every percentage reduction, work, force, and power value inherits that uncertainty.
* **The baseline human torque** is synthetic (two-link inverse dynamics with approximate Winter (2009) ratios). It is not measured data. It does not change the optimal profile, but it does change the reported total human torque.
* **Torque balance.** The balance assumes identical suited and unsuited kinematics, a massless and aligned exoskeleton, and no suit mass.
* **Constant moment arm.** Peak-force results are only as good as the constant-moment-arm assumption.
* **Electrical energy** is a lower bound: it omits I²R holding losses and standby power.
* **Metabolic cost.** Nothing in this project supports a claim about metabolic cost.
* **The sensitivity trends** are qualitative engineering insight under the model, not design values.

## 4. Known numerical and environmental notes

* NumPy 2.0.x linked against Apple Accelerate emits spurious divide-by-zero and overflow warnings from `@` (matrix multiplication), even when all values are finite. This was verified by comparing against `einsum`, which gave identical results. `optimization._matmul` suppresses those flags and **explicitly checks that the result is finite** instead, so a genuine overflow still raises an error.
* Matplotlib 3.9 calls pyparsing APIs that pyparsing 3.3 deprecates. The pytest configuration ignores exactly those messages. All other warnings remain errors.
* SLSQP's stopping tolerance leaves about 10⁻⁶ N·m error on the ill-conditioned μ > 0 problem. The tests use 10⁻⁵ N·m there, which is 4×10⁻⁷ of the peak torque.
