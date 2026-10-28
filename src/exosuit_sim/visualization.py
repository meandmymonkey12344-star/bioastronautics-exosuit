"""Publication-quality Matplotlib figures (PNG + PDF) and the leg animation (GIF).

Plotting only: every number shown is computed elsewhere. Colours follow one
fixed entity → colour mapping across all figures so a series never changes
colour between plots. Every figure uses one y-scale per axis (no twin axes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless, reproducible rendering

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .kinematics import leg_points  # noqa: E402
from .sensitivity import SensitivitySweep  # noqa: E402
from .simulation import ScenarioResult, SolverComparison  # noqa: E402

# --------------------------------------------------------------------------- #
# Style: fixed categorical slots (validated reference palette) and neutrals
# --------------------------------------------------------------------------- #
COLORS: Dict[str, str] = {
    "human_assisted": "#2a78d6",  # slot 1 blue — the key result
    "suit": "#eb6834",  # slot 2 orange
    "actuator": "#1baf7a",  # slot 3 aqua
    "baseline": "#eda100",  # slot 4 yellow
    "human_unassisted": "#8a8985",  # neutral grey reference (dashed)
    "limit": "#52514e",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "grid": "#e4e3df",
}
SOLVER_COLORS = {"SLSQP": "#2a78d6", "L-BFGS-B": "#eb6834", "Projected gradient": "#1baf7a"}
SEQUENTIAL_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

LABELS = {
    "baseline": "Baseline human, unsuited (synthetic)",
    "suit": "Suit resistance $\\tau_{suit}$",
    "actuator": "Exoskeleton actuator $\\tau_a$ (optimised)",
    "human_unassisted": "Human, suited, no assist",
    "human_assisted": "Human, suited + exoskeleton",
}

SIM_NOTE = "Simulated with an illustrative model — not experimental data"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": COLORS["text_secondary"],
            "axes.labelcolor": COLORS["text"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": COLORS["text_secondary"],
            "ytick.color": COLORS["text_secondary"],
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )


def _sequential(n: int) -> List[str]:
    """n evenly spaced steps from the sequential ramp (light → dark)."""
    if n == 1:
        return [SEQUENTIAL_BLUE[6]]
    idx = np.linspace(0, len(SEQUENTIAL_BLUE) - 1, n).round().astype(int)
    return [SEQUENTIAL_BLUE[i] for i in idx]


def save_figure(fig: Figure, out_dir: Path, name: str, dpi: int = 200, save_pdf: bool = True) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight")
    if save_pdf:
        paths.append(out_dir / f"{name}.pdf")
        fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _footnote(fig: Figure, extra: str = "") -> None:
    text = SIM_NOTE + (f" · {extra}" if extra else "")
    fig.text(0.01, -0.01, text, fontsize=7.5, color=COLORS["text_secondary"], ha="left", va="top")


def _torque_series(ax: plt.Axes, x: np.ndarray, r: ScenarioResult) -> None:
    ax.axhline(0, color=COLORS["text_secondary"], lw=0.8)
    ax.plot(x, r.human_unassisted_Nm, color=COLORS["human_unassisted"], ls="--", lw=1.6, label=LABELS["human_unassisted"])
    ax.plot(x, r.suit_torque_Nm, color=COLORS["suit"], label=LABELS["suit"])
    ax.plot(x, r.actuator_torque_Nm, color=COLORS["actuator"], label=LABELS["actuator"])
    ax.plot(x, r.baseline_knee_torque_Nm, color=COLORS["baseline"], lw=3.2, label=LABELS["baseline"])
    ax.plot(x, r.human_assisted_Nm, color=COLORS["human_assisted"], lw=1.6, label=LABELS["human_assisted"])
    ax.set_ylabel("Knee torque (N·m, + = flexion)")


# --------------------------------------------------------------------------- #
# Figures 1–2: torque comparison
# --------------------------------------------------------------------------- #
def plot_torque_time(r: ScenarioResult) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    _torque_series(ax, r.trajectory.time_s, r)
    ax.set_xlabel("Time (s)")
    ax.set_xlim(r.trajectory.time_s[0], r.trajectory.time_s[-1])
    ax.set_title("Knee torque contributions during 0°→90° flexion")
    ax.legend(loc="upper left", ncol=1)
    _footnote(fig, f"baseline: {r.baseline_source}")
    return fig


def plot_torque_angle(r: ScenarioResult) -> Figure:
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    _torque_series(ax, r.trajectory.knee_deg, r)
    lim = r.actuator.effective_torque_limit_Nm
    ax.axhline(lim, color=COLORS["limit"], ls=":", lw=1.2)
    ax.annotate(f"actuator limit {lim:.0f} N·m", (45, lim), xytext=(0, -12), textcoords="offset points",
                ha="center", fontsize=8, color=COLORS["text_secondary"])
    ax.set_xlabel("Knee flexion angle φ (deg)")
    ax.set_xlim(0, max(90.0, float(r.trajectory.knee_deg.max())))
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_title("Torque–angle profiles over the knee-flexion range")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _footnote(fig)
    return fig


# --------------------------------------------------------------------------- #
# Figure 3: actuator performance
# --------------------------------------------------------------------------- #
def plot_actuator_performance(r: ScenarioResult) -> Figure:
    act = r.actuator
    phi = r.trajectory.knee_deg
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    lb, ub = act.torque_bounds()

    ax = axes[0]
    ax.plot(phi, r.suit_torque_Nm, color=COLORS["suit"], lw=1.4, label=LABELS["suit"])
    ax.plot(phi, r.actuator_torque_Nm, color=COLORS["actuator"], label=LABELS["actuator"])
    ax.axhline(ub, color=COLORS["limit"], ls=":", lw=1.2, label=f"Torque limit ±{ub:.0f} N·m ({act.limiting_factor})")
    ax.set_ylabel("Torque (N·m)")
    ax.set_title("(a) Actuator joint torque")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(phi, r.actuator_force_N, color=COLORS["actuator"], label=f"Actuator force $F = \\tau_a / r$, r = {act.moment_arm_m * 100:.0f} cm")
    ax.axhline(act.max_force_N, color=COLORS["limit"], ls=":", lw=1.2, label=f"Force limit {act.max_force_N:.0f} N")
    ax.set_ylabel("Force (N)")
    ax.set_title("(b) Actuator force through constant moment arm")
    ax.legend(loc="lower right")

    ax = axes[2]
    rate = np.diff(r.actuator_torque_Nm) / r.trajectory.dt
    suit_rate = np.diff(r.suit_torque_Nm) / r.trajectory.dt
    mid = 0.5 * (phi[:-1] + phi[1:])
    ax.plot(mid, suit_rate, color=COLORS["suit"], lw=1.4, label="Suit torque rate (what exact tracking would need)")
    ax.plot(mid, rate, color=COLORS["actuator"], label="Actuator torque rate")
    if act.max_torque_rate_Nm_s is not None and r.optimized.rate_limit_enforced:
        ax.axhline(act.max_torque_rate_Nm_s, color=COLORS["limit"], ls=":", lw=1.2,
                   label=f"Rate limit ±{act.max_torque_rate_Nm_s:.0f} N·m/s")
        ax.axhline(-act.max_torque_rate_Nm_s, color=COLORS["limit"], ls=":", lw=1.2)
    ax.set_ylabel("dτ/dt (N·m/s)")
    ax.set_xlabel("Knee flexion angle φ (deg)")
    ax.set_title("(c) Torque rate")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    fig.align_ylabels(axes)
    fig.tight_layout()
    _footnote(fig, "limits are illustrative, not hardware specifications")
    return fig


# --------------------------------------------------------------------------- #
# Figure 4: convergence
# --------------------------------------------------------------------------- #
def plot_convergence(r: ScenarioResult, comparison: SolverComparison) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax = axes[0]
    hist = np.asarray(r.optimized.history)
    ax.semilogy(np.arange(hist.size), hist, color=COLORS["human_assisted"], marker="o", ms=4, lw=1.6,
                label=f"{r.optimized.method} (bounds + rate limit)")
    if r.analytic_box_Nm is not None:
        lower = r.problem.objective(r.analytic_box_Nm)
        ax.axhline(lower, color=COLORS["limit"], ls=":", lw=1.2, label="Lower bound: exact optimum without rate limit")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective J (dimensionless)")
    ax.set_title("(a) Main constrained problem")
    ax.legend(loc="upper right")

    ax = axes[1]
    for name, res in comparison.results.items():
        sub = np.maximum(np.asarray(res.history) - comparison.exact_objective, 1e-17)
        ax.loglog(np.arange(1, sub.size + 1), sub, color=SOLVER_COLORS[name], lw=1.8,
                  label=f"{name} ({res.n_iterations} it)")
    ax.set_xlabel("Iteration + 1")
    ax.set_ylabel("J − J*  (suboptimality)")
    ax.set_title("(b) Solver comparison, bounds only, μ = 10⁻³")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _footnote(fig, "J* in (b) is the lowest objective reached by any method")
    return fig


# --------------------------------------------------------------------------- #
# Figure 5: energy
# --------------------------------------------------------------------------- #
def _power_panel(ax: plt.Axes, r: ScenarioResult) -> None:
    t, p = r.trajectory.time_s, r.actuator_power_W
    ax.axhline(0, color=COLORS["text_secondary"], lw=0.8)
    if np.any(p > 0):
        ax.fill_between(t, p, 0, where=p >= 0, color=COLORS["actuator"], alpha=0.25, lw=0,
                        label="Positive work (drives motion)")
    if np.any(p < 0):
        ax.fill_between(t, p, 0, where=p < 0, color=COLORS["suit"], alpha=0.25, lw=0,
                        label="Negative work (absorbs energy)")
    ax.plot(t, p, color=COLORS["actuator"], label="Actuator power $P = \\tau_a\\dot\\varphi$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mechanical power (W)")
    ax.set_xlim(t[0], t[-1])


def plot_energy(r: ScenarioResult) -> Figure:
    from .energy import cumulative_work

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    _power_panel(axes[0], r)
    axes[0].set_title("(a) Actuator mechanical power")
    axes[0].legend(loc="upper left")

    ax = axes[1]
    phi = r.trajectory.knee_rad
    ax.plot(r.trajectory.knee_deg, cumulative_work(r.suit_torque_Nm, phi), color=COLORS["suit"],
            label="Work done against suit $\\int\\tau_{suit}d\\varphi$")
    ax.plot(r.trajectory.knee_deg, r.actuator_cumulative_work_J, color=COLORS["actuator"],
            label="Actuator work $\\int\\tau_a d\\varphi$")
    m = r.metrics
    ax.annotate(f"W⁺ = {m['actuator_positive_work_J']:.1f} J\nE_el ≈ {m['actuator_electrical_energy_estimate_J']:.1f} J "
                f"(η = {r.actuator.efficiency:.2f})", xy=(0.03, 0.62), xycoords="axes fraction", fontsize=8.5,
                color=COLORS["text"])
    ax.set_xlabel("Knee flexion angle φ (deg)")
    ax.set_ylabel("Cumulative work (J)")
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_title("(b) Cumulative mechanical work")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _footnote(fig, "mechanical work ≠ electrical energy ≠ human metabolic energy")
    return fig


# --------------------------------------------------------------------------- #
# Figure 6: kinematics
# --------------------------------------------------------------------------- #
def plot_kinematics(r: ScenarioResult, n_poses: int = 5) -> Figure:
    traj, geom = r.trajectory, r.geometry
    fig = plt.figure(figsize=(9.6, 4.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.25], hspace=0.15, wspace=0.28)
    ax = fig.add_subplot(grid[:, 0])
    targets = np.linspace(traj.knee_deg.min(), traj.knee_deg.max(), n_poses)
    idx = [int(np.argmin(np.abs(traj.knee_deg - k))) for k in targets]
    colors = _sequential(n_poses)
    ax.plot(r.ankle_xy[:, 0], r.ankle_xy[:, 1], color=COLORS["text_secondary"], ls="--", lw=1.0, label="Ankle path")
    for color, i in zip(colors, idx):
        hip, knee, ankle = leg_points(traj.hip_rad[i], traj.knee_rad[i], geom)
        xs, ys = [hip[0], knee[0], ankle[0]], [hip[1], knee[1], ankle[1]]
        ax.plot(xs, ys, color=color, lw=2.6, marker="o", ms=5, mfc="white", mec=color, mew=1.6,
                label=f"φ = {traj.knee_deg[i]:.0f}°")
    ax.set_aspect("equal")
    ax.set_xlabel("x, anterior (m)")
    ax.set_ylabel("y, superior (m)")
    ax.set_title("(a) Sagittal-plane leg poses (hip at origin)")
    ax.set_xlim(-0.55, 0.35)
    ax.set_ylim(-0.95, 0.08)
    ax.annotate("hip", (0, 0), xytext=(6, 2), textcoords="offset points", fontsize=8, color=COLORS["text_secondary"])
    ax.legend(loc="lower left", fontsize=7.5)

    series = [
        (traj.knee_deg, "φ (deg)", "(b) Knee angle, velocity, acceleration (minimum-jerk)"),
        (np.degrees(traj.knee_vel_rad_s), "φ̇ (deg/s)", None),
        (np.degrees(traj.knee_acc_rad_s2), "φ̈ (deg/s²)", None),
    ]
    axes = []
    for row, (y, label, title) in enumerate(series):
        a = fig.add_subplot(grid[row, 1], sharex=axes[0] if axes else None)
        a.plot(traj.time_s, y, color=COLORS["human_assisted"], lw=1.8)
        a.set_ylabel(label)
        if title:
            a.set_title(title)
        if row < 2:
            plt.setp(a.get_xticklabels(), visible=False)
        axes.append(a)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(traj.time_s[0], traj.time_s[-1])
    fig.align_ylabels(axes)
    _footnote(fig, f"thigh {geom.thigh_length_m:.2f} m, shank {geom.shank_length_m:.2f} m (configurable)")
    return fig


def save_leg_animation(r: ScenarioResult, path: Path, n_frames: int = 45, fps: int = 15) -> Path:
    traj, geom = r.trajectory, r.geometry
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.35)
    ax.set_ylim(-0.95, 0.08)
    ax.set_xlabel("x, anterior (m)")
    ax.set_ylabel("y, superior (m)")
    ax.plot(r.ankle_xy[:, 0], r.ankle_xy[:, 1], color=COLORS["text_secondary"], ls="--", lw=1.0)
    (line,) = ax.plot([], [], color=COLORS["human_assisted"], lw=3.0, marker="o", ms=6, mfc="white")
    label = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=8.5, color=COLORS["text"])
    frames = np.linspace(0, traj.n_samples - 1, n_frames).round().astype(int)

    def update(i: int):  # type: ignore[no-untyped-def]
        hip, knee, ankle = leg_points(traj.hip_rad[i], traj.knee_rad[i], geom)
        line.set_data([hip[0], knee[0], ankle[0]], [hip[1], knee[1], ankle[1]])
        label.set_text(
            f"t = {traj.time_s[i]:.2f} s   φ = {traj.knee_deg[i]:.0f}°\n"
            f"τ_suit = {r.suit_torque_Nm[i]:.1f} N·m   τ_a = {r.actuator_torque_Nm[i]:.1f} N·m"
        )
        return line, label

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figure 7: flexion–extension cycle (hysteresis, negative work)
# --------------------------------------------------------------------------- #
def plot_cycle(r: ScenarioResult) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    phi = r.trajectory.knee_deg
    turn = int(np.argmax(phi))

    ax = axes[0]
    ax.plot(phi[: turn + 1], r.suit_torque_Nm[: turn + 1], color=COLORS["suit"], label="Suit, flexion (loading)")
    ax.plot(phi[turn:], r.suit_torque_Nm[turn:], color=COLORS["suit"], ls="--", label="Suit, extension (unloading)")
    ax.plot(phi, r.actuator_torque_Nm, color=COLORS["actuator"], lw=1.4, label="Actuator $\\tau_a$")
    ax.fill(np.concatenate([phi[: turn + 1], phi[turn:]]),
            np.concatenate([r.suit_torque_Nm[: turn + 1], r.suit_torque_Nm[turn:]]),
            color=COLORS["suit"], alpha=0.10, lw=0)
    loss = r.metrics["work_done_against_suit_J"]
    ax.annotate(f"loop area = {loss:.1f} J\n(energy dissipated by suit)", xy=(0.50, 0.04), xycoords="axes fraction",
                fontsize=8.5)
    ax.set_xlabel("Knee flexion angle φ (deg)")
    ax.set_ylabel("Torque (N·m)")
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_title("(a) Hysteresis loop, 0°→90°→0°")
    ax.legend(loc="upper left", fontsize=7.5)

    _power_panel(axes[1], r)
    axes[1].set_title("(b) Actuator power")
    axes[1].legend(loc="lower left", fontsize=7.5)

    ax = axes[2]
    m = r.metrics
    ax.plot(r.trajectory.time_s, r.actuator_cumulative_work_J, color=COLORS["actuator"], label="Actuator net work")
    ax.axhline(m["actuator_net_work_J"], color=COLORS["limit"], ls=":", lw=1.0)
    ax.annotate(f"W⁺ = {m['actuator_positive_work_J']:.1f} J\nW⁻ = {m['actuator_negative_work_J']:.1f} J\n"
                f"W_net = {m['actuator_net_work_J']:.1f} J", xy=(0.04, 0.70), xycoords="axes fraction", fontsize=8.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative work (J)")
    ax.set_xlim(r.trajectory.time_s[0], r.trajectory.time_s[-1])
    ax.set_title("(c) Cumulative actuator work")
    fig.tight_layout()
    _footnote(fig, "hysteresis law and coefficients are illustrative")
    return fig


# --------------------------------------------------------------------------- #
# Figure 8: analytic vs numerical
# --------------------------------------------------------------------------- #
def plot_analytic_comparison(r: ScenarioResult) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    t = r.trajectory.time_s
    w = r.problem.weights
    shrink = w.w_tracking / (w.w_tracking + w.lambda_effort)
    ax = axes[0]
    ax.plot(t, r.ideal_profile_Nm, color=COLORS["suit"], lw=1.4, label="Ideal: $\\tau_a = \\tau_{suit}$")
    ax.plot(t, r.analytic_unconstrained_Nm, color=COLORS["text_secondary"], ls="--", lw=1.4,
            label=f"Analytic unconstrained: {shrink:.3f}" + r"$\,\tau_{suit}$")
    if r.analytic_box_Nm is not None:
        ax.plot(t, r.analytic_box_Nm, color=COLORS["baseline"], lw=3.0, label="Analytic with torque bounds (clipped)")
    ax.plot(t, r.actuator_torque_Nm, color=COLORS["actuator"], lw=1.8, label="Numerical, bounds + rate limit (SLSQP)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Actuator torque (N·m)")
    ax.set_xlim(t[0], t[-1])
    ax.set_title("(a) Reference solutions vs. numerical optimum")
    ax.legend(loc="upper left", fontsize=7.8)

    ax = axes[1]
    if r.analytic_box_Nm is not None:
        ax.axhline(0, color=COLORS["text_secondary"], lw=0.8)
        ax.plot(t, r.actuator_torque_Nm - r.analytic_box_Nm, color=COLORS["actuator"])
        ax.set_ylabel(r"$\tau_{a,\mathrm{numerical}} - \tau_{a,\mathrm{analytic\ box}}$ (N·m)")
        ax.set_title("(b) Effect of the rate limit on the optimum")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t[0], t[-1])
    fig.tight_layout()
    _footnote(fig, "rate limit forces the optimiser to start ramping early (non-causal, offline optimum)")
    return fig


# --------------------------------------------------------------------------- #
# Sensitivity figures
# --------------------------------------------------------------------------- #
def _fmt_value(v: float) -> str:
    return f"{v:g}"


def plot_sensitivity_sweep(sweep: SensitivitySweep) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    colors = _sequential(len(sweep.points))
    unit = "" if sweep.unit in ("–", "×") else f" {sweep.unit}"
    ax = axes[0]
    for color, p in zip(colors, sweep.points):
        ax.plot(p.knee_deg, p.actuator_torque_Nm, color=color, lw=1.6, label=f"{_fmt_value(p.value)}{unit}")
    ax.set_xlabel("Knee flexion angle φ (deg)")
    ax.set_ylabel("Optimised actuator torque (N·m)")
    ax.set_xlim(0, 90)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.set_title("(a) Optimised profiles")
    ax.legend(title=sweep.label, fontsize=7, title_fontsize=7.5, loc="upper left")

    values = np.array([p.value for p in sweep.points])
    for ax, key, ylabel, title in (
        (axes[1], "reduction_rms_additional_torque_pct", "Reduction of added human torque (% RMS)", "(b) Suit compensation"),
        (axes[2], sweep.focus_metric, sweep.focus_label, "(c) " + sweep.focus_label.split(" (")[0]),
    ):
        y = np.array([p.metrics[key] for p in sweep.points])
        ax.plot(values, y, color=COLORS["human_assisted"], lw=1.8)
        ax.scatter(values, y, c=colors, s=42, zorder=3, edgecolors="white", linewidths=1.2)
        if sweep.log_scale:
            ax.set_xscale("log")
        ax.set_xlabel(f"{sweep.label} ({sweep.unit})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[1].set_ylim(min(0.0, axes[1].get_ylim()[0]), 102)
    fig.suptitle(f"Sensitivity: {sweep.label}", x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _footnote(fig, "parameter ranges are illustrative, not flight specifications")
    return fig


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate_all_figures(
    primary: ScenarioResult,
    comparison: SolverComparison,
    out_dir: Path,
    cycle: Optional[ScenarioResult] = None,
    sweeps: Sequence[SensitivitySweep] = (),
    dpi: int = 200,
    save_pdf: bool = True,
    make_animation: bool = True,
) -> List[Path]:
    """Render every figure into ``out_dir`` and return the written paths."""
    apply_style()
    written: List[Path] = []
    figures = [
        ("fig01_torque_comparison_time", plot_torque_time(primary)),
        ("fig02_torque_vs_angle", plot_torque_angle(primary)),
        ("fig03_actuator_performance", plot_actuator_performance(primary)),
        ("fig04_optimization_convergence", plot_convergence(primary, comparison)),
        ("fig05_energy_analysis", plot_energy(primary)),
        ("fig06_leg_kinematics", plot_kinematics(primary)),
        ("fig08_analytic_vs_numerical", plot_analytic_comparison(primary)),
    ]
    if cycle is not None:
        figures.append(("fig07_flexion_extension_cycle", plot_cycle(cycle)))
    for i, sweep in enumerate(sweeps, start=1):
        figures.append((f"fig{8 + i:02d}_sensitivity_{sweep.key}", plot_sensitivity_sweep(sweep)))
    for name, fig in figures:
        written.extend(save_figure(fig, out_dir, name, dpi=dpi, save_pdf=save_pdf))
    if make_animation:
        written.append(save_leg_animation(primary, out_dir / "leg_animation.gif"))
    return written
