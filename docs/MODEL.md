# Scientific Design (Phase 1)

This document fixes the conventions, equations, and assumptions **before** any code
is written. Every module in `src/exosuit_sim/` implements one section of this file,
and the tests in `tests/` check the equations stated here.

> **Status of the model.** Everything here is a *simplified engineering model*. The
> spacesuit resistance law, its coefficients, and the actuator limits are
> **illustrative**. They are not fitted to measured suit data and are not flight
> specifications. The baseline human torque comes from rigid-body inverse dynamics
> with textbook anthropometric ratios. It is **not** measured joint-torque data.

---

## 1. Ambiguities in the original brief, and how they are resolved

| # | Issue in the brief | Resolution |
|---|---|---|
| 1 | The forward-kinematics formula $R(\theta_1)[L_1,0]^T + R(\theta_1+\theta_2)[L_2,0]^T$ puts each segment along its local **x**-axis. In a standard "x forward, y up" frame, that makes the neutral leg point forward, not down. | The formula is kept exactly but stated in a **leg frame** whose x-axis points down the neutral standing leg. The leg frame is the global frame rotated by $-90^\circ$ about $z$. Both frames are implemented and tested. |
| 2 | "Knee-relative angle $\theta_2$" carries no sign. With counter-clockwise (CCW) rotation positive (viewed from the right side, facing +z), anatomical knee flexion swings the shank **backward**, which is a **negative** CCW rotation. | Link angle $\theta_2 = -\varphi$, where $\varphi \ge 0$ is the anatomical knee flexion. Hip flexion $q_h = \theta_1$ (CCW, thigh swings forward). |
| 3 | The torque-balance equation had no stated sign convention. | All torques below are **scalar torques about the knee axis, positive in the flexion direction**, except $\tau_{suit}$, which is defined as a resistance (Section 4). |
| 4 | With the cost $J = \sum w(\tau_{h,s}-\tau_{h,b})^2 + \lambda\sum \tau_a^2$ the optimum is **not** $\tau_a = \tau_{suit}$. | The exact unconstrained minimiser is $\tau_a^\* = \frac{w}{w+\lambda}\tau_{suit}$. Ideal compensation is only the $\lambda \to 0$ limit (Section 6.2). |
| 5 | The baseline human torque **cancels** out of that objective, because $\tau_{h,s}-\tau_{h,b} = \tau_{suit}-\tau_a$. | Stated explicitly. The baseline still matters for reporting total human torque and for any future objective that penalises total human effort. It does not affect the optimal actuator profile under this objective. |
| 6 | $\tau = k\theta^2$ has zero torque **and** zero stiffness at $\theta=0$. It has no direction dependence, and the brief's "preload" would push a relaxed leg into hyperextension. | Elastic part $k_1\varphi + k_2\varphi\lvert\varphi\rvert$ (restoring, zero at the neutral angle). Plus a **rate-independent hysteresis** term with memory (a Dahl-type branch variable) and a direction-dependent friction (breakaway) term in place of a static preload. With $k_1 = \tau_f = h = 0$ the model reduces exactly to $k\theta^2$. |
| 7 | "EVA" means microgravity on the ISS but partial gravity on the Moon. That changes the baseline torque completely. | Gravity is a configuration parameter. The default is lunar surface gravity, $1.62\ \mathrm{m/s^2}$, the case where suited knee flexion matters most for walking and kneeling. $g=0$ gives an ISS-like case in which only inertia remains. |
| 8 | `init.py` in the requested tree. | It is `__init__.py`; Markdown removed the underscores. |
| 9 | Python 3.11 was requested. | The only interpreter on the build machine is CPython 3.9.6. The code targets **Python ≥ 3.9** (`from __future__ import annotations`, no 3.10+ syntax) and was executed and tested on 3.9.6. It is expected to run unchanged on 3.11 and 3.12, but that has not been tested here. |

---

## 2. Coordinate frames and kinematics

**Global frame $\{G\}$:** origin at the hip joint centre, $x$ anterior (forward), $y$ superior
(up), $z = x \times y$ lateral (out of the sagittal plane toward the viewer). Positive
rotations are CCW about $+z$.

**Leg frame $\{L\}$:** $\{G\}$ rotated by $-90^\circ$ about $z$, so $x_L$ points down the
neutral leg and $y_L$ points anteriorly: $\;{}^{G}p = R(-\pi/2)\,{}^{L}p$.

**Joint variables**

| symbol | meaning | sign | configurable |
|---|---|---|---|
| $q_h = \theta_1$ | hip flexion, thigh angle from the downward vertical | + = thigh forward (CCW) | start/end values |
| $\varphi$ | anatomical knee flexion | + = shank folds backward | start/end values (default 0° → 90°) |
| $\theta_2 = -\varphi$ | knee link angle (relative, CCW) | — | derived |
| $L_1, L_2$ | thigh and shank lengths | — | yes (0.45 m, 0.43 m) |

**Planar forward kinematics** (the user's equation, in $\{L\}$):

$$
{}^{L}P_{ankle} = R(\theta_1)\begin{bmatrix}L_1\\0\end{bmatrix} + R(\theta_1+\theta_2)\begin{bmatrix}L_2\\0\end{bmatrix},
\qquad R(\alpha)=\begin{bmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{bmatrix}.
$$

**Homogeneous chain (3-D ready)**

$$
{}^{G}T_{ankle} = R_z(-\tfrac{\pi}{2})\; R_z(\theta_1)\; T_x(L_1)\; R_z(\theta_2)\; T_x(L_2)
$$

Each factor is a $4\times4$ matrix in $SE(3)$. The planar model is the special case
in which every joint axis is $z$. A 3-D leg replaces $R_z(\theta_1)$ with a hip
ball joint, e.g. $R_z(\text{flex})R_x(\text{abd})R_y(\text{rot})$, and adds tibial
rotation, without changing the chain structure.

**Reference configurations (unit tests):**

| hip | knee flexion | ankle in $\{G\}$ (m) |
|---|---|---|
| 0° | 0° | $(0,\,-0.88)$ |
| 90° | 0° | $(0.88,\,0)$ |
| 0° | 90° | $(-0.43,\,-0.45)$ |
| 90° | 90° | $(0.45,\,-0.43)$ |

**Movement trajectory:** minimum-jerk interpolation (Flash & Hogan, 1985), with
$s(\sigma) = 10\sigma^3 - 15\sigma^4 + 6\sigma^5$ and $\sigma = t/T$. The angular
velocity and acceleration are analytic, and both are zero at the start and end of
each phase. Two modes are available. `flexion` runs 0° → 90°. `flexion_extension`
runs 0° → 90° → 0°, which exercises hysteresis and negative work.

---

## 3. Baseline (unsuited) human joint torque

### Four torques that are easy to confuse

* **Joint torque, also called net joint moment:** the *resultant* moment about the joint
  axis that all structures crossing the joint must supply to produce the observed
  motion. Inverse dynamics gives exactly this quantity.
* **Muscle-generated torque:** the moment produced by muscle–tendon forces alone. Because
  agonists and antagonists co-contract, muscle torques can be much larger than the net
  joint torque. Estimating them needs a musculoskeletal model, which is **not implemented**.
* **Net torque the human must supply in the suit**, $\tau_{h,s}$: the net joint torque
  that the *biological* joint must still produce after the suit and the exoskeleton act.
* **Required actuator torque**, $\tau_a$: the moment the exoskeleton applies about the
  knee axis. It is what the optimiser chooses.

### Synthetic reference: two-link inverse dynamics (default)

Link 1 is the thigh. Link 2 is the shank with the foot lumped as a point mass at the
ankle. Segment parameters are approximately those in Winter (2009), *Biomechanics and
Motor Control of Human Movement*: mass fraction, centre-of-mass fraction, and radius of
gyration about the COM. The defaults should be checked against that table before they
are relied on, and all of them can be changed. With link angles $\theta$ measured from
the downward vertical and $c_i$ the COM distance:

$$
\boldsymbol\tau = M(\theta)\ddot\theta + C(\theta,\dot\theta) + G(\theta)
$$

$$
\begin{aligned}
M_{11} &= I_1 + I_2 + m_1c_1^2 + m_2(L_1^2 + c_2^2 + 2L_1c_2\cos\theta_2)\\
M_{12} &= M_{21} = I_2 + m_2(c_2^2 + L_1c_2\cos\theta_2), \qquad M_{22} = I_2 + m_2c_2^2\\
C_1 &= -m_2L_1c_2\sin\theta_2\,(2\dot\theta_1\dot\theta_2 + \dot\theta_2^2), \qquad
C_2 = m_2L_1c_2\sin\theta_2\,\dot\theta_1^2\\
G_1 &= g\,[\,(m_1c_1 + m_2L_1)\sin\theta_1 + m_2c_2\sin(\theta_1+\theta_2)\,], \qquad
G_2 = g\,m_2c_2\sin(\theta_1+\theta_2)
\end{aligned}
$$

The knee flexion torque is $\tau_{h,b} = -\tau_2$. For a fixed hip at $q_h = 0$ it
reduces to $\tau_{h,b} = I_k\ddot\varphi + m_2gc_2\sin\varphi$: flexing against gravity
needs flexor torque. A unit test checks the energy balance
$\int \boldsymbol\tau^T\dot{\boldsymbol\theta}\,dt = \Delta(KE + PE)$.

**Limitations of this synthetic baseline:** rigid segments, no soft-tissue motion, no
ground contact (an open-chain leg swing), no suit mass, and no muscle co-contraction.
It is a *physics-based synthetic curve*, not a physiological measurement.

### Measured data (optional)

Set `baseline_torque.source = "csv"` and point `csv_path` at a file with the columns
`knee_flexion_deg, knee_torque_Nm`. The profile is evaluated at arbitrary angles by
**monotone piecewise-cubic Hermite (PCHIP) interpolation**, which does not overshoot
between samples. Extrapolation outside the tabulated range is refused. An
angle-indexed table is only meaningful for **quasi-static or monotonic** motions. For a
flexion–extension cycle the same angle occurs twice, possibly with different torques.

---

## 4. Spacesuit resistance

$\tau_{suit}$ is defined as the torque the suit applies **against flexion**: positive
values act in the extension direction. Its signed contribution to the knee's flexion
moment balance is therefore $-\tau_{suit}$.

$$
\boxed{\;\tau_{suit}(\varphi, s) = E(\varphi)\,\bigl(1 + h\,s\bigr) + \tau_f\,s,
\qquad E(\varphi) = k_1\varphi + k_2\varphi\lvert\varphi\rvert\;}
$$

| term | meaning | units |
|---|---|---|
| $E(\varphi)$ | elastic, pressure-volume restoring torque toward the neutral angle | N·m |
| $k_1$ | linear stiffness | N·m/rad |
| $k_2$ | quadratic stiffness (the brief's $k$) | N·m/rad² |
| $h$ | hysteresis fraction, $0 \le h < 1$ | – |
| $\tau_f$ | direction-dependent friction / breakaway torque | N·m |
| $s \in [-1, 1]$ | branch state: +1 on the flexion (loading) branch, −1 on the extension branch | – |
| $\varphi_t$ | transition angle for switching branches | rad (configured in degrees) |

Because the radian is dimensionless, $k_1\varphi$ and $k_2\varphi^2$ are both in N·m
when $\varphi$ is in rad. **Degrees are converted to radians before any physics is
evaluated.**

**Branch state with memory (rate-independent).** For each sample increment $\Delta\varphi_i$:

$$
s_{i+1} = s_i + \bigl(\operatorname{sgn}(\Delta\varphi_i) - s_i\bigr)\bigl(1 - e^{-\lvert\Delta\varphi_i\rvert/\varphi_t}\bigr)
$$

This is the exact solution of $ds/d\varphi = (\operatorname{sgn}\dot\varphi - s)/\varphi_t$
over each increment, so it does not depend on the sample spacing. The suit torque
therefore depends on the angle and on the **direction history**, but not on speed.
When motion stops, the suit stays on its current branch, which is the defining
property of hysteresis. The area of a full 0 → φ → 0 loop is the energy the suit
dissipates per cycle, approximately $2\int_0^{\varphi}(hE + \tau_f)\,d\varphi$.

**Modelled:** nonlinear stiffness, direction dependence, hysteresis with memory, and
breakaway friction.
**Omitted:** gas thermodynamics and volume change, pressure dependence beyond what the
coefficients absorb, fabric creep and stress relaxation, rate-dependent viscous effects,
bearing joints, cross-joint coupling between hip and knee in the suit, suit mass and
inertia, and suit–body contact forces and misalignment.

**This is not a validated model of EVA suit mechanics.** Published suit joint-torque
measurements report strongly suit- and joint-specific torque–angle profiles. Matty (2010)
groups them into characteristic profile types. Schmidt (2001) fitted a Preisach hysteresis
model to suited knee-flexion data, among other joints, and attributed most of the stiffness to
gas compression. The Dahl-type branch law used here is a much simpler, single-loop stand-in
for that Preisach model. The default
coefficients here were chosen only so that the curves are of plausible shape and of
order 10–30 N·m. They do not represent any particular suit.

---

## 5. Actuator

* The torque is the decision variable. If the actuator is specified by force, then
  $\tau_a = F_a\,r$ with an **effective perpendicular moment arm** $r$, which is
  configurable and held constant.
* Limitation of a constant $r$: for a linear or cable actuator, $r$ changes with knee
  angle as the line of action moves. The real cuff–leg interface is compliant and
  migrates. Knee joints also roll and slide, so the instantaneous centre of rotation
  moves. A geometry-dependent $r(\varphi)$ is listed as future work.
* Constraints:
  * $\lvert\tau_a\rvert \le \tau_{lim} = \min(\tau_{max},\, F_{max}\,r)$, or $0 \le \tau_a \le \tau_{lim}$ if the actuator is one-directional, as with a cable that can only pull.
  * Optional rate limit $\lvert\dot\tau_a\rvert \le \dot\tau_{max}$, applied as the linear inequalities $\lvert\tau_{a,i+1}-\tau_{a,i}\rvert \le \dot\tau_{max}\Delta t$.
  * Drive efficiency $\eta$ and regeneration fraction $\eta_r$, used only for the electrical-energy estimate.

---

## 6. Torque balance and optimisation

### 6.1 Torque balance (sign convention)

The biological net knee torque that would produce the movement without the suit is
$\tau_{h,b}$. With the suit and exoskeleton, the knee's flexion moment balance requires
$\tau_{h,s} - \tau_{suit} + \tau_a = \tau_{h,b}$, so

$$
\boxed{\;\tau_{h,s} = \tau_{h,b} + \tau_{suit} - \tau_a\;}
\qquad\Rightarrow\qquad \Delta\tau_h \equiv \tau_{h,s} - \tau_{h,b} = \tau_{suit} - \tau_a .
$$

**When this is valid:** the prescribed kinematics are the same with and without the
suit, the suit acts only as a joint torque (no added mass, no change in posture), and
the exoskeleton is massless and aligned with the joint. **A full inverse-dynamics model
is needed** when suit or exoskeleton mass and inertia matter, when the suit changes the
movement strategy (suited subjects move differently), when the hip and knee couple
through the suit, or when the movement involves ground contact. In those cases the
baseline is not a fixed quantity that can be subtracted.

### 6.2 Objective

Every term is normalised so that the weights are dimensionless and $J$ does not depend
on the number of samples $N$:

$$
J(\boldsymbol\tau_a) = \frac{1}{N}\left[
w_\tau\sum_{i=1}^{N}\Bigl(\frac{\tau_{suit,i}-\tau_{a,i}}{\tau_{ref}}\Bigr)^2
+ \lambda\sum_{i=1}^{N}\Bigl(\frac{\tau_{a,i}}{\tau_{ref}}\Bigr)^2
+ \mu\sum_{i=1}^{N-1}\Bigl(\frac{\tau_{a,i+1}-\tau_{a,i}}{\Delta t\,\dot\tau_{ref}}\Bigr)^2
\right]
$$

| term | physical meaning | raw units | normaliser |
|---|---|---|---|
| tracking | extra knee torque the human still supplies because of the suit | (N·m)² | $\tau_{ref} = \max\lvert\tau_{suit}\rvert$, or a configured value |
| effort | actuator torque magnitude, a proxy for motor current, heat, and mass | (N·m)² | $\tau_{ref}$ |
| smoothness | rate of change of actuator torque, a proxy for bandwidth and comfort | (N·m/s)² | $\dot\tau_{ref} = \tau_{ref}/T$ |

* $w_\tau$ sets the importance of transparency, meaning how fully the suit is cancelled.
* $\lambda$ trades suit compensation against actuator effort. Larger $\lambda$ shrinks the assistance.
* $\mu$ penalises rapid torque changes and couples neighbouring samples.
* Only the **ratios** $\lambda/w_\tau$ and $\mu/w_\tau$ change the optimum.

$J$ is a convex quadratic, $J = \tfrac12\tau_a^T H \tau_a + g^T\tau_a + c$, with

$$
H = \frac{2}{N}\left[\frac{w_\tau+\lambda}{\tau_{ref}^2} I + \frac{\mu}{\dot\tau_{ref}^2} D^TD\right],
\quad g = -\frac{2}{N}\frac{w_\tau}{\tau_{ref}^2}\tau_{suit},
\quad c = \frac{1}{N}\frac{w_\tau}{\tau_{ref}^2}\lVert\tau_{suit}\rVert^2,
$$

where $D$ is the $(N-1)\times N$ forward-difference operator divided by $\Delta t$.
$H$ is symmetric positive definite whenever $w_\tau + \lambda > 0$, so the minimiser is unique.

### 6.3 Closed-form solutions (used to validate the numerical solver)

* **Unconstrained, $\mu = 0$:** the problem separates sample by sample, giving
  $\tau_{a,i}^\* = \dfrac{w_\tau}{w_\tau+\lambda}\,\tau_{suit,i}$. Exact compensation
  happens only when $\lambda = 0$.
* **Box bounds only, $\mu = 0$:** each sample is a 1-D convex quadratic, so **clipping**
  the unconstrained optimum to $[\tau_{lb}, \tau_{ub}]$ is exactly optimal.
* **Unconstrained, $\mu > 0$:** solve the linear system $H\tau_a = -g$, which is tridiagonal.
* **Rate limits, or $\mu > 0$ together with bounds:** samples are coupled and inequalities
  are active, so there is **no closed form**. A numerical QP solver is required, and that
  is the role numerical optimisation plays in this project.

### 6.4 Numerical methods

* `scipy.optimize.minimize(method="SLSQP")` with analytic gradient, box bounds, and
  linear inequality constraints for the rate limit. This is the main solver.
* `L-BFGS-B` (box bounds only) and a hand-written **projected gradient descent** with
  step $1/L$, where $L = \lambda_{max}(H)$. These are an educational comparison on the
  box-constrained problem, where the exact answer is known.
* The convergence history ($J$ per iteration), status, iteration count, and maximum
  constraint violation are recorded. The returned profile is checked against every
  actuator limit.

---

## 7. Work and energy

The mechanical work done by the actuator on the leg is

$$
W_a = \int \tau_a\, d\varphi = \int \tau_a\dot\varphi\,dt .
$$

It is discretised as the trapezoidal increments
$\Delta W_i = \tfrac12(\tau_{a,i}+\tau_{a,i+1})(\varphi_{i+1}-\varphi_i)$. This is exact
for constant torque on any trajectory, and $W^+ + W^- = W_{net}$ holds identically.
Positive work means the actuator drives the motion. Negative work means it brakes the
motion and absorbs energy, as it does during the extension phase of a cycle.

**Electrical energy estimate:** $E_{el} = W^+/\eta - \eta_r\lvert W^-\rvert$. This
**ignores** holding losses ($I^2R$ copper loss at zero speed), controller standby power,
and transmission friction. Those can dominate for slow, high-torque motion, so this
estimate is a lower bound on what a real drive would draw.

**None of these quantities is human metabolic energy.** Muscle metabolic cost depends on
co-contraction, muscle fibre mechanics, and isometric force costs that no joint-level
mechanical model captures. The model reports a reduction in **modelled additional knee
torque**, not a reduction in astronaut metabolic expenditure.

---

## 8. Simplified model vs. a full human–suit dynamics model

| aspect | this project | full model |
|---|---|---|
| leg | planar 2-link rigid body | 3-D multibody, joints with 6 degrees of freedom, soft tissue |
| muscles | none (net torque only) | musculoskeletal model (e.g. Hill-type), co-contraction |
| suit | 1-DOF hysteretic torque law | pressurised-fabric FEM or measured multi-joint torque maps, suit mass and inertia |
| interface | perfect torque transfer | cuff compliance, misalignment, pressure points |
| motion | prescribed (inverse dynamics) | predictive or forward simulation; the human adapts to the suit |
| energy | actuator mechanical work, efficiency estimate | metabolic models validated against VO₂ measurements |
