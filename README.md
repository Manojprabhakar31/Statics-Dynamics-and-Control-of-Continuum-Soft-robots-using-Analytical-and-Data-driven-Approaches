# Continuum Robot Dynamics and MuJoCo Simulation

This repository contains the computational implementation developed for modeling and simulating a **tendon-driven continuum robot (TDCR)** using Cosserat/Kirchhoff rod mechanics and a continuum-mechanics-informed MuJoCo discretization.

The current implementation focuses on:

* Dynamic Cosserat rod modeling
* Kirchhoff rod specialization
* Static equilibrium through shooting and spatial integration
* Dynamic simulation using BDF1/BDF2 time discretization
* Point-moment and distributed tendon loading
* Straight and general tendon routing
* Gravity and external tip loading
* Spatial RK4 integration
* MuJoCo rigid-link/elastic-joint discretization of the continuum robot
* Tendon actuation and interactive forward dynamics in MuJoCo
* Comparison of continuum-rod and MuJoCo responses

---

## 1. Continuum Robot Model
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/e4693096-c26f-4033-ab73-56a2a5be1bd5" />

<img width="756" height="757" alt="image" src="https://github.com/user-attachments/assets/ebcb9349-e634-439f-b47b-64a4daad551b" />

The backbone is represented using a Cosserat-rod formulation.

The spatial state used for the static model is

$$
Y_s =
\begin{bmatrix}
p\\
R\\
n\\
m
\end{bmatrix}
$$

where

* \(p(s)\in\mathbb{R}^3\) — backbone position
* \(R(s)\in SO(3)\) — backbone orientation
* \(n(s)\in\mathbb{R}^3\) — internal force
* \(m(s)\in\mathbb{R}^3\) — internal moment

The dynamic state extends this to

$$
Y =
\begin{bmatrix}
p & R & n & m & q & \omega
\end{bmatrix}
$$

where

* \(q\) — body-frame translational velocity
* \(\omega\) — body-frame angular velocity

The strain variables are recovered from the internal wrench:

$$
v=v^*+K_{se}^{-1}R^Tn
$$

$$
u=u^*+K_{bt}^{-1}R^Tm
$$

For the Kirchhoff specialization, shear and extension are suppressed:

$$
v=v^*=
\begin{bmatrix}
0&0&1
\end{bmatrix}^T
$$

while bending and torsion remain represented through \(u\).

---

## 2. Kirchhoff and Cosserat Models

The notebook implements both configurations using the same computational framework.

### Kirchhoff configuration

```python
k_rod = DynamicCosserat_Kirchhoff(
    L=L,
    E=E,
    G=G,
    radius=radius,
    density=density,
    crt=False
)
```

For this case:

$$
v=v^*
$$

and the deformation is described primarily through bending and torsion.

### Cosserat configuration

```python
c_rod = DynamicCosserat_Kirchhoff(
    L=L,
    E=E,
    G=G,
    radius=radius,
    density=density,
    crt=True
)
```

Here both shear/extension and bending/torsion strains are recovered from the internal force and moment.

Therefore, the notebook can directly compare:

$$
\boxed{\text{Kirchhoff rod}}
$$

against

$$
\boxed{\text{Full Cosserat rod}}
$$

using the same numerical solver.

---

## 3. Static Equilibrium

The static problem is solved before starting the dynamic simulation.

The spatial equations are

$$
p_s=Rv
$$

$$
R_s=R\hat{u}
$$

and the equilibrium equations are solved in terms of the internal wrench derivatives.

For the basic case,

$$
n_s=0
$$

$$
m_s=-p_s\times n
$$

External gravity and tendon loading can be added to these equations.

### Shooting method

The unknown base wrench

$$
z=
\begin{bmatrix}
n(0)\\
m(0)
\end{bmatrix}
$$

is determined using a nonlinear root solver.

The spatial equations are integrated from the base and the tip residual is constructed from the tendon/tip loading conditions.

The notebook uses:

```python
res = root(
    residual,
    base_guess,
    method="hybr",
    options={"xtol": 1e-6}
)
```

Thus the static computation follows:

$$
\boxed{
\text{Base wrench guess}
\rightarrow
\text{spatial integration}
\rightarrow
\text{tip residual}
\rightarrow
\text{root solve}
}
$$

<img width="578" height="672" alt="image" src="https://github.com/user-attachments/assets/43343fff-b33c-4c02-8b48-74dd32389f43" />
<img width="557" height="671" alt="image" src="https://github.com/user-attachments/assets/8c51c3e6-346a-4eaa-89ed-3cbfa6a4781b" />
<img width="557" height="677" alt="image" src="https://github.com/user-attachments/assets/0eb427b3-8a08-4989-99b5-d48c9ba918f6" />

---

## 4. Spatial Numerical Integration

The notebook uses a fixed-step **fourth-order Runge-Kutta method (RK4)** for spatial integration.

For each spatial step,

$$
Y_{i+1}
=
Y_i+
\frac{\Delta s}{6}
(k_1+2k_2+2k_3+k_4)
$$

The implementation is contained in:

```python
runge_kutta4(...)
```

A step-doubling estimate is also implemented to estimate the spatial integration error.

The fine and coarse spatial solutions are compared using the fourth-order Richardson estimate:

$$
e_s\approx
\frac{Y_{\text{fine}}-Y_{\text{coarse}}}{15}
$$

The notebook also checks this estimate against the specified absolute and relative tolerances.

---

## 5. Tendon Modeling

The notebook supports a general tendon representation:

$$
\rho(s)
$$

with

$$
\rho_s(s),\qquad
\rho_{ss}(s)
$$

and a time-varying tension

$$
T(t)
$$

The tendon object is defined using:

```python
@dataclass
class Tendon:
    rho: callable
    rho_s: callable
    rho_ss: callable
    tension: callable
```

Two routing types are explicitly implemented.

### Straight tendon

A constant radial offset is used:

$$
\rho(s)=
\begin{bmatrix}
r\cos\theta\\
r\sin\theta\\
0
\end{bmatrix}
$$

with

$$
\rho_s=0,\qquad \rho_{ss}=0
$$

### Helical/general tendon

The notebook also contains a helical routing representation:

$$
\rho(s)=
\begin{bmatrix}
r\cos\phi(s)\\
r\sin\phi(s)\\
0
\end{bmatrix}
$$

where

$$
\phi(s)=\frac{2\pi N_t}{L}s+\phi_0
$$

The associated first and second spatial derivatives are implemented explicitly.

---

## 6. Point-Moment and Distributed Tendon Loading

The notebook supports two tendon-loading formulations.

### Point-moment approximation

The tendon effect is applied through the tendon wrench at the tip.

The tendon force and moment are calculated from the local tendon direction and moment arm.

### Distributed tendon loading

For the distributed formulation, the tendon force and moment are incorporated directly into the spatial equations.

The tendon contribution is written in the form

$$
f=f_0+F_n n_s+F_m m_s
$$

$$
l=l_0+L_n n_s+L_m m_s
$$

which produces a coupled wrench-derivative system:

$$
\begin{bmatrix}
I+F_n & F_m\\
L_n & I+L_m
\end{bmatrix}
\begin{bmatrix}
n_s\\
m_s
\end{bmatrix}
=
\begin{bmatrix}
\text{force RHS}\\
\text{moment RHS}
\end{bmatrix}
$$

The notebook solves this system directly using:

```python
wrench_derivative = np.linalg.solve(K, rhs)
```

This is the main implementation used to avoid separately integrating tendon-strain derivative equations.

---

## 7. Gravity and External Loading

Gravity is included through the distributed mass force

$$
f_e=\rho A g
$$

The solver supports:

* no gravity
* gravity
* tip force
* tip moment

These can be combined with either point-moment or distributed tendon loading.

The notebook therefore evaluates four main loading configurations for each rod model:

| Case | Tendon model | Gravity |
| ---- | ------------ | ------- |
| 1    | Point moment | No      |
| 2    | Distributed  | No      |
| 3    | Point moment | Yes     |
| 4    | Distributed  | Yes     |

---

## 8. Dynamic Cosserat Formulation

The dynamic state contains

$$
Y=
[p,R,n,m,q,\omega]
$$

The dynamic spatial equations use the inertial terms associated with translational and rotational acceleration.

The translational acceleration is obtained from

$$
p_{tt}
=
R(\dot q+\hat{\omega}q)
$$

and the rotational dynamics contain

$$
\rho J\dot{\omega}
+
\omega\times(\rho J\omega)
$$

The dynamic wrench derivative is therefore solved simultaneously with the spatial kinematic equations.

---

## 9. BDF Time Discretization

The notebook uses a two-stage backward differentiation scheme.

### First time step — BDF1

At

$$
t=\Delta t
$$

the first-order approximation is

$$
\dot{x}_k
\approx
\frac{x_k-x_{k-1}}{\Delta t}
$$

### Subsequent time steps — BDF2

For

$$
t\ge2\Delta t
$$

the implemented second-order approximation is

$$
\dot{x}_k
\approx
\frac{3x_k-4x_{k-1}+x_{k-2}}
{2\Delta t}
$$

Therefore the complete time integration is

$$
\boxed{
\text{Static equilibrium}
\rightarrow
\text{BDF1}
\rightarrow
\text{BDF2}
\rightarrow
\text{BDF2}
\rightarrow\cdots
}
$$

The BDF derivatives are applied to quantities including

$$
u,\quad v,\quad q,\quad\omega
$$

as required by the selected Cosserat/Kirchhoff formulation.

---

## 10. Dynamic Shooting at Every Time Step

The dynamic problem remains a spatial boundary-value problem at every time step.

At each \(t_k\):

1. Previous spatial solution is interpolated.
2. BDF derivatives are evaluated.
3. A base wrench is estimated.
4. The spatial dynamic equations are integrated with RK4.
5. The tip boundary residual is calculated.
6. `scipy.optimize.root` updates the base wrench.
7. The process repeats until the shooting solution is obtained.

Therefore:

$$
(q_k,\omega_k,Y_{k-1},Y_{k-2})
\rightarrow
\text{BDF derivatives}
\rightarrow
\text{spatial BVP}
\rightarrow
\text{shooting solve}
\rightarrow
Y_k
$$

This is the main dynamic computational loop implemented in the notebook.

---

## 11. Implemented Dynamic Cases

The notebook explicitly runs both Kirchhoff and Cosserat models.

### Kirchhoff

```python
result1 = k_solver.simulate(..., dt=False, g=False, pm=True)
result2 = k_solver.simulate(..., dt=False, g=True,  pm=True)
result3 = k_solver.simulate(..., dt=True,  g=False, pm=False)
result4 = k_solver.simulate(..., dt=True,  g=True,  pm=False)
```

### Cosserat

```python
result5 = c_solver.simulate(..., dt=False, g=False, pm=True)
result6 = c_solver.simulate(..., dt=False, g=True,  pm=True)
result7 = c_solver.simulate(..., dt=True,  g=False, pm=False)
result8 = c_solver.simulate(..., dt=True, g=True,  pm=False)
```

Hence the notebook contains the four requested combinations for both rod theories:

$$
\boxed{
\begin{array}{c}
\text{Kirchhoff + point moment}\\
\text{Kirchhoff + distributed tendon}\\
\text{Cosserat + point moment}\\
\text{Cosserat + distributed tendon}
\end{array}}
$$

with and without gravity.

---

## 12. Time-Varying Tendon Actuation

The notebook implements several tendon tension profiles, including:

* step loading
* linear ramp
* parabolic ramp
* sinusoidal loading
* cosine loading
* multi-tendon loading
* sequential tendon activation
* combined sinusoidal/cosine actuation

For example:

$$
T_1(t)=10\quad\text{after a specified activation time}
$$

or

$$
T(t)=10|\sin(\pi t)|
$$

These profiles are directly passed to the tendon model and evaluated during the dynamic simulation.

---

## 13. MuJoCo TDCR Model

The notebook also contains a MuJoCo implementation based on a **continuum-mechanics-informed rigid-link discretization**.

The continuum robot is represented as a chain of rigid cylindrical links connected by elastic rotational joints.

The model parameters are derived from:

* Young's modulus \(E\)
* Poisson ratio \(\nu\)
* density
* backbone radius
* segment length
* number of links
* tendon radius

The shear modulus is calculated as

$$
G=\frac{E}{2(1+\nu)}
$$

and the bending/torsional stiffness parameters are calculated from the section properties.

For each discretized segment:

$$
k_x=\frac{EI}{l_i},
\qquad
k_y=\frac{EI}{l_i},
\qquad
k_z=\frac{GJ}{l_i}
$$

These stiffnesses are directly written into the MuJoCo XML.

---

## 14. MuJoCo XML Generator

The notebook contains a configurable XML generator:

```python
generate_tdcr_xml_from_config(...)
```

The configuration specifies:

* number of segments
* links per segment
* segment lengths
* material properties
* tendon count
* tendon radius
* tendon angular offsets
* joint configuration
* joint damping
* actuator limits
* MuJoCo settings

The generator automatically creates:

* rigid links
* elastic joints
* tendon sites
* spatial tendons
* tendon motors
* end-effector site
* collision exclusions
* keyframe/pretension configuration

This allows the continuum robot parameters to be converted into an MJCF model automatically.

---

## 15. MuJoCo Tendon Actuation

The generated model uses MuJoCo spatial tendons and tendon motors.

The notebook maps the commanded tension to the actuator control:

```python
data.ctrl[act_id] = -controller.tension[idx]
```

The interactive controller supports individual tendon tension adjustment.

For the first three tendons:

```text
1 / q → Tendon 1
2 / w → Tendon 2
3 / e → Tendon 3
```

Additional tendon controls are included for multi-segment models.

Other controls include:

```text
r     → reset tensions
SPACE → pause/resume
ESC   → exit
```

---

## 16. MuJoCo Forward Dynamics

The notebook runs MuJoCo forward dynamics using:

```python
mujoco.mj_step(model, data)
```

The simulation records:

* time
* tip position
* tendon tension
* tendon length
* joint positions
* joint velocities
* joint accelerations
* actuator forces/torques
* backbone link positions

The backbone configuration is reconstructed from the positions of the discretized links.

The output therefore provides a direct time history of the simulated continuum robot.

---

## 17. Continuum Model vs MuJoCo

The notebook collects results from:

```text
Kirchhoff + point moment
Kirchhoff + distributed tendon
Cosserat + point moment
Cosserat + distributed tendon
MuJoCo
```

These can be plotted together to inspect the dynamic behavior and tip trajectory.

The current implementation therefore provides a common computational comparison between:

$$
\boxed{\text{continuum mechanics}}
$$

and

$$
\boxed{\text{rigid-link MuJoCo approximation}}
$$

---

## 18. Visualization Implemented

The notebook contains visualization functions for:

### Backbone shape

3D backbone configuration at a selected time.

### Tip trajectory

3D trajectory of the end effector.

### 2D tip trajectory

Projection onto:

* XY
* XZ
* YZ

planes.

### Tip coordinates

Time histories of

$$
x(t),\quad y(t),\quad z(t)
$$

### Tendon tensions

Time histories of each tendon tension.

These plots allow the continuum models and MuJoCo model to be compared using the same simulation outputs.

---

## 19. Error and Comparison Utilities

The notebook contains functions for calculating:

### Tip position error

$$
e_p=
100\frac{\|p_{\text{model}}-p_{\text{ref}}\|}{L}
$$

### Centerline RMS error

$$
e_{\text{RMS}}
=
100\frac{
\sqrt{\operatorname{mean}\left(
\|p_{\text{model}}(s)-p_{\text{ref}}(s)\|^2
\right)}
}{L}
$$

### Tip orientation error

The relative rotation is calculated as

$$
R_e=R_{\text{ref}}^TR_{\text{model}}
$$

and the angular error is obtained from

$$
\theta=
\cos^{-1}
\left(
\frac{\operatorname{tr}(R_e)-1}{2}
\right)
$$

These utilities provide the basis for quantitative comparison between models.

---

## 20. Current Computational Pipeline

The implemented notebook can be summarized as:

```text
Robot/material parameters
        ↓
Tendon routing + tension profiles
        ↓
Cosserat / Kirchhoff rod definition
        ↓
Static shooting solution
        ↓
Spatial RK4 integration
        ↓
BDF1/BDF2 time discretization
        ↓
Dynamic shooting solution
        ↓
Backbone / tip trajectory
        ↓
Comparison and visualization
```

In parallel, the MuJoCo path is:

```text
Robot/material parameters
        ↓
TDCR configuration
        ↓
MJCF XML generation
        ↓
Rigid-link + elastic-joint TDCR
        ↓
MuJoCo spatial tendons
        ↓
Tendon tension input
        ↓
MuJoCo forward dynamics
        ↓
Tip / joint / backbone trajectories
```

---

## 21. What Is Currently Implemented

The uploaded notebook currently implements:

* [x] Cosserat rod formulation
* [x] Kirchhoff rod specialization
* [x] Static equilibrium
* [x] Dynamic rod simulation
* [x] Shooting-based boundary solution
* [x] Fixed-step RK4 spatial integration
* [x] BDF1 time discretization
* [x] BDF2 time discretization
* [x] Straight tendon routing
* [x] Helical/general tendon routing functions
* [x] Time-varying tendon tension
* [x] Point-moment tendon loading
* [x] Distributed tendon loading
* [x] Gravity loading
* [x] Tip force and tip moment interfaces
* [x] Spatial integration error estimation
* [x] Kirchhoff/Cosserat comparison
* [x] MuJoCo TDCR XML generation
* [x] PCS-style rigid-link discretization used for the MuJoCo model
* [x] Elastic joint stiffness derived from material properties
* [x] MuJoCo spatial tendons
* [x] Interactive tendon tension control
* [x] MuJoCo forward dynamics
* [x] Backbone and tip trajectory extraction
* [x] Model comparison and visualization

---

## 22. Not Yet Implemented in This Notebook

The following topics appear in the presentation/research roadmap but should **not** be claimed as implemented in the current notebook:

* GVS reduced-order dynamics
* Legendre reduced-order dynamics
* Fourier reduced-order dynamics
* FEM-like reduced-order dynamics
* Gaussian/dependent strain bases
* Explicit SoRoSim reduced-order dynamics implementation
* Standalone PCC forward-kinematics implementation
* Standalone PCS forward-kinematics solver
* Reduced-order \(M,C,K,D,B\) matrix assembly
* Jacobian-based task-space inverse kinematics
* Jacobian-based inverse dynamics controller
* Closed-loop tip trajectory tracking controller
* Automatic tendon-tension optimization for a desired tip trajectory
* Experimental hardware validation
* System identification of \(E\) or tendon parameters

These can remain as **future development directions**, but they should be separated from the current implementation in the repository description.

---

## 23. Repository Development Direction

The current notebook establishes the mechanics and simulation foundation:

$$
\boxed{
\text{Cosserat/Kirchhoff}
\rightarrow
\text{BDF dynamics}
\rightarrow
\text{MuJoCo discretization}
}
$$

The next development stages can build on this foundation toward reduced-order modeling and closed-loop control, but those components are not included in the current implementation described above.

---

## References

### Continuum Robot Mechanics

D. C. Rucker and R. J. Webster III,
**“Statics and Dynamics of Continuum Robots With General Tendon Routing and External Loading,”**
IEEE Transactions on Robotics, 2011.

### MuJoCo Continuum Robot Discretization

C. Shentu, N. Baldassini, T. Zheng, P. Rao, and J. Burgner-Kahrs,
**“Do Rigid-Body Simulators Dream of Soft Robots? Learning Contact-Rich Manipulation for Tendon-Driven Continuum Robots,”**
arXiv:2606.22397, 2026.

### Reduced-Order Modeling

Mathew et al.,
**“Reduced Order Modeling of Hybrid Soft-Rigid Robots Using Global, Local, and State-Dependent Strain Parameterization.”**

---

## Project Status

**Current focus:** continuum mechanics implementation and simulation validation.

```text
✓ Static Cosserat/Kirchhoff
✓ Dynamic Cosserat/Kirchhoff
✓ BDF1/BDF2 time discretization
✓ Point/distributed tendon loading
✓ Gravity loading
✓ General tendon representation
✓ MuJoCo TDCR generation
✓ MuJoCo tendon actuation
✓ MuJoCo forward dynamics
✓ Model visualization/comparison

→ Reduced-order modeling
→ Task-space inverse modeling
→ Trajectory control
→ Experimental validation
```
