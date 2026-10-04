import numpy as np
from scipy.optimize import root
# ============================================================
# BASIC OPERATIONS
# ============================================================
def hat(x):
    """Skew-symmetric matrix such that hat(x) @ y = x cross y"""
    x1, x2, x3 = x
    return np.array([[0.0, -x3,  x2], [x3,  0.0, -x1], [-x2, x1,  0.0]])

def orthonormalize(R):
    """Project a numerically drifted matrix onto SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    Rproj = U @ Vt

    if np.linalg.det(Rproj) < 0.0: U[:, -1] *= -1.0; Rproj = U @ Vt
    return Rproj


# ============================================================
# 3. COSSERAT/KIRCHHOFF ROD MODELS
# ============================================================
class DynamicCosserat_Kirchhoff:
    def __init__(self, L=0.20, E=1e7, G=5e6, radius=0.005, density=1000.0, u_star=None, v_star=None, crt=False):
        self.L = L
        self.E = E
        self.G = G
        self.radius = radius
        self.density = density
        self.crt = crt

        self.u_star = np.zeros(3) if u_star is None else np.asarray(u_star, dtype=float)
        self.v_star = np.array([0.0, 0.0, 1.0]) if v_star is None else np.asarray(v_star, dtype=float)

        self.A = np.pi * radius**2
        self.Ix = np.pi * radius**4 / 4.0
        self.Iy = self.Ix
        self.Jpolar = self.Ix + self.Iy

        self.mu = density * self.A
        self.J = np.diag([self.Ix, self.Iy, self.Jpolar])
        self.rhoJ = density * self.J

        self.Kse = np.diag([G * self.A, G * self.A, E * self.A])
        self.Kbt = np.diag([E * self.Ix, E * self.Iy, G * self.Jpolar])
        self.Kse_inv = np.diag([1.0 / (G * self.A), 1.0 / (G * self.A), 1.0 / (E * self.A)])
        self.Kbt_inv = np.diag([1.0 / (E * self.Ix), 1.0 / (E * self.Iy), 1.0 / (G * self.Jpolar)])

        self.v_star_s = np.zeros(3)
        self.u_star_s = np.zeros(3)

    @staticmethod
    def pack_state(p, R, n, m, q, omega):
        out = np.empty(24)
        out[0:3] = p
        out[3:12] = R.reshape(9)
        out[12:15] = n
        out[15:18] = m
        out[18:21] = q
        out[21:24] = omega
        return out

    @staticmethod
    def unpack_state(Y):
        return Y[0:3], Y[3:12].reshape(3, 3), Y[12:15], Y[15:18], Y[18:21], Y[21:24]

    def strain_from_wrench(self, R, n, m):
        n_body = R.T @ n
        m_body = R.T @ m
        v = (self.v_star + self.Kse_inv @ n_body) if self.crt else self.v_star
        u = self.u_star + self.Kbt_inv @ m_body
        return v, u, n_body, m_body

# ============================================================
# 4. OPTIMIZED NUMERICAL SOLVER RUCKER PAPER REPLICATION
# ============================================================

class DynamicCosseratSolver:
    def __init__(self, rod, tendons, dt, gravity=None, tip_force=None, tip_moment=None, rtol=1e-6, atol=1e-6, max_step=None):
        self.rod = rod
        self.tendons = tendons
        self.N_tendons = len(tendons)
        self.dt = dt

        self.gravity = np.zeros(3) if gravity is None else np.asarray(gravity, dtype=float)
        self.fe = rod.mu * self.gravity
        self.le = np.zeros(3)

        self.tip_force = (lambda t: np.zeros(3)) if tip_force is None else (tip_force if callable(tip_force) else lambda t: np.asarray(tip_force, dtype=float))
        self.tip_moment = (lambda t: np.zeros(3)) if tip_moment is None else (tip_moment if callable(tip_moment) else lambda t: np.asarray(tip_moment, dtype=float))

        # rtol/atol now actually used: they set the tolerance that the
        # step-doubling error estimate (_estimate_spatial_error /
        # _check_error_tolerance below) is checked against, since the
        # spatial integrator is a fixed-step RK4 with no adaptivity of its
        # own. max_step is accepted for API compatibility but not
        # currently used by the fixed-step integrator (num_points controls
        # step size instead).
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

        self.last_base_guess = np.zeros(6)

    @staticmethod
    def bdf_derivative(current, previous, previous_previous, dt, order):
        if order == 1:
            return (current - previous) / dt
        alpha = -0.25
        c0 = (1.5 + alpha) / (dt * (1.0 + alpha))
        c1 = -2.0 / dt
        c2 = (0.5 + alpha) / (dt * (1.0 + alpha))
        d1 = alpha / (1.0 + alpha)
        dot_previous = (previous-previous_previous)/dt
        #return c0*current + c1*previous + c2*previous_previous + d1*dot_previous
        return (3.0 * current - 4.0 * previous + previous_previous) / (2.0 * dt)

    @staticmethod
    def eval_interp_state(s_grid, Y_grid, s_val):
        """Fast 1D linear evaluation along spatial length s."""
        out = np.empty(Y_grid.shape[1])
        for col in range(Y_grid.shape[1]):
            out[col] = np.interp(s_val, s_grid, Y_grid[:, col])
        return out

    def _estimate_spatial_error(self, ode_fn, Y0, s_full):
        """
        Step-doubling (Richardson) accuracy check for the fixed-step RK4
        spatial integration. RK4 is 4th-order accurate, so integrating the
        SAME initial condition Y0 once on the full node grid (step h) and
        once on every other node (step 2h) gives, for the true solution y:
            y - Y_fine   ~ C h^4
            y - Y_coarse ~ C (2h)^4 = 16 C h^4
        so (Y_fine - Y_coarse) ~ 15 C h^4, and the error remaining in the
        fine solution is approximately:
            err ~ (Y_fine - Y_coarse) / 15
        This only measures the local truncation error of the forward
        integration for a fixed shooting parameter z (i.e. how much finer
        sampling would change the answer) -- not the sensitivity of the
        full boundary-value solution to num_points -- but it's a cheap,
        real accuracy signal where previously there was none (fixed-step
        RK4 has no adaptivity/error control of its own).

        Returns (Y_fine, tip_err) where Y_fine is the full-resolution
        trajectory (same as a plain runge_kutta4 call) and tip_err is the
        estimated absolute error on the tip (final) state.
        """
        Y_fine = self.runge_kutta4(ode_fn, s_full, Y0)

        s_coarse = s_full[::2]
        if s_coarse[-1] != s_full[-1]:
            s_coarse = np.append(s_coarse, s_full[-1])
        Y_coarse = self.runge_kutta4(ode_fn, s_coarse, Y0)

        tip_err = np.abs(Y_fine[-1] - Y_coarse[-1]) / 15.0
        return Y_fine, tip_err

    def _check_error_tolerance(self, tip_err, Y_tip, context=""):
        """
        Compare the step-doubling tip-state error estimate against
        rtol/atol (scipy-style: tol = atol + rtol*|y|) and print a
        WARNING if any state component exceeds it. Returns True if within
        tolerance.
        """
        tol = self.atol + self.rtol * np.abs(Y_tip)
        exceeded = tip_err > tol
        if np.any(exceeded):
            worst_ratio = np.max(tip_err / np.maximum(tol, 1e-300))
            print(f"WARNING: estimated spatial discretization error exceeds "
                  f"tolerance ({context}) -- worst component is "
                  f"{worst_ratio:.2f}x over tol (max abs error "
                  f"{np.max(tip_err):.3e}). Consider increasing num_points.")
            return False
        return True

    def tendon_direct_coefficients(self, s, R, n, m, t):
        f0 = np.zeros(3)
        Fn = np.zeros((3, 3))
        Fm = np.zeros((3, 3))
        l0 = np.zeros(3)
        Ln = np.zeros((3, 3))
        Lm = np.zeros((3, 3))

        v, u, n_body, m_body = self.rod.strain_from_wrench(R, n, m)
        hat_u = hat(u)

        if self.rod.crt:
            v_s_cnst = self.rod.v_star_s - self.rod.Kse_inv @ hat_u @ n_body
            Vn = self.rod.Kse_inv @ R.T
        else:
            v_s_cnst = np.zeros(3)
            Vn = np.zeros((3, 3))

        Um = self.rod.Kbt_inv @ R.T
        u_s_cnst = self.rod.u_star_s - self.rod.Kbt_inv @ hat_u @ m_body

        for tendon in self.tendons:
            T_i = tendon.tension(t)
            rho = tendon.rho(s)
            rho_s = tendon.rho_s(s)
            rho_ss = tendon.rho_ss(s)

            hat_rho = hat(rho)
            g = hat_u @ rho + rho_s + v
            p_i_s = R @ g
            p_i_s_norm = np.linalg.norm(p_i_s)

            if p_i_s_norm < 1e-10:
                raise RuntimeError(f"Tendon tangent singular at s={s:.6e}")

            hat_p = hat(p_i_s)
            D = (hat_p @ hat_p) / (p_i_s_norm**3)

            u_s_const_cross_rho = -hat_rho @ u_s_cnst
            H0 = hat_u @ g + u_s_const_cross_rho + hat_u @ rho_s + rho_ss + v_s_cnst
            Hn = Vn
            Hm = -hat_rho @ Um

            P0 = R @ H0
            Pn = R @ Hn
            Pm = R @ Hm

            f_i0 = -T_i * (D @ P0)
            f_in = -T_i * (D @ Pn)
            f_im = -T_i * (D @ Pm)

            moment_arm = R @ rho
            hat_arm = hat(moment_arm)

            l_i0 = hat_arm @ f_i0
            l_in = hat_arm @ f_in
            l_im = hat_arm @ f_im

            f0 += f_i0
            Fn += f_in
            Fm += f_im
            l0 += l_i0
            Ln += l_in
            Lm += l_im

        return f0, Fn, Fm, l0, Ln, Lm

    def tendon_tip_wrench(self, R, n, m, t):
        F_tip = np.zeros(3)
        M_tip = np.zeros(3)
        v, u, _, _ = self.rod.strain_from_wrench(R, n, m)
        
        for tendon in self.tendons:
            T_i = tendon.tension(t)
            rho = tendon.rho(self.rod.L)
            rho_s = tendon.rho_s(self.rod.L)

            g = hat(u) @ rho + rho_s + v
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-10:
                raise RuntimeError(f"Tendon tangent singular at t={t:.6e}")

            tendon_direction = (R @ g) / g_norm
            moment_arm = R @ rho

            F_i = -T_i * tendon_direction
            M_i = np.cross(moment_arm, F_i)
            F_tip += F_i
            M_tip += M_i

        return F_tip, M_tip

    # ============================================================
    # FAST FIXED-STEP RK4 SPATIAL INTEGRATOR
    # ============================================================
    def runge_kutta4(self, ode_func, s_eval, Y0):
        N = len(s_eval)
        Y_res = np.zeros((N, len(Y0)))
        Y_res[0] = Y0

        for i in range(N - 1):
            s_curr = s_eval[i]
            ds = s_eval[i + 1] - s_curr

            y = Y_res[i]

            k1 = ode_func(s_curr, y)
            k2 = ode_func(s_curr + 0.5*ds, y + 0.5*ds*k1)
            k3 = ode_func(s_curr + 0.5*ds, y + 0.5*ds*k2)
            k4 = ode_func(s_curr + ds, y + ds*k3)

            Y_res[i + 1] = y + (ds/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        return Y_res

    def static_spatial_ode(self, s, Y, t, dist_tendon=False, grav=False):
        p = Y[0:3]
        R = Y[3:12].reshape(3, 3)
        n = Y[12:15]
        m = Y[15:18]

        v, u, _, _ = self.rod.strain_from_wrench(R, n, m)
        p_s = R @ v
        R_s = R @ hat(u)

        rhs_n = np.zeros(3)
        rhs_m = -np.cross(p_s, n)
        K = np.eye(6)

        if dist_tendon:
            f0, Fn, Fm, l0, Ln, Lm = self.tendon_direct_coefficients(s, R, n, m, t)
            K11 = np.eye(3) + Fn
            K12 = Fm
            rhs_n += -f0

            K21 = Ln
            K22 = np.eye(3) + Lm
            rhs_m += -l0
            K = np.block([[K11, K12], [K21, K22]])

        if grav:
            rhs_n += -self.fe
            rhs_m += -self.le

        rhs = np.concatenate([rhs_n, rhs_m])
        wrench_derivative = np.linalg.solve(K, rhs)

        out = np.empty(18)
        out[0:3] = p_s
        out[3:12] = R_s.reshape(9)
        out[12:15] = wrench_derivative[0:3]
        out[15:18] = wrench_derivative[3:6]
        return out

    def dynamic_spatial_ode(self, s, Y, t, s_grid, Y_prev_grid, Y_prev_prev_grid, order, dist_tendon=False, grav=False):
        p, R, n, m, q, omega = self.rod.unpack_state(Y)
        v, u, _, _ = self.rod.strain_from_wrench(R, n, m)

        Y_prev = self.eval_interp_state(s_grid, Y_prev_grid, s)
        _, R_prev, n_prev, m_prev, _, _ = self.rod.unpack_state(Y_prev)
        v_prev, u_prev, _, _ = self.rod.strain_from_wrench(R_prev, n_prev, m_prev)

        if order == 2:
            Y_prev_prev = self.eval_interp_state(s_grid, Y_prev_prev_grid, s)
            _, R_prev_prev, n_prev_prev, m_prev_prev, _, _ = self.rod.unpack_state(Y_prev_prev)
            v_prev_prev, u_prev_prev, _, _ = self.rod.strain_from_wrench(R_prev_prev, n_prev_prev, m_prev_prev)
        else:
            v_prev_prev, u_prev_prev = v_prev, u_prev
            Y_prev_prev = Y_prev

        u_t = self.bdf_derivative(u, u_prev, u_prev_prev, self.dt, order)
        v_t = self.bdf_derivative(v, v_prev, v_prev_prev, self.dt, order) if self.rod.crt else np.zeros(3)
        q_t = self.bdf_derivative(q, Y_prev[18:21], Y_prev_prev[18:21], self.dt, order)
        omega_t = self.bdf_derivative(omega, Y_prev[21:24], Y_prev_prev[21:24], self.dt, order)

        #r_t=Rq
        #R_t=Rw
        p_s = R @ v
        hat_u = hat(u)
        hat_w = hat(omega)
        R_s = R @ hat_u
        p_tt = R @ (q_t + hat_w @ q)

        K_dynamic = np.eye(6)
        rhs_n = self.rod.mu * p_tt
        rhs_m = R@(self.rod.rhoJ @ omega_t + hat_w @ (self.rod.rhoJ @ omega)) - np.cross(p_s, n)

        if dist_tendon:
            f0, Fn, Fm, l0, Ln, Lm = self.tendon_direct_coefficients(s, R, n, m, t)
            K11 = np.eye(3) + Fn
            K12 = Fm
            rhs_n += -f0

            K21 = Ln
            K22 = np.eye(3) + Lm
            rhs_m += -l0
            K_dynamic = np.block([[K11, K12], [K21, K22]])

        if grav:
            rhs_n += -self.fe
            rhs_m += -self.le

        rhs_dynamic = np.concatenate([rhs_n, rhs_m])
        wrench_derivative = np.linalg.solve(K_dynamic, rhs_dynamic)

        omega_s = u_t - hat_u @ omega
        q_s = v_t - hat_u @ q + hat_w @ v

        return self.rod.pack_state(p_s, R_s, wrench_derivative[0:3], wrench_derivative[3:6], q_s, omega_s)

    def solve_static(self, t, dt=False, g=False, pm=False, num_points=30,
        base_guess=None, check_error=True):
        if base_guess is None:
            base_guess = np.zeros(6)

        s_eval = np.linspace(0.0, self.rod.L, num_points)

        def integrate_from_base(z):
            Y0 = np.concatenate([[0, 0, 0], np.eye(3).reshape(9), z])
            ode_fn = lambda s, Y: self.static_spatial_ode(s, Y, t, dist_tendon=dt, grav=g)
            return self.runge_kutta4(ode_fn, s_eval, Y0)

        def residual(z):
            Y_res = integrate_from_base(z)
            Y_tip = Y_res[-1]
            R_tip = Y_tip[3:12].reshape(3, 3)
            n_tip, m_tip = Y_tip[12:15], Y_tip[15:18]

            if pm:
                F_tendon = np.zeros(3)
                _,M_tendon = self.tendon_tip_wrench(R_tip, n_tip, m_tip, t)
            else:F_tendon, M_tendon = self.tendon_tip_wrench(R_tip, n_tip, m_tip, t)
            target_n = F_tendon + self.tip_force(t)
            target_m = M_tendon + self.tip_moment(t)

            return np.concatenate([n_tip - target_n, m_tip - target_m])

        res = root(residual, base_guess, method="hybr", options={"xtol": 1e-6})
        """
        if not res.success:
            print(f"WARNING: Static shooting did not fully converge (t={t:.4f}). "
                  f"Message: {res.message}. Residual norm: {np.linalg.norm(res.fun):.3e}")
        """
        ode_fn = lambda s, Y: self.static_spatial_ode(s, Y, t, dist_tendon=dt, grav=g)
        Y0 = np.concatenate([[0, 0, 0], np.eye(3).reshape(9), res.x])
        if check_error:
            sol_y, tip_err = self._estimate_spatial_error(ode_fn, Y0, s_eval)
            self._check_error_tolerance(tip_err, sol_y[-1],
                context=f"static solve, t={t:.4f}, num_points={num_points}")
        else:
            sol_y = self.runge_kutta4(ode_fn, s_eval, Y0)

        self.last_base_guess = res.x.copy()

        return s_eval, sol_y, res

    def solve_dynamic(self, t, s_grid, Y_prev_grid, Y_prev_prev_grid, order,
        dt=False, g=False, pm=False, num_points=30, base_guess=None, check_error=True):
        if base_guess is None:
            base_guess = self.last_base_guess.copy()

        s_eval = np.linspace(0.0, self.rod.L, num_points)

        def integrate_from_base(z):
            Y0 = np.concatenate([[0, 0, 0], np.eye(3).reshape(9), z, [0, 0, 0], [0, 0, 0]])
            ode_fn = lambda s, Y: self.dynamic_spatial_ode(s, Y, t, s_grid, Y_prev_grid, Y_prev_prev_grid, order, dist_tendon=dt, grav=g)
            return self.runge_kutta4(ode_fn, s_eval, Y0)

        def residual(z):
            Y_res = integrate_from_base(z)
            Y_tip = Y_res[-1]
            R_tip = Y_tip[3:12].reshape(3, 3)
            n_tip, m_tip = Y_tip[12:15], Y_tip[15:18]

            if pm:
                F_tendon = np.zeros(3)
                _,M_tendon = self.tendon_tip_wrench(R_tip, n_tip, m_tip, t)
            else:F_tendon, M_tendon = self.tendon_tip_wrench(R_tip, n_tip, m_tip, t)
            target_n = F_tendon + self.tip_force(t)
            target_m = M_tendon + self.tip_moment(t)

            return np.concatenate([n_tip - target_n, m_tip - target_m])

        res = root(residual, base_guess, method="hybr", options={"xtol": 1e-6})
        """
        if not res.success:
            print(f"WARNING: Dynamic shooting did not fully converge (t={t:.4f}, "
                  f"BDF order={order}). Message: {res.message}. "
                  f"Residual norm: {np.linalg.norm(res.fun):.3e}")
        """
        ode_fn = lambda s, Y: self.dynamic_spatial_ode(s, Y, t, s_grid, Y_prev_grid, Y_prev_prev_grid, order, dist_tendon=dt, grav=g)
        Y0 = np.concatenate([[0, 0, 0], np.eye(3).reshape(9), res.x, [0, 0, 0], [0, 0, 0]])
        if check_error:
            sol_y, tip_err = self._estimate_spatial_error(ode_fn, Y0, s_eval)
            #self._check_error_tolerance(tip_err, sol_y[-1],
            #    context=f"dynamic solve, t={t:.4f}, BDF order={order}, num_points={num_points}")
        else:
            sol_y = self.runge_kutta4(ode_fn, s_eval, Y0)

        self.last_base_guess = res.x.copy()

        return s_eval, sol_y, res

    def project_rotations(self, Y):
        Y_projected = Y.copy()
        for i in range(Y_projected.shape[0]):
            R = Y_projected[i, 3:12].reshape(3, 3)
            Y_projected[i, 3:12] = orthonormalize(R).reshape(9)
        return Y_projected

    def simulate(self, time_array, num_points=30, print_progress=True, dt=False, g=False,
        pm=False, check_error=True):
        time_array = np.asarray(time_array, dtype=float)

        if print_progress:print("t = 0 : STATIC EQUILIBRIUM")

        s_grid, static_y, static_result = self.solve_static(t=0.0, num_points=num_points, dt=dt, g=g,
            pm=pm, check_error=check_error)
        N = len(s_grid)
        Y0 = np.zeros((N, 24))

        for i in range(N):
            Y0[i, 0:18] = static_y[i]
            
        states = [Y0.copy()]
        dynamic_base_guess = self.last_base_guess.copy()

        for k in range(1, len(time_array)):
            t = time_array[k]

            if k == 1:
                order = 1
                Y_prev_prev = Y0
                if print_progress:
                    print(f"Step {k}/{len(time_array)-1} (BDF1), t={t:.4f}")
            else:
                order = 2
                Y_prev_prev = states[-2]
                if print_progress:
                    print(f"Step {k}/{len(time_array)-1} (BDF2), t={t:.4f}")

            _, sol_y, _ = self.solve_dynamic(
                t=t, s_grid=s_grid, Y_prev_grid=states[-1], Y_prev_prev_grid=Y_prev_prev,
                order=order, num_points=num_points, base_guess=dynamic_base_guess, dt=dt, g=g, pm=pm,
                check_error=check_error
            )

            Y_new = self.project_rotations(sol_y)
            states.append(Y_new.copy())
            dynamic_base_guess = self.last_base_guess.copy()

        return {
            "time": time_array,
            "s": s_grid,
            "states": np.asarray(states),
            "static_result": static_result
        }