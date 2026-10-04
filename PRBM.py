import numpy as np
from scipy.integrate import solve_ivp
# ========================================================
# AnanthaSuresh IISc Mechanical Engineering
# PRBM 2R Pseudo-Rigid-Body Model for a TDCR
# ========================================================

class PRBM2R:
    """
    2-DOF Pseudo-Rigid-Body Model for a TDCR.

    Generalized coordinates:
        q = [alpha_x, alpha_y]
        q_dot = [alpha_x_dot, alpha_y_dot]

    No torsional DOF is included:
        alpha_z = 0

    First-stage tendon model:
        tau = sum_i(T_i * r_i)

    where r_i = [r_ix, r_iy] is the fixed bending moment-arm
    vector of tendon i.

    The configuration-dependent tendon Jacobian J_T(q)
    will be added later.
    """

    def __init__(self,L=0.20,r=0.003,m=0.005,E=1e7,
                 I_geom=None,K=None,C=None,g=9.81,l_g=None,
                 moment_arms=None):
        self.L = float(L)
        self.r = float(r)
        self.m = float(m)
        self.E = float(E)
        self.g = float(g)

        # PRBM hinge and rotating-link lengths.
        self.s_h = self.L/6.0*250/225
        self.L_rot = self.L*(-1/6.0*250/225+1)
        self.l_g = self.s_h/2 if l_g is None else float(l_g)

        # Geometric second moment of area.
        self.I_geom = np.pi*self.r**4/4 if I_geom is None else float(I_geom)

        # Rotational inertia of the equivalent rotating link.
        self.Im = self.m*self.L_rot**2/3.0

        # Bending stiffness of each PRBM rotational DOF.
        K0 = 1.25*self.E*self.I_geom/self.L_rot
        if K is None:
            self.Kq = np.diag([K0,K0])
        elif np.isscalar(K):
            self.Kq = np.diag([float(K),float(K)])
        else:
            self.Kq = np.asarray(K,dtype=float).reshape(2,2)

        # Damping matrix.
        if C is None: self.Cq = self.Kq*0.01
        else: self.Cq = float(C)*np.eye(2) 
        

        # 2-DOF rotational inertia matrix.
        self.Mq = np.diag([self.Im,self.Im])

        # Fixed tendon moment-arm vectors.
        # Each row corresponds to one tendon:
        # [r_ix, r_iy] -> bending moment produced by 1 N tension.
        if moment_arms is None:
            self.moment_arms = np.array([
                [0.012,0.0],
                [-0.006,0.0103923],
                [-0.006,-0.0103923]
            ],dtype=float)
        else:
            self.moment_arms = np.asarray(moment_arms,dtype=float)

        if self.moment_arms.ndim != 2 or self.moment_arms.shape[1] != 2:
            raise ValueError("moment_arms must have shape (N_tendons,2).")

        self.n_tendons = self.moment_arms.shape[0]

    # =========================================================
    # PRBM centerline
    # =========================================================

    def shape(self,q,n=100):
        """
        Return 3-D PRBM centerline.

        q = [alpha_x, alpha_y]

        The rotating section is approximated as a straight
        rigid link whose direction is obtained from the two
        bending angles.
        """
        q = np.asarray(q,dtype=float).reshape(2)
        ax,ay = q

        s = np.linspace(0.0,self.L,n)
        r = np.zeros((n,3))

        fixed = s <= self.s_h
        r[fixed,2] = s[fixed]

        sr = s[~fixed]-self.s_h

        # Small-angle-compatible 3-D direction.
        # x = sin(alpha_x), y = sin(alpha_y).
        dx = np.sin(ax)
        dy = np.sin(ay)

        # Normalize so the rotating section remains a rigid link.
        dz = np.sqrt(max(0.0,1.0-dx**2-dy**2))

        r[~fixed,0] = sr*dx
        r[~fixed,1] = sr*dy
        r[~fixed,2] = self.s_h+sr*dz

        return s,r

    # =========================================================
    # Tip kinematics
    # =========================================================

    def tip_pose(self,q):
        """Return 3-D tip position and bending orientation."""
        q = np.asarray(q,dtype=float).reshape(2)
        ax,ay = q

        dx = np.sin(ax)
        dy = np.sin(ay)
        dz = np.sqrt(max(0.0,1.0-dx**2-dy**2))

        p = np.array([
            self.L_rot*dx,
            self.L_rot*dy,
            self.s_h+self.L_rot*dz
        ])

        return p,q.copy()

    # =========================================================
    # Tendon moments
    # =========================================================

    def tendon_moments(self,tensions,t=None):
        """
        Compute total 2-D bending moment from all tendons.

        tau = sum_i(r_i*T_i)

        Output:
            tau = [tau_x,tau_y]
        """
        if callable(tensions):
            if t is None:
                raise ValueError("Time t is required for callable tensions.")
            tensions = tensions(t)

        T = np.asarray(tensions,dtype=float).reshape(-1)

        if T.size != self.n_tendons:
            raise ValueError(
                f"Expected {self.n_tendons} tendon tensions, got {T.size}."
            )

        # Each row:
        # [T_i*r_ix, T_i*r_iy]
        tendon_moments = self.moment_arms*T[:,None]

        # Sum moments from all tendons.
        return np.sum(tendon_moments,axis=0)

    # =========================================================
    # Individual tendon moment history
    # =========================================================

    def individual_tendon_moments(self,tensions,t=None):
        """
        Return individual tendon moment vectors.

        Output shape:
            (N_tendons,2)
        """
        if callable(tensions):
            if t is None:
                raise ValueError("Time t is required for callable tensions.")
            tensions = tensions(t)

        T = np.asarray(tensions,dtype=float).reshape(-1)

        if T.size != self.n_tendons:
            raise ValueError(
                f"Expected {self.n_tendons} tendon tensions, got {T.size}."
            )

        return self.moment_arms*T[:,None]

    # =========================================================
    # Gravity generalized moment
    # =========================================================

    def gravity_moment(self,q):
        """
        Generalized gravity moment.

        This first implementation uses a decoupled approximation:
            tau_gx = m*g*l_g*sin(alpha_x)
            tau_gy = m*g*l_g*sin(alpha_y)

        This will be refined when the full 3-D PRBM geometry
        is introduced.
        """
        q = np.asarray(q,dtype=float).reshape(2)

        return self.m*self.g*self.l_g*np.sin(q)

    # =========================================================
    # Input torque
    # =========================================================

    def input_torque(self,t,q,input_function=None,tensions=None):
        """
        Return total 2-DOF generalized tendon torque.

        input_function, if supplied, must return:
            [tau_x,tau_y]

        Otherwise tensions are converted into moments.
        """
        q = np.asarray(q,dtype=float).reshape(2)

        if input_function is not None:
            tau = np.asarray(input_function(t,q),dtype=float).reshape(2)
            return tau

        if tensions is not None:
            return self.tendon_moments(tensions,t)

        return np.zeros(2)

    # =========================================================
    # Nonlinear state equation
    # =========================================================

    def state_equation(self,t,state,input_function=None,tensions=None):
        """
        2-DOF nonlinear PRBM dynamics:

            M*q_ddot + C*q_dot + K*q - tau_g = tau

        Therefore:

            q_ddot = M^{-1}[
                tau + tau_g - C*q_dot - K*q
            ]
        """
        state = np.asarray(state,dtype=float).reshape(4)

        q = state[:2]
        q_dot = state[2:]

        tau = self.input_torque(
            t,q,
            input_function=input_function,
            tensions=tensions
        )

        tau_g = self.gravity_moment(q)

        # Rearranged equation of motion.
        q_ddot = np.linalg.solve(
            self.Mq,
            tau + tau_g
            - self.Cq@q_dot
            - self.Kq@q
        )

        return np.hstack((q_dot,q_ddot))

    # =========================================================
    # Simulation
    # =========================================================

    def simulate(self,t_span=(0.0,10.0),x0=None,dt=0.002,
                 input_function=None,tensions=None,
                 method="RK45",rtol=1e-8,atol=1e-10):

        """
        Simulate the 2-DOF PRBM.

        State:
            [alpha_x,alpha_y,alpha_x_dot,alpha_y_dot]

        The returned structure is compatible with the
        continuum-model plotting functions.
        """
        if x0 is None:
            x0 = np.zeros(4)

        x0 = np.asarray(x0,dtype=float).reshape(4)

        t_eval = np.arange(
            t_span[0],
            t_span[1]+dt,
            dt
        )
        t_eval = t_eval[t_eval <= t_span[1]+1e-12]

        sol = solve_ivp(
            lambda t,x: self.state_equation(
                t,x,
                input_function=input_function,
                tensions=tensions
            ),
            t_span,
            x0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        time = sol.t
        q = sol.y[:2].T
        q_dot = sol.y[2:].T
        nt = len(time)

        # -----------------------------------------------------
        # Spatial grid.
        # -----------------------------------------------------
        s = np.linspace(0.0,self.L,100)
        ns = len(s)

        # Common state:
        # [p(3),R(9),n(3),m(3),q(3)] = 21 states
        states = np.zeros((nt,ns,21))

        # Histories.
        q_ddot = np.zeros((nt,2))
        tau = np.zeros((nt,2))
        tendon_moments = np.zeros((nt,self.n_tendons,2))

        for k,t in enumerate(time):
            qk = q[k]
            qdk = q_dot[k]

            # PRBM acceleration.
            xk = np.hstack((qk,qdk))
            dxk = self.state_equation(
                t,xk,
                input_function=input_function,
                tensions=tensions
            )
            q_ddot[k] = dxk[2:4]

            # Total tendon moment.
            tau[k] = self.input_torque(
                t,qk,
                input_function=input_function,
                tensions=tensions
            )

            # Individual tendon contributions.
            if tensions is not None:
                tendon_moments[k] = self.individual_tendon_moments(
                    tensions,t
                )

            # -------------------------------------------------
            # PRBM backbone geometry.
            # -------------------------------------------------
            _,centerline = self.shape(qk,n=ns)

            states[k,:,0:3] = centerline

            # -------------------------------------------------
            # Approximate 3-D orientation.
            # -------------------------------------------------
            ax,ay = qk

            Rx = np.array([
                [1.0,0.0,0.0],
                [0.0,np.cos(ax),-np.sin(ax)],
                [0.0,np.sin(ax),np.cos(ax)]
            ])

            Ry = np.array([
                [np.cos(ay),0.0,np.sin(ay)],
                [0.0,1.0,0.0],
                [-np.sin(ay),0.0,np.cos(ay)]
            ])

            R = Ry@Rx

            fixed = s <= self.s_h

            states[k,fixed,3:12] = np.eye(3).reshape(1,9)
            states[k,~fixed,3:12] = R.reshape(1,9)

            # PRBM does not explicitly solve Cosserat n,m.
            states[k,:,12:15] = 0.0
            states[k,:,15]=tau[k,1]
            states[k,:,16]=tau[k,0]
            states[k,:,17]=0.0
            # Common q slot stores:
            # [alpha_x,alpha_y,alpha_z=0]
            states[k,:,18] = ax
            states[k,:,19] = ay
            states[k,:,20] = 0.0

        # -----------------------------------------------------
        # Tip position.
        # -----------------------------------------------------
        tip = np.zeros((nt,3))

        for k in range(nt):
            tip[k] = self.tip_pose(q[k])[0]

        # -----------------------------------------------------
        # Energies.
        # -----------------------------------------------------
        T_energy = 0.5*np.sum(
            q_dot*(q_dot@self.Mq),
            axis=1
        )

        V_spring = np.array([
            0.5*q[k]@self.Kq@q[k]
            for k in range(nt)
        ])

        # Gravity potential corresponding to the
        # decoupled approximation.
        V_gravity = self.m*self.g*self.l_g*(
            np.cos(q[:,0])-1.0
            +np.cos(q[:,1])-1.0
        )

        V = V_spring+V_gravity
        E_total = T_energy+V

        Q_spring = -(q@self.Kq.T)
        Q_damping = -(q_dot@self.Cq.T)
        Q_gravity = np.array([
            self.gravity_moment(q[k])
            for k in range(nt)
        ])

        return {
            "time":time,
            "t":time,
            "s":s,
            "states":states,
            "q":q,
            "q_dot":q_dot,
            "q_ddot":q_ddot,
            "alpha_x":q[:,0],
            "alpha_y":q[:,1],
            "alpha_x_dot":q_dot[:,0],
            "alpha_y_dot":q_dot[:,1],
            "alpha_x_ddot":q_ddot[:,0],
            "alpha_y_ddot":q_ddot[:,1],
            "u":tau,
            "tau":tau,
            "tendon_moments":tendon_moments,
            "tip":tip,
            "tip_position":tip,
            "x_tip":tip[:,0],
            "y_tip":tip[:,1],
            "z_tip":tip[:,2],
            "theta_tip":q,
            "T":T_energy,
            "V":V,
            "V_spring":V_spring,
            "V_gravity":V_gravity,
            "E_total":E_total,
            "Q_spring":Q_spring,
            "Q_damping":Q_damping,
            "Q_gravity":Q_gravity,
            "M":self.Mq,
            "C":self.Cq,
            "D":self.Cq,
            "K":self.Kq,
            "moment_arms":self.moment_arms,
            "success":sol.success,
            "message":sol.message
        }

    # =========================================================
    # Model parameters
    # =========================================================

    def parameters(self):
        return {
            "L":self.L,
            "r":self.r,
            "m":self.m,
            "E":self.E,
            "s_h":self.s_h,
            "L_rot":self.L_rot,
            "l_g":self.l_g,
            "I_geom":self.I_geom,
            "I_dyn":self.Im,
            "M":self.Mq,
            "C":self.Cq,
            "K":self.Kq,
            "g":self.g,
            "n_tendons":self.n_tendons,
            "moment_arms":self.moment_arms
        }