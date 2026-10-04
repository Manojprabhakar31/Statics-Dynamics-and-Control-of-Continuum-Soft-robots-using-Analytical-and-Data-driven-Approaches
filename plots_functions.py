import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MARKERS = ["o", "s", "^", "D", "v", "p"]
LINE_STYLES = ["-", "--", "-.", ":"]


class Plots:
    """Utility class for plotting 3D spatial curves, tip trajectories,

    2D projections, phase portraits, internal mechanics, and tendon tensions

    for Cosserat rod simulations.

    Usage:

        from plotter import Plotter

        Plots.plot_backbone_snapshots(results)

    """

    @staticmethod
    def _prep_results_and_labels(results, labels):
        """Standardizes input results and labels into list formats."""
        if not isinstance(results, (list, tuple)):
            results = [results]
        if labels is None:
            labels = [f"Case {i+1}" for i in range(len(results))]
        elif len(labels) != len(results):
            raise ValueError("Number of labels must match number of results.")
        return results, labels

    @staticmethod
    def _apply_ax_style(ax, title="", xlabel="", ylabel="", zlabel=None, aspect_equal=False):
        """Applies common formatting and grid/legend styles to Matplotlib axes."""
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)

        if zlabel and hasattr(ax, "set_zlabel"):
            ax.set_zlabel(zlabel, labelpad=8)
            ax.set_box_aspect([1, 1, 1])
        elif aspect_equal:
            ax.set_aspect("equal", adjustable="datalim")

        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

    # -----------------------------------------------------------------------------
    # 3D Spatial & Trajectory Plots
    # -----------------------------------------------------------------------------

    @classmethod
    def plot_backbone_snapshots(cls, results, labels=None, snapshot="final"):
        """Plots 3D spatial curve(s) of the rod backbone at a specified snapshot."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for idx, (res, label) in enumerate(zip(results, labels)):
            states = res["states"]
            p = (states[-1] if snapshot == "final" else states[snapshot])[:, :3]
            ax.plot(
                p[:, 0], p[:, 1], p[:, 2],
                linewidth=2.5,
                label=label,
                marker=MARKERS[idx % len(MARKERS)],
                markevery=max(1, len(p) // 10),
                markersize=5,
                alpha=0.85
            )

        title_str = snapshot.capitalize() if isinstance(snapshot, str) else f"Step {snapshot}"
        cls._apply_ax_style(ax, f"Dynamic Cosserat Rod Shape ({title_str} State)", "X [m]", "Y [m]", "Z [m]")
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_tip_trajectory(cls, results, labels=None):
        """Plots 3D spatial trajectory of the rod's end-effector (tip) over time."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for idx, (res, label) in enumerate(zip(results, labels)):
            tip = res["states"][:, -1, :3]
            m_style = MARKERS[idx % len(MARKERS)]

            ax.plot(
                tip[:, 0], tip[:, 1], tip[:, 2],
                linewidth=2,
                label=label,
                marker=m_style,
                markevery=max(1, len(tip) // 15),
                markersize=4,
                alpha=0.85
            )
            ax.scatter(*tip[0], color="green", s=50, edgecolors="k", zorder=5, label="Start" if idx == 0 else None)
            ax.scatter(*tip[-1], color="red", s=50, marker="X", edgecolors="k", zorder=5, label="End" if idx == 0 else None)

        cls._apply_ax_style(ax, "Dynamic Cosserat Tip 3D Trajectory", "X [m]", "Y [m]", "Z [m]")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------------
    # 2D Projections & Analysis
    # -----------------------------------------------------------------------------

    @classmethod
    def plot_tip_trajectory2(cls, results, labels=None, plane="XZ"):
        """Plots 2D projection of tip trajectory."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fig, ax = plt.subplots(figsize=(8, 5))

        plane_map = {"XY": (0, 1, "X [m]", "Y [m]"), "XZ": (0, 2, "X [m]", "Z [m]"), "YZ": (1, 2, "Y [m]", "Z [m]")}
        i1, i2, xlabel, ylabel = plane_map.get(plane.upper(), plane_map["XZ"])

        for idx, (res, label) in enumerate(zip(results, labels)):
            tip = res["states"][:, -1, :3]
            ax.plot(
                tip[:, i1], tip[:, i2],
                linewidth=2,
                linestyle=LINE_STYLES[idx % 4],
                marker=MARKERS[idx % len(MARKERS)],
                markevery=max(1, len(tip) // 12),
                markersize=5,
                label=label,
                alpha=0.85
            )
            ax.scatter(tip[0, i1], tip[0, i2], color="green", s=50, edgecolors="k", zorder=5)
            ax.scatter(tip[-1, i1], tip[-1, i2], color="red", marker="X", s=60, edgecolors="k", zorder=5)

        cls._apply_ax_style(ax, f"Tip Trajectory ({plane.upper()} Projection)", xlabel, ylabel, aspect_equal=True)
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_tip_trajectory3(cls, results, labels=None, plane="XZ", snapshot_indices=None, show_backbone=True, draw_arc_approx=True):
        """Plots 2D projection of tip trajectory with backbone snapshots and circular arc approximation."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fig, ax = plt.subplots(figsize=(8, 5.5))

        plane_map = {"XY": (0, 1, "X [m]", "Y [m]"), "XZ": (0, 2, "X [m]", "Z [m]"), "YZ": (1, 2, "Y [m]", "Z [m]")}
        i1, i2, xlabel, ylabel = plane_map.get(plane.upper(), plane_map["XZ"])

        for idx, (res, label) in enumerate(zip(results, labels)):
            time, states = res["time"], res["states"]
            tip = states[:, -1, :3]
            snaps = np.linspace(0, len(time) - 1, 4, dtype=int) if snapshot_indices is None else snapshot_indices

            # Tip path & points
            ax.plot(tip[:, i1], tip[:, i2], lw=2, ls=LINE_STYLES[idx % 4], label=f"{label} Tip Path", alpha=0.85)
            ax.scatter(tip[0, i1], tip[0, i2], color="green", s=50, edgecolors="k", zorder=5, label="Start" if idx == 0 else None)
            ax.scatter(tip[-1, i1], tip[-1, i2], color="red", marker="X", s=60, edgecolors="k", zorder=5, label="End" if idx == 0 else None)

            # Backbone snapshots
            if show_backbone:
                alphas = np.linspace(0.4, 0.9, len(snaps))
                for k_idx, k in enumerate(snaps):
                    p = states[k, :, :3]
                    ax.plot(p[:, i1], p[:, i2], lw=1.5, alpha=alphas[k_idx], label=f"{label} (t={time[k]:.2f}s)")
                    ax.scatter(p[0, i1], p[0, i2], color="k", s=15, zorder=4)

            # Circular arc approximation
            if draw_arc_approx:
                cx, cy, r = 0.0, 0.4 / 6, 0.4 * 5 / 6
                theta = np.linspace(np.radians(90), np.radians(30), 100)
                ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), "--", color="magenta", lw=1.8, label="Arc Approx", zorder=3)
                ax.scatter(cx, cy, color="magenta", marker="+", s=80, lw=2, zorder=6, label="Arc Center")

                for k in snaps:
                    ax.plot([cx, states[k, -1, i1]], [cy, states[k, -1, i2]], ":", color="gray", lw=1.2, alpha=0.7)

        cls._apply_ax_style(ax, f"Tip Trajectory & Backbone ({plane.upper()})", xlabel, ylabel, aspect_equal=True)
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------------
    # Time Series & Phase Portraits
    # -----------------------------------------------------------------------------

    @classmethod
    def plot_tip_coordinates(cls, results, labels=None):
        """Plots X, Y, Z coordinates vs. time in 3 subplots."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        for idx, (res, label) in enumerate(zip(results, labels)):
            time, tip = res["time"], res["states"][:, -1, :3]
            for dim, comp in enumerate(["X", "Y", "Z"]):
                axs[dim].plot(time, tip[:, dim], lw=1.8, ls=LINE_STYLES[idx % 4], label=label)
                axs[dim].set_ylabel(f"{comp} [m]")
                axs[dim].grid(True, linestyle=":", alpha=0.6)

        axs[0].set_title("Tip Position Coordinates vs. Time", fontweight="bold")
        axs[2].set_xlabel("Time [s]")
        axs[0].legend(loc="best")
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_tip_phase_portraits(cls, results, labels=None):
        """Plots X, Y, Z tip phase portraits (Position vs Velocity)."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        for idx, (res, label) in enumerate(zip(results, labels)):
            time = np.asarray(res["time"])
            if "states" in res:
                states = np.asarray(res["states"])
                p_tip = states[:, -1, :3]
                q_tip, R_tip = states[:, -1, 18:21], states[:, -1, 3:12].reshape(-1, 3, 3)
                v_tip = np.einsum("nij,nj->ni", R_tip, q_tip)
            elif "tip" in res:
                p_tip = np.asarray(res["tip"])
                v_tip = np.gradient(p_tip, time, axis=0)

            for dim in range(3):
                ax = axs[dim]
                ax.plot(p_tip[:, dim], v_tip[:, dim], ls=LINE_STYLES[idx % 4], lw=1.8, label=label)
                ax.plot(p_tip[0, dim], v_tip[0, dim], marker=MARKERS[idx % len(MARKERS)], mfc="none", ms=6)
                ax.plot(p_tip[-1, dim], v_tip[-1, dim], marker="*", ms=8)

        for ax, comp in zip(axs, ["X", "Y", "Z"]):
            ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
            cls._apply_ax_style(ax, f"{comp} Phase Portrait", f"{comp} pos [m]", f"{comp} vel [m/s]")

        fig.suptitle("Tip Phase Portraits", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------------
    # Internal Mechanics (Wrench & Tension)
    # -----------------------------------------------------------------------------

    @classmethod
    def plot_internal_wrench_snapshots(cls, results, labels=None, snapshot_indices=None):
        """Plots internal force n(s) and moment m(s) along the rod at selected times."""
        results, labels = cls._prep_results_and_labels(results, labels)

        for res, label in zip(results, labels):
            time, s, states = res["time"], res["s"], res["states"]
            snaps = np.linspace(0, len(time) - 1, 4, dtype=int) if snapshot_indices is None else snapshot_indices

            for title_type, unit, slice_idx in [("Force", "N", 12), ("Moment", "N m", 15)]:
                fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
                for j, comp in enumerate(["X", "Y", "Z"]):
                    for k in snaps:
                        val = states[k, :, slice_idx + j]
                        axs[j].plot(s, val, lw=1.8, label=f"t = {time[k]:.3f}s")
                    axs[j].set_ylabel(f"\({title_type[0].lower()}_{comp}\) [{unit}]")
                    axs[j].grid(True, linestyle=":", alpha=0.6)

                axs[0].set_title(f"{label}: Internal {title_type} Along Rod", fontweight="bold")
                axs[-1].set_xlabel("Arc length s [m]")
                axs[0].legend(loc="best", fontsize=8)
                plt.tight_layout()
                plt.show()

    @classmethod
    def plot_internal_wrench_time_history(cls, results, labels=None, backbone_fractions=None):
        """Plots n(s_i,t) and m(s_i,t) at fixed locations along the backbone vs time."""
        results, labels = cls._prep_results_and_labels(results, labels)
        fractions = [0.0, 0.25, 0.50, 0.75, 1.0] if backbone_fractions is None else backbone_fractions

        for res, label in zip(results, labels):
            time, s, states = res["time"], res["s"], res["states"]
            s_indices = [np.argmin(np.abs(s - f * s[-1])) for f in fractions]

            for title_type, unit, slice_idx in [("Force", "N", 12), ("Moment", "N m", 15)]:
                fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
                for j, comp in enumerate(["X", "Y", "Z"]):
                    for idx_s, frac in zip(s_indices, fractions):
                        axs[j].plot(time, states[:, idx_s, slice_idx + j], lw=1.8, label=f"s/L = {frac:.2f}")
                    axs[j].set_ylabel(f"\({title_type[0].lower()}_{comp}\) [{unit}]")
                    axs[j].grid(True, linestyle=":", alpha=0.6)
                    axs[j].legend(loc="best", fontsize=8)

                axs[0].set_title(f"{label}: Internal {title_type} vs Time", fontweight="bold")
                axs[-1].set_xlabel("Time [s]")
                plt.tight_layout()
                plt.show()

    @classmethod
    def plot_tensions(cls, time, tendons):
        """Plots time-varying tension profile for each tendon."""
        fig, ax = plt.subplots(figsize=(8, 4))
        for idx, tendon in enumerate(tendons):
            T = [tendon.tension(t) for t in time]
            ax.plot(time, T, lw=2, ls=LINE_STYLES[idx % 4], label=getattr(tendon, "name", f"Tendon {idx+1}"))

        cls._apply_ax_style(ax, "Time-Varying Tendon Tension", "Time [s]", "Tension [N]")
        plt.tight_layout()
        plt.show()