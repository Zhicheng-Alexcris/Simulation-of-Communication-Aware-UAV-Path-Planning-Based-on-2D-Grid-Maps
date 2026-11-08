# plot.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D


class PathVisualizer:
    def __init__(self, environment, pipe_radius=2.0):
        self.env = environment
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.pipe_radius = float(pipe_radius)

        # Tableau 10 (10 distinct colors)
        self.color_cycle = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        self.collision_color = '#d62728'
        self.building_color = 'gray'
        self.building_alpha = 0.4
        self.pipe_alpha = 0.7

        self._path_count = 0
        self._start_end_drawn = False

        self.start_zone_labels = {
            0: "Takeoff Zone 1",
            1: "Takeoff Zone 2",
            2: "UAV Takeoff Station",
            3: "UAV Takeoff Station",
            4: "UAV Takeoff Station",
        }
        self.end_zone_labels = {
            0: "Landing Zone 1",
            1: "Landing Zone 2",
            2: "Landing Zone 3",
            3: "Landing Zone 4",
            4: "Landing Zone 5",
        }

    # -----------------------------
    # 3D components
    # -----------------------------
    def plot_buildings(self):
        for building in getattr(self.env, "buildings", []):
            verts = building.vertices
            x = [v[0] for v in verts]
            y = [v[1] for v in verts]
            z = [v[2] for v in verts]
            self.ax.bar3d(min(x), min(y), min(z),
                          max(x) - min(x), max(y) - min(y), max(z) - min(z),
                          color=self.building_color,
                          alpha=self.building_alpha,
                          edgecolor='k')

    def plot_ues_3d(self, z=0.8, color="tab:green", marker="^", size=55):
        """Plot UEs in 3D at a fixed height z (slightly above ground)."""
        if not getattr(self.env, "ues", None):
            return
        U = np.asarray(self.env.ues, dtype=float)
        if U.ndim != 2 or U.shape[1] < 2:
            return

        Z = np.full((U.shape[0],), float(z))
        self.ax.scatter(
            U[:, 0], U[:, 1], Z,
            marker=marker, s=size,
            color=color, edgecolors="k", linewidths=0.4,
            alpha=0.95, label="_nolegend_"
        )

    def create_cylinder(self, center, direction, radius, height):
        d = np.array(direction, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            d = np.array([0.0, 0.0, 1.0])
        else:
            d = d / n

        if abs(d[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])

        v1 = np.cross(d, a)
        nv1 = np.linalg.norm(v1)
        if nv1 < 1e-9:
            a = np.array([0.0, 0.0, 1.0])
            v1 = np.cross(d, a)
            nv1 = np.linalg.norm(v1)

        v1 = v1 / (nv1 + 1e-12)
        v2 = np.cross(d, v1)
        v2 = v2 / (np.linalg.norm(v2) + 1e-12)

        zz = np.linspace(0, height, 10)
        theta = np.linspace(0, 2 * np.pi, 20)
        th, zg = np.meshgrid(theta, zz)
        xg = radius * np.cos(th)
        yg = radius * np.sin(th)

        points = np.stack([xg, yg, zg])
        R = np.column_stack([v1, v2, d])
        rot = np.tensordot(R, points, axes=1)
        trans = rot + np.array(center).reshape(3, 1, 1)
        return trans[0], trans[1], trans[2]

    def create_pipe_segment(self, start, end, radius):
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        if length <= 1e-12:
            return None, None, None
        direction = direction / length
        return self.create_cylinder(start, direction, radius, length)

    def plot_trajectory_as_pipe(self, trajectory, color=None, label=None, alpha=None):
        """Draw a path as a thick pipe (3D surfaces)."""
        if trajectory is None or len(trajectory) < 2:
            return

        if color is None:
            color = self.color_cycle[self._path_count % len(self.color_cycle)]
        if alpha is None:
            alpha = self.pipe_alpha

        traj = [np.asarray(p, dtype=float) for p in trajectory]

        # start cylinder
        start_dir = traj[1] - traj[0]
        x, y, z = self.create_cylinder(traj[0], start_dir, self.pipe_radius, self.pipe_radius * 2)
        self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

        # end cylinder
        end_dir = traj[-1] - traj[-2]
        x, y, z = self.create_cylinder(traj[-1], end_dir, self.pipe_radius, self.pipe_radius * 2)
        self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

        # pipe segments
        for i in range(len(traj) - 1):
            x, y, z = self.create_pipe_segment(traj[i], traj[i + 1], self.pipe_radius)
            if x is not None:
                self.ax.plot_surface(x, y, z, color=color, alpha=alpha)

        self._path_count += 1

    def plot_takeoff_landing_labels(self, colors, uav_station_color="#111111",
                                    top_margin=1.0, xy_offset=(2.0, 2.0)):
        if getattr(self, "_start_end_drawn", False):
            return
        if not (hasattr(self.env, "starts") and hasattr(self.env, "ends")):
            return

        def bbox():
            return dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)

        station_labeled = False
        station_ids = {2, 3, 4}  # Drone3/4/5 (0-based)

        z_top = float(getattr(self.env, "zlim", (0.0, 40.0))[1])
        z_text = z_top - float(top_margin)

        dx, dy = float(xy_offset[0]), float(xy_offset[1])

        for i, (start, end) in enumerate(zip(self.env.starts, self.env.ends)):
            c = colors[i % len(colors)]
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)

            s_label = str(self.start_zone_labels.get(i, f"Takeoff Zone {i + 1}")).strip()
            e_label = str(self.end_zone_labels.get(i, f"Landing Zone {i + 1}")).strip()

            # markers (no legend)
            self.ax.scatter(*start, color='g', s=150, marker='^', label="_nolegend_")
            self.ax.scatter(*end, color='r', s=150, marker='v', label="_nolegend_")

            sx, sy, sz = start[0] + dx, start[1] + dy, z_text
            ex, ey, ez = end[0] + dx, end[1] + dy, z_text

            # Takeoff label
            if i in station_ids:
                if not station_labeled:
                    self.ax.text(
                        sx, sy, sz, "UAV Takeoff Station",
                        fontsize=10, color=uav_station_color,
                        ha='left', va='bottom', bbox=bbox()
                    )
                    station_labeled = True
            else:
                self.ax.text(
                    sx, sy, sz, s_label,
                    fontsize=10, color=c,
                    ha='left', va='bottom', bbox=bbox()
                )

            # Landing label
            self.ax.text(
                ex, ey, ez, e_label,
                fontsize=10, color=c,
                ha='left', va='bottom', bbox=bbox()
            )

        self._start_end_drawn = True

    def plot_vertiports(self, alpha=1.0, colors=None, show_labels=False):
        if colors is None:
            colors = self.color_cycle

        r_align = float(self.pipe_radius)
        z0, z1 = (0.0, 40.0)

        for vp in getattr(self.env, "vertiports", []):
            c = colors[int(vp.drone_id) % len(colors)] if vp.drone_id is not None else '#7e57c2'

            zmin = float(getattr(vp, "z_min", 0.0))
            zmax = float(getattr(vp, "z_max", 10.0))
            zmin = max(z0, zmin)
            zmax = min(z1, zmax)
            h = max(0.1, zmax - zmin)

            xg, yg, zg = self.create_cylinder(
                center=(vp.x, vp.y, zmin),
                direction=(0, 0, 1),
                radius=r_align,
                height=h
            )
            self.ax.plot_surface(xg, yg, zg, color=c, alpha=alpha, edgecolor='none')

            if show_labels:
                z_text = min(zmax + 1.0, z1 - 0.5)
                self.ax.text(vp.x, vp.y, z_text, getattr(vp, "name", "Vertiport"),
                             fontsize=9, color=c, ha='center')

    def set_legend_like_topdown(self, n_drones, colors, ue_color="tab:green"):
        handles = []
        for i in range(n_drones):
            c = colors[i % len(colors)]
            handles.append(Line2D([0], [0], color=c, lw=2.6, label=f"Drone {i+1}"))

        handles.append(Line2D(
            [0], [0], linestyle="None",
            marker="^", markersize=8,
            markerfacecolor=ue_color, markeredgecolor="k",
            label="UEs"
        ))

        self.ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.15, 1),
                       fontsize=10, frameon=True)

    def _decorate_plot(self):
        x0, x1 = -120.0, 120.0
        y0, y1 = -120.0, 120.0
        z0, z1 = 0.0, 40.0

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_zlim(z0, z1)
        self.ax.set_zticks([0, 20, 40])


        try:
            self.ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
        except Exception:
            pass

        self.ax.set_xlabel('X (m)', fontsize=12, labelpad=15)
        self.ax.set_ylabel('Y (m)', fontsize=12, labelpad=15)
        self.ax.set_zlabel('Height (m)', fontsize=12, labelpad=15)
        self.ax.set_title('Multi-UAV Pipe Path Planning (3D View)', fontsize=14, pad=20)

        try:
            self.ax.zaxis.set_label_position("left")
        except Exception:
            pass
        try:
            # (x,y) 是 axes 坐标系，不是数据坐标；x<0 就会往左挪
            self.ax.zaxis.set_label_coords(-0.08, 0.5)
        except Exception:
            pass

        plt.tight_layout()

    def show(self):
        plt.show()

    def save(self, filename, dpi=300, bbox_inches='tight'):
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)

    # ---------------------------------------------------------
    # Top-down (XY)
    # ---------------------------------------------------------
    def plot_topdown(self,
                     paths,
                     title="Top-down projection (XY)",
                     colors=None,
                     labels=None,
                     show_buildings=True,
                     show_vertiports=True,
                     show_ues=True,
                     ue_color="tab:green",
                     ue_marker="^",
                     ue_size=55,
                     save_path=None,
                     dpi=300):

        if colors is None:
            colors = self.color_cycle

        if labels is None:
            labels = [f"Drone {i + 1}" for i in range(len(paths))]

        fig, ax = plt.subplots(figsize=(7.6, 7.1))

        # 1) buildings
        if show_buildings:
            for b in getattr(self.env, "buildings", []):
                ax.add_patch(
                    Rectangle(
                        (b.x_bounds[0], b.y_bounds[0]),
                        b.x_bounds[1] - b.x_bounds[0],
                        b.y_bounds[1] - b.y_bounds[0],
                        facecolor="gray",
                        alpha=0.25,
                        edgecolor="k",
                        linewidth=0.8,
                        zorder=1,
                    )
                )

        # 2) paths
        for i, path in enumerate(paths):
            P = np.asarray(path, dtype=float)
            if P.size == 0:
                continue
            c = colors[i % len(colors)]
            lab = labels[i] if i < len(labels) else f"Drone {i + 1}"
            ax.plot(P[:, 0], P[:, 1], lw=2.4, color=c, label=lab, zorder=3)

        # 3) vertiports: KEEP POINTS, remove old text
        if show_vertiports and hasattr(self.env, "vertiports"):
            r_align = float(self.pipe_radius)
            for vp in self.env.vertiports:
                if getattr(vp, "drone_id", None) is not None:
                    c = colors[int(vp.drone_id) % len(colors)]
                else:
                    c = "#7e57c2"

                ax.add_patch(
                    Circle(
                        (vp.x, vp.y),
                        radius=r_align,
                        facecolor=c,
                        alpha=0.95,
                        edgecolor=c,
                        linewidth=2.0,
                    )
                )


        def bbox():
            return dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.2)

        station_ids = {2, 3, 4}
        station_labeled = False
        uav_station_color = "#111111"
        off = 2.0

        for i, path in enumerate(paths):
            P = np.asarray(path, dtype=float)
            if P.size == 0 or P.shape[0] < 2:
                continue

            c = colors[i % len(colors)]

            s_label = str(self.start_zone_labels.get(i, f"Takeoff Zone {i + 1}")).strip()
            e_label = str(self.end_zone_labels.get(i, f"Landing Zone {i + 1}")).strip()

            sx, sy = float(P[0, 0]), float(P[0, 1])
            ex, ey = float(P[-1, 0]), float(P[-1, 1])

            if i in station_ids:
                if not station_labeled:
                    ax.text(sx + off, sy + off, "UAV Takeoff Station",
                            fontsize=10, color=uav_station_color, fontweight="bold",
                            ha="left", va="bottom", bbox=bbox(), zorder=7)
                    station_labeled = True
            else:
                ax.text(sx + off, sy + off, s_label,
                        fontsize=10, color=c, fontweight="bold",
                        ha="left", va="bottom", bbox=bbox(), zorder=7)

            ax.text(ex + off, ey + off, e_label,
                    fontsize=10, color=c, fontweight="bold",
                    ha="left", va="bottom", bbox=bbox(), zorder=7)

        # 5) UEs
        if show_ues and getattr(self.env, "ues", None):
            U = np.asarray(self.env.ues, dtype=float)
            if U.ndim == 2 and U.shape[1] >= 2:
                ax.scatter(U[:, 0], U[:, 1],
                           marker=ue_marker,
                           s=ue_size,
                           color=ue_color,
                           edgecolors="k",
                           linewidths=0.4,
                           label="UEs",
                           zorder=6)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-120.0, 120.0)
        ax.set_ylim(-120.0, 120.0)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            print(f"Saved: {save_path}")
        else:
            plt.show()


