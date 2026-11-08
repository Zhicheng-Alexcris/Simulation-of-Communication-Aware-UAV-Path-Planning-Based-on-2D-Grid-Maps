# environment.py
import math

PIPE_RADIUS = 3.0
VERTIPORT_RADIUS = PIPE_RADIUS
LANE_GAP = 4.0
MIN_HEIGHT = 5.0


class Building:
    def __init__(self, vertices):
        self.vertices = vertices
        self.x_bounds = (min(v[0] for v in vertices), max(v[0] for v in vertices))
        self.y_bounds = (min(v[1] for v in vertices), max(v[1] for v in vertices))
        self.z_bounds = (min(v[2] for v in vertices), max(v[2] for v in vertices))

    def contains_with_margin(self, point, safety_margin=2.0):
        x, y, z = point
        return (self.x_bounds[0] - safety_margin <= x <= self.x_bounds[1] + safety_margin and
                self.y_bounds[0] - safety_margin <= y <= self.y_bounds[1] + safety_margin and
                self.z_bounds[0] - safety_margin <= z <= self.z_bounds[1] + safety_margin)


class SafeLayer:
    def __init__(self, z_min: float, z_max: float, name: str = "safe layer"):
        assert z_max > z_min
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.name = name

    @property
    def mid(self) -> float:
        return 0.5 * (self.z_min + self.z_max)


class Vertiport:
    """Vertiport cylinder."""
    def __init__(self, x: float, y: float, radius: float,
                 z_min: float = 0.0, z_max: float = 40.0,
                 name: str = "Vertiport", drone_id=None):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.name = name
        self.drone_id = drone_id

    def contains_xy(self, x: float, y: float, margin: float = 0.0) -> bool:
        return math.hypot(x - self.x, y - self.y) <= max(0.0, self.radius - margin)

    def contains(self, p, margin: float = 0.0) -> bool:
        x, y, z = p
        return self.contains_xy(x, y, margin) and (self.z_min <= z <= self.z_max)


class UAV:
    def __init__(self, x, y, z, min_height=MIN_HEIGHT):
        self.x = float(x)
        self.y = float(y)
        self.z = max(float(z), min_height)

    @property
    def pos(self):
        return (self.x, self.y, self.z)

    def move_to(self, new_pos, min_height=MIN_HEIGHT):
        self.x, self.y, self.z = new_pos
        self.z = max(self.z, min_height)


class Environment:
    def __init__(self):
        # Buildings (unchanged — keep your latest list)
        self.buildings = [
            Building([[-99.39755, 66.38202, 0.0], [-69.397552, 66.38202, 0.0],
                      [-99.39755, 101.38202, 0.0], [-69.397552, 101.38202, 0.0],
                      [-99.39755, 66.38202, 28.742558], [-69.397552, 66.38202, 28.742558],
                      [-99.39755, 101.38202, 28.742558], [-69.397552, 101.38202, 28.742558]]),

            Building([[-99.36653, -84.30113, 0.0], [-69.36653, -84.30113, 0.0],
                      [-99.36653, -24.301128, 0.0], [-69.36653, -24.301128, 0.0],
                      [-99.36653, -84.30113, 27.326977], [-69.36653, -84.30113, 27.326977],
                      [-99.36653, -24.301128, 27.326977], [-69.36653, -24.301128, 27.326977]]),

            Building([[-95.73451, 18.077122, 0.0], [-70.734512, 18.077122, 0.0],
                      [-95.73451, 48.07712, 0.0], [-70.734512, 48.07712, 0.0],
                      [-95.73451, 18.077122, 33.946167], [-70.734512, 18.077122, 33.946167],
                      [-95.73451, 48.07712, 33.946167], [-70.734512, 48.07712, 33.946167]]),

            Building([[-54.065247, 18.395721, 0.0], [-29.065247, 18.395721, 0.0],
                      [-54.065247, 43.39572, 0.0], [-29.065247, 43.39572, 0.0],
                      [-54.065247, 18.395721, 28.0035], [-29.065247, 18.395721, 28.0035],
                      [-54.065247, 43.39572, 28.0035], [-29.065247, 43.39572, 28.0035]]),

            Building([[-52.93194, -54.667038, 0.0], [-27.931938, -54.667038, 0.0],
                      [-52.93194, -24.667038, 0.0], [-27.931938, -24.667038, 0.0],
                      [-52.93194, -54.667038, 39.60136], [-27.931938, -54.667038, 39.60136],
                      [-52.93194, -24.667038, 39.60136], [-27.931938, -24.667038, 39.60136]]),

            Building([[-50.595566, 61.352776, 0.0], [-25.595566, 61.352776, 0.0],
                      [-50.595566, 91.352776, 0.0], [-25.595566, 91.352776, 0.0],
                      [-50.595566, 61.352776, 34.47477], [-25.595566, 61.352776, 34.47477],
                      [-50.595566, 91.352776, 34.47477], [-25.595566, 91.352776, 34.47477]]),

            Building([[-48.065582, -93.114586, 0.0], [-28.065582, -93.114586, 0.0],
                      [-48.065582, -73.114586, 0.0], [-28.065582, -73.114586, 0.0],
                      [-48.065582, -93.114586, 29.087406], [-28.065582, -93.114586, 29.087406],
                      [-48.065582, -73.114586, 29.087406], [-28.065582, -73.114586, 29.087406]]),

            Building([[23.50177, 63.122696, 0.0], [93.50177, 63.122696, 0.0],
                      [23.50177, 93.122696, 0.0], [93.50177, 93.122696, 0.0],
                      [23.50177, 63.122696, 28.742558], [93.50177, 63.122696, 28.742558],
                      [23.50177, 93.122696, 28.742558], [93.50177, 93.122696, 28.742558]]),

            Building([[19.871971, -93.073074, 0.0], [44.87197, -93.073074, 0.0],
                      [19.871971, -68.073074, 0.0], [44.87197, -68.073074, 0.0],
                      [19.871971, -93.073074, 36.211174], [44.87197, -93.073074, 36.211174],
                      [19.871971, -68.073074, 36.211174], [44.87197, -68.073074, 36.211174]]),

            Building([[22.53323, 18.391376, 0.0], [47.53323, 18.391376, 0.0],
                      [22.53323, 43.391376, 0.0], [47.53323, 43.391376, 0.0],
                      [22.53323, 18.391376, 37.34879], [47.53323, 18.391376, 37.34879],
                      [22.53323, 43.391376, 37.34879], [47.53323, 43.391376, 37.34879]]),

            Building([[23.766113, -50.405323, 0.0], [48.766113, -50.405323, 0.0],
                      [23.766113, -20.405323, 0.0], [48.766113, -20.405323, 0.0],
                      [23.766113, -50.405323, 37.34879], [48.766113, -50.405323, 37.34879],
                      [23.766113, -20.405323, 37.34879], [48.766113, -20.405323, 37.34879]]),

            Building([[66.16034, -47.813103, 0.0], [96.16034, -47.813103, 0.0],
                      [66.16034, -27.813103, 0.0], [96.16034, -27.813103, 0.0],
                      [66.16034, -47.813103, 21.38055], [96.16034, -47.813103, 21.38055],
                      [66.16034, -27.813103, 21.38055], [96.16034, -27.813103, 21.38055]]),

            Building([[67.83895, 19.057272, 0.0], [97.83895, 19.057272, 0.0],
                      [67.83895, 39.057274, 0.0], [97.83895, 39.057274, 0.0],
                      [67.83895, 19.057272, 37.11637], [97.83895, 19.057272, 37.11637],
                      [67.83895, 39.057274, 37.11637], [97.83895, 39.057274, 37.11637]]),

            Building([[66.27215, -92.871155, 0.0], [91.27215, -92.871155, 0.0],
                      [66.27215, -67.871155, 0.0], [91.27215, -67.871155, 0.0],
                      [66.27215, -92.871155, 28.0035], [91.27215, -92.871155, 28.0035],
                      [66.27215, -67.871155, 28.0035], [91.27215, -67.871155, 28.0035]]),
        ]

        # ✅ 7 drones now (新增无人机6/7)
        self.starts = [
            (-110, -110, 12),  # drone0
            (-110, 110, 18),  # drone1
            (0, -110, 24),  # drone2
            (0, -110, 30),  # drone3
            (0, -110, 36),  # drone4
            (-110, 55, 35),  # drone5
            (-110, -15, 27),  # drone6
            (55, -60, 19),  # drone7 (新增)
            (-58, -60, 11),  # drone8 (新增)
        ]

        self.ends = [
            (110, 110, 12),  # drone0
            (110, -110, 18),  # drone1
            (-110, 0, 24),  # drone2
            (110, 0, 30),  # drone3
            (0, 110, 36),  # drone4
            (110, -55, 35),  # drone5
            (110, 55, 27),  # drone6
            (-60, 55, 19),  # drone7
            (55, 50, 11),  # drone8
        ]

        assert len(self.starts) == len(self.ends), (
            f"starts/ends length mismatch: {len(self.starts)} vs {len(self.ends)}"
        )

        self.min_height = MIN_HEIGHT

        self.safe_layers = [
            SafeLayer(10, 12, "L1"),  # mid 11
            SafeLayer(18, 20, "L2"),  # mid 19
            SafeLayer(26, 28, "L3"),  # mid 27
            SafeLayer(34, 36, "L4"),  # mid 35
            SafeLayer(38, 40, "L5"),  # mid 39 (顶层稍挤)
        ]
        self.vertiports = self._build_vertiports_from_starts_ends(
            z_min=0.0, z_max=40.0, nd=2
        )

        # UE layout (cars treated as users)
        self.ues = [
            (-55.908882, 8.473935, 0.0),
            (-42.47412, -4.922293, 0.0),
            (-30.15337, 7.871437, 0.0),
            (-14.238827, -4.922293, 0.0),
            (-6.27928, -69.91425, 0.0),
            (-6.27928, -25.971306, 0.0),
            (-6.27928, 31.71241, 0.0),
            (-6.27928, 55.117256, 0.0),
            (-6.27928, 73.46995, 0.0),
            (6.86512, -64.550971, 0.0),
            (6.86512, -35.88601, 0.0),
            (6.86512, 45.202553, 0.0),
            (6.86512, 63.555244, 0.0),
            (20.563005, 6.448998, 0.0),
            (33.997769, -6.94723, 0.0),
            (46.31852, 5.8465, 0.0),
            (62.233061, -6.94723, 0.0),
            (76.6634, 5.8465, 0.0),
        ]

        # planner uses this for sampling etc.
        self.step_size = 10

        self.xlim = (-110.0, 110.0)
        self.ylim = (-110.0, 110.0)
        self.zlim = (0.0, 40.0)

    def get_start_end(self, drone_id):
        assert 0 <= drone_id < len(self.starts), f"Drone ID must be 0..{len(self.starts) - 1}"
        return self.starts[drone_id], self.ends[drone_id]

    def clamp_to_layer(self, z: float, layer_idx: int) -> float:
        L = self.safe_layers[layer_idx]
        return min(max(z, L.z_min), L.z_max)


    def layer_mid_z(self, layer_idx: int) -> float:
        return self.safe_layers[layer_idx].mid

    @staticmethod
    def calculate_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


#新增无人机
    @staticmethod
    def _xy_key(p, nd=2):
        return (round(float(p[0]), nd), round(float(p[1]), nd))

    def _build_vertiports_from_starts_ends(self, z_min=0.0, z_max=40.0, nd=2):
        start_map = {}
        end_map = {}

        for i, s in enumerate(self.starts):
            k = self._xy_key(s, nd)
            start_map.setdefault(k, {"xy": (float(s[0]), float(s[1])), "ids": []})
            start_map[k]["ids"].append(i)

        for i, e in enumerate(self.ends):
            k = self._xy_key(e, nd)
            end_map.setdefault(k, {"xy": (float(e[0]), float(e[1])), "ids": []})
            end_map[k]["ids"].append(i)

        vps = []
        seen = set()

        for k, info in start_map.items():
            x, y = info["xy"]
            ids_start = info["ids"]
            ids_end = end_map.get(k, {}).get("ids", [])

            if len(ids_start) >= 2:
                name = "UAV Takeoff Station"
                drone_id = None
            elif ids_end:
                name = "Shared Vertiport"
                drone_id = None
            else:
                idx = ids_start[0]
                name = f"Start {idx + 1}"
                drone_id = idx

            vps.append(
                Vertiport(x, y, radius=VERTIPORT_RADIUS,
                          z_min=float(z_min), z_max=float(z_max),
                          name=name, drone_id=drone_id)
            )
            seen.add(k)

        # remaining ends
        for k, info in end_map.items():
            if k in seen:
                continue

            x, y = info["xy"]
            ids_end = info["ids"]

            if len(ids_end) >= 2:
                name = "Shared Landing Zone"
                drone_id = None
            else:
                idx = ids_end[0]
                name = f"End {idx + 1}"
                drone_id = idx

            vps.append(
                Vertiport(x, y, radius=VERTIPORT_RADIUS,
                          z_min=float(z_min), z_max=float(z_max),
                          name=name, drone_id=drone_id)
            )
            seen.add(k)

        return vps
