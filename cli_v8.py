"""CLI entry point for fixed-budget CA-Diffusion-ES v8 validation."""

import csv

from . import cli as _cli
from .integration_v8 import plan_drone_ca_diffusion_es


_HISTORY_FIELDS = [
    "topology_conditioned_count",
    "topology_conditioned_feasible_ratio",
    "topology_reference_count",
    "topology_radius_scale_min",
    "topology_radius_scale_max",
    "online_skeleton_parent_count",
    "online_skeleton_projected_count",
    "replay_parent_count",
]


def main() -> None:
    original_writer = csv.DictWriter

    def v8_writer(handle, fieldnames, *args, **kwargs):
        fields = list(fieldnames)
        if "generation" in fields:
            fields.extend(field for field in _HISTORY_FIELDS if field not in fields)
        return original_writer(handle, fields, *args, **kwargs)

    _cli.plan_drone_ca_diffusion_es = plan_drone_ca_diffusion_es
    csv.DictWriter = v8_writer
    try:
        _cli.main()
    finally:
        csv.DictWriter = original_writer


if __name__ == "__main__":
    main()
