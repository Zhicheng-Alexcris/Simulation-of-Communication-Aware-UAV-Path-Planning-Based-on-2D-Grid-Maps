"""CLI entry point for CA-Diffusion-ES v7 validation runs."""

import csv

from . import cli as _cli
from .integration_v7 import plan_drone_ca_diffusion_es


_HISTORY_FIELDS = [
    "topology_conditioned_count",
    "topology_conditioned_feasible_ratio",
    "topology_residual_max_radius_m",
]


def main() -> None:
    original_writer = csv.DictWriter

    def v7_writer(handle, fieldnames, *args, **kwargs):
        fields = list(fieldnames)
        if "generation" in fields:
            fields.extend(field for field in _HISTORY_FIELDS if field not in fields)
        return original_writer(handle, fields, *args, **kwargs)

    _cli.plan_drone_ca_diffusion_es = plan_drone_ca_diffusion_es
    csv.DictWriter = v7_writer
    try:
        _cli.main()
    finally:
        csv.DictWriter = original_writer


if __name__ == "__main__":
    main()
