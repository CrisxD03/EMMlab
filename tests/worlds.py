"""Reference Worlds used across the test suite (all from research/04-path-model-math.md)."""

from __future__ import annotations

from pathlib import Path

from emmlab.spec import World, load_world

PRESETS = Path(__file__).resolve().parents[1] / "src" / "emmlab" / "presets"


def world_a() -> World:
    return load_world(PRESETS / "minimal.yaml")


def world_b() -> World:
    d = world_a().model_dump(by_alias=True)
    d["name"] = "world_b"
    d["nodes"] = {"climate": {"level": "school", "root": {"mean": 0, "sd": 0.4}}, **d["nodes"]}
    d["arrows"] += [{"from": "climate", "to": "feedback", "slope": 1.0},
                    {"from": "climate", "to": "achievement", "slope": 0.5}]
    d["attributes"]["gender"] = {"level": "student", "values": {"m": 0.5, "f": 0.5}}
    d["shifts"].append({"when": {"gender": "f"}, "target": "selfeff", "delta": 0.2})
    d["twists"].append({"arrow": "selfeff -> achievement", "when": {"gender": "f"}, "delta": 0.15})
    return World.model_validate(d)


def world_simpson() -> World:
    return World.model_validate({
        "name": "simpson",
        "attributes": {"ses": {"level": "school", "values": {"poor": 0.5, "rich": 0.5}}},
        "nodes": {"feedback": {"recipe": {"noise_sd": 1.0}}, "achievement": {"recipe": {"noise_sd": 1.0}}},
        "arrows": [{"feedback -> achievement": -0.20}],
        "shifts": [{"when": {"ses": "rich"}, "target": "feedback", "delta": 2.0},
                   {"when": {"ses": "rich"}, "target": "achievement", "delta": 2.0}],
    })


def world_pcase() -> World:
    """Mediator with another parent (R4 §3.4): residualization needs the total-effect rule."""
    return World.model_validate({
        "name": "pcase",
        "nodes": {"motivation": {"root": {"mean": 0, "sd": 1}},
                  "feedback": {"recipe": {"noise_sd": 1.0}}, "selfeff": {"recipe": {"noise_sd": 0.9}},
                  "achievement": {"recipe": {"noise_sd": 0.8}}},
        "arrows": [{"motivation -> feedback": 0.5}, {"motivation -> selfeff": 0.4}, {"feedback -> selfeff": 0.3},
                   {"selfeff -> achievement": 0.25}, {"feedback -> achievement": 0.2},
                   {"motivation -> achievement": 0.1}],
    })
