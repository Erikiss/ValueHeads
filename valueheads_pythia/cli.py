"""Command dispatcher: `python -m valueheads_pythia.cli <command> [args]`.

Each command lives in valueheads_pythia/commands/<command>.py and exposes add_args(parser) and run(args).
Dashes in command names map to underscores (eval-likelihood -> commands/eval_likelihood.py). Only the
requested command module is imported, so commands can be added independently.
"""
import argparse
import importlib
import os
import sys

from . import config

COMMANDS = {
    "extract": "Schritt 1: Head-Aktivierungen extrahieren (layer-major fp16-Shards, DONE-Marker, resume)",
    "probe": "Schritt 2: eine logistische Regression je Head, Ranking, Top-k",
    "probe-stats": "Schritt 2b: Bootstrap-CIs, Permutations-Null, signifikante Menge, Konfirmations-Split",
    "directions": "Mass-Mean-Richtungen + sigma je STD-Set",
    "eval-likelihood": "Likelihood-Metriken (TQA pairwise/MC, HH, HelpSteer2, WikiText-PPL) unter Base/TopK/Random/Steer",
    "eval-generate": "Freie Generation (paper | fewshot_qa | qa_plain) unter Interventionen -> jsonl",
    "judge": "Judge-Stack je Stufe auf generierten Antworten",
    "sweep": "Checkpoint-Sweep (extract -> probe -> directions je Revision)",
    "analyze": "Heatmaps, Sweep-Kurven, Korrelationen, Report",
    "selftest": "Hook-Layout, Masken-Positionen, Datenzaehlungen",
}


def command_module_name(command):
    return command.replace("-", "_")


def load_command(command):
    name = command_module_name(command)
    try:
        return importlib.import_module(f"valueheads_pythia.commands.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"valueheads_pythia.commands.{name}":
            raise SystemExit(f"command {command!r} is not available (module valueheads_pythia/commands/{name}.py missing)")
        raise


def add_common_args(parser, model=True, root=True, device=True, mode=True, seed=True):
    """Arguments shared by most commands (helper for command modules; use is optional)."""
    if model:
        parser.add_argument("--model", default="pythia-160m", help="registry short name (config.MODELS) or HF id")
        parser.add_argument("--revision", default=None, help="HF revision, e.g. step143000 (default: main)")
    if root:
        parser.add_argument("--root", default=None, help="run directory (default: env VH_ROOT or ./vh_runs)")
    if device:
        parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
        parser.add_argument("--dtype", default=None, help="float32 | float16 | bfloat16 (default: numerics policy)")
        parser.add_argument("--attn_impl", default=None, help="eager | sdpa (default: numerics policy)")
    if mode:
        parser.add_argument("--mode", default="pythia", choices=config.MODES, help="paper = repo definitions, pythia = DESIGN 3-5")
    if seed:
        parser.add_argument("--seed", type=int, default=42)
    return parser


def usage():
    lines = ["usage: python -m valueheads_pythia.cli <command> [args]", "", "commands:"]
    for name, desc in COMMANDS.items():
        avail = os.path.exists(os.path.join(os.path.dirname(__file__), "commands", command_module_name(name) + ".py"))
        lines.append(f"  {name:16s} {desc}{'' if avail else '  [nicht installiert]'}")
    lines.append("")
    lines.append("python -m valueheads_pythia.cli <command> --help  zeigt die Argumente eines Befehls")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    command = argv[0]
    if command not in COMMANDS and command_module_name(command) not in [command_module_name(c) for c in COMMANDS]:
        print(f"unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = load_command(command)
    parser = argparse.ArgumentParser(prog=f"python -m valueheads_pythia.cli {command}", description=COMMANDS.get(command))
    module.add_args(parser)
    args = parser.parse_args(argv[1:])
    result = module.run(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
