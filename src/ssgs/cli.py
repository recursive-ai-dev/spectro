"""Spectro command-line audio workbench."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__
from . import workflows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectro", description="Train spectral-state models and synthesize new audio.")
    parser.add_argument("--version", action="version", version=f"Spectro {__version__}")
    parser.add_argument("--verbose", action="store_true", help="show detailed processing logs")
    commands = parser.add_subparsers(dest="command")
    train = commands.add_parser("train", help="learn from audio files or folders")
    train.add_argument("inputs", nargs="+", help="audio files or folders")
    train.add_argument("--output", "-o", required=True, help=".npz or .safetensors model")
    train.add_argument("--states", type=int, default=16)
    train.add_argument("--iterations", type=int, default=10)
    train.add_argument("--sample-rate", type=int, default=16000, help="analysis rate in Hz (max 768000)")
    train.add_argument("--lpc-order", type=int, default=12)
    train.add_argument("--frame-size", type=int, default=1024)
    train.add_argument("--hop-size", type=int, default=256)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--recursive", "-r", action="store_true")
    train.add_argument("--force", action="store_true", help="replace an existing output")
    adapt = commands.add_parser("adapt", help="update a saved model with new audio, saving a separate model")
    adapt.add_argument("model", help="input .npz or .safetensors model")
    adapt.add_argument("inputs", nargs="+", help="new audio files or folders")
    adapt.add_argument("--output", "-o", required=True, help="separate output .npz or .safetensors model")
    adapt.add_argument("--adaptation-rate", type=float, default=0.25, help="weight of new statistics, 0–1")
    adapt.add_argument("--stability-bias", type=float, default=0.05, help="regularization strength, 0–1")
    adapt.add_argument("--memory-blend", type=float, default=0.7, help="fraction of memory reserved for earlier frames, 0–1")
    adapt.add_argument("--memory-limit", type=int, default=6000, help="maximum cached training frames")
    adapt.add_argument("--iterations", type=int, default=1, help="EM refinement iterations per new recording")
    adapt.add_argument("--seed", type=int, default=0)
    adapt.add_argument("--recursive", "-r", action="store_true")
    adapt.add_argument("--force", action="store_true")
    adapt.add_argument("--json", action="store_true", help="emit per-recording adaptation metrics as JSON")
    generate = commands.add_parser("generate", help="render a model to a 24-bit WAV")
    generate.add_argument("model")
    generate.add_argument("--output", "-o", required=True)
    generate.add_argument("--duration", type=float, default=3.0, help="duration in seconds (default: 3; max 3600)")
    generate.add_argument(
        "--sample-rate", type=int, default=22050,
        help="output rate in Hz (max 768000, and at most 268435456 samples per render)",
    )
    generate.add_argument("--method", choices=("stochastic", "viterbi"), default="stochastic")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--fidelity", type=float, default=0.0)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--excitation", choices=("Karplus-Strong", "Wavelet"), default="Karplus-Strong")
    generate.add_argument("--force", action="store_true")
    inspect = commands.add_parser("inspect", help="view model settings and training statistics")
    inspect.add_argument("model")
    inspect.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    demo = commands.add_parser("demo", help="train and generate using a built-in synthetic signal")
    demo.add_argument("--output-dir", default="spectro-demo")
    demo.add_argument("--force", action="store_true")
    doctor = commands.add_parser("doctor", help="report runtime, audio format and backend availability")
    doctor.add_argument("--json", action="store_true")
    commands.add_parser("gui", help="open the desktop audio workbench")
    return parser


def _progress(message: str, fraction: float) -> None:
    print(f"[{fraction:4.0%}] {message}", file=sys.stderr)


def _display_info(info: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        for name, value in info.items():
            label = name.replace("_", " ").capitalize()
            print(f"{label}: {json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        if args.command == "train":
            model = workflows.train_model(
                args.inputs, args.output, n_states=args.states, iterations=args.iterations,
                sample_rate=args.sample_rate, lpc_order=args.lpc_order,
                frame_size=args.frame_size, hop_size=args.hop_size, seed=args.seed,
                recursive=args.recursive, force=args.force, progress=_progress,
            )
            print(f"Saved model: {args.output} ({model.n_states} states)")
        elif args.command == "adapt":
            _, metrics = workflows.adapt_model(
                args.model, args.inputs, args.output, adaptation_rate=args.adaptation_rate,
                stability_bias=args.stability_bias, memory_blend=args.memory_blend,
                memory_limit=args.memory_limit, iterations=args.iterations, seed=args.seed,
                recursive=args.recursive, force=args.force, progress=_progress,
            )
            if args.json:
                print(json.dumps({"output": args.output, "recordings": metrics}, indent=2, sort_keys=True))
            else:
                print(f"Saved adapted model: {args.output} ({len(metrics)} recording(s))")
        elif args.command == "generate":
            output = workflows.generate_audio(
                args.model, args.output, duration=args.duration, sample_rate=args.sample_rate,
                method=args.method, seed=args.seed, fidelity=args.fidelity,
                temperature=args.temperature, excitation=args.excitation, force=args.force,
                progress=_progress,
            )
            print(f"Saved audio: {output}")
        elif args.command == "inspect":
            _display_info(workflows.model_info(args.model), args.json)
        elif args.command == "doctor":
            _display_info(workflows.doctor_info(), args.json)
        elif args.command == "demo":
            outputs = workflows.run_demo(args.output_dir, force=args.force, progress=_progress)
            for name, path in outputs.items():
                print(f"{name.capitalize()}: {path}")
        elif args.command == "gui":
            from .desktop import main as desktop_main
            return desktop_main([]) or 0
        return 0
    except KeyboardInterrupt:
        print("spectro: interrupted", file=sys.stderr)
        return 130
    except workflows.WorkflowCancelled as exc:
        print(f"spectro: {exc}", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, FloatingPointError, MemoryError, OverflowError) as exc:
        if args.verbose:
            logging.exception("Operation failed")
        else:
            print(f"spectro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
