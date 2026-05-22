import argparse
import subprocess
import sys


CORE_PLOTS = (
    "scripts.plot_localizer_decode",
    "scripts.plot_localizers",
    "scripts.plot_localizers_hierarchy",
    "scripts.plot_motion_object",
    "scripts.plot_neural_alignment",
    "scripts.plot_smoothness",
    "scripts.plot_wiring_cost",
    "scripts.plot_localizers_cortex",
    "scripts.plot_hierarchy_alignment",
)

SLOW_PLOTS = (
    "scripts.plot_localizer_motion",
    "scripts.plot_localizers_seed",
    "scripts.plot_wiring_cost_fmri",
)

EXTRA_PLOTS = (
    "scripts.plot_alignment_task_loss_timeseries",
    "scripts.plot_alignment_task_loss_timeseries2",
    "scripts.plot_autocorr",
    "scripts.plot_konkle_localizers",
    "scripts.plot_robert_distribution",
    "scripts.plot_single_sheet",
    "scripts.plot_stimulation",
)


def _config_id(model_name, data_name, lr, batch_size=32, seed=42, prefix=""):
    lr_str = (
        f"lr{lr}".replace(".", "p")
        if lr >= 0.001
        else f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    )
    return f"{prefix}{model_name}_{data_name}_{lr_str}_bs{batch_size}_sd{seed}"


def _run(command, dry_run=False):
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command).returncode


def _checkpoint_name(args):
    layer_id = "_".join(str(idx) for idx in args.layer_indices)
    model_name = f"{args.model_name}_{layer_id}_single"
    if args.tissue_config != "small" and args.inf_neighborhood:
        model_name += "_neighbInf"
    if args.tissue_config != "vtc":
        model_name += f"_{args.tissue_config}"
    rf_overlap = args.rf_overlap
    if args.tissue_config == "small" and rf_overlap is None:
        rf_overlap = 0.1
    if rf_overlap is not None:
        model_name += f"_rf{str(rf_overlap).replace('.', 'p')}"
    config_id = _config_id(
        model_name,
        args.data_name,
        args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        prefix="global_",
    )
    if args.train_split != "all":
        config_id += f"_{args.train_split}"
    return f"best_transformed_model_{config_id}.pt"


def _train_command(args):
    command = [
        sys.executable,
        "train.py",
        "--model_name",
        args.model_name,
        "--data_name",
        args.data_name,
        "--lr",
        str(args.lr),
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--samples_per_batch",
        str(args.samples_per_batch),
        "--tissue_config",
        args.tissue_config,
        "--train_split",
        args.train_split,
        "--train_split_seed",
        str(args.train_split_seed),
        "--seed",
        str(args.seed),
        "--layer_indices",
        *[str(idx) for idx in args.layer_indices],
    ]
    if args.resume_training:
        command.append("--resume_training")
    if args.rf_overlap is not None:
        command.extend(["--rf_overlap", str(args.rf_overlap)])
    if not args.inf_neighborhood:
        command.append("--no-inf_neighborhood")
    if args.use_wandb:
        command.append("--use_wandb")
    return command


def _som_train_command(args):
    command = [
        sys.executable,
        "train_som.py",
        "--data_name",
        args.data_name,
        "--num_epochs",
        str(args.som_num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--samples_per_batch",
        str(args.som_samples_per_batch),
        "--som_batch_size",
        str(args.som_batch_size),
        "--seed",
        str(args.seed),
        "--unit_mm",
        str(args.som_unit_mm),
        "--layer_indices",
        *[str(idx) for idx in args.som_layer_indices],
    ]
    if args.som_max_batches is not None:
        command.extend(["--max_batches", str(args.som_max_batches)])
    if args.resume_som:
        command.append("--resume")
    return command


def _eval_command(args):
    checkpoint_name = args.checkpoint_name or _checkpoint_name(args)
    return [
        sys.executable,
        "eval.py",
        "--checkpoint_name",
        checkpoint_name,
        "--dataset_names",
        *args.eval_datasets,
        "--batch_size",
        str(args.eval_batch_size),
        "--fwhm_mm",
        str(args.fwhm_mm),
        "--resolution_mm",
        str(args.resolution_mm),
    ]


def _plot_modules(args):
    modules = []
    if args.plot_group in {"core", "all"}:
        modules.extend(CORE_PLOTS)
    if args.plot_group in {"slow", "all"}:
        modules.extend(SLOW_PLOTS)
    if args.plot_group in {"extra", "all"}:
        modules.extend(EXTRA_PLOTS)
    if args.plots:
        modules.extend(args.plots)
    return tuple(dict.fromkeys(modules))


def _plot_commands(args):
    return [[sys.executable, "-m", module] for module in _plot_modules(args)]


def get_args():
    parser = argparse.ArgumentParser(
        description="Run TopoTransform training, evaluation, and plot reproduction commands."
    )
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "plots", "all"],
        default="plots",
        help="Default uses existing cache/checkpoints and reproduces plots.",
    )
    parser.add_argument(
        "--plot_group",
        choices=["core", "slow", "extra", "all", "none"],
        default="core",
        help="Plot set to run when stage includes plots.",
    )
    parser.add_argument(
        "--plots",
        nargs="*",
        default=(),
        help="Additional plot modules, e.g. scripts.plot_robert_distribution.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")

    parser.add_argument("--data_name", default="kinetics400", choices=["smthsmthv2", "kinetics400", "imagenet"])
    parser.add_argument("--model_name", default="vjepa", choices=["vjepa", "clip", "videomae"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_batch", type=int, default=8192 * 2)
    parser.add_argument("--tissue_config", choices=["vtc", "small"], default="vtc")
    parser.add_argument("--rf_overlap", type=float, default=None)
    parser.add_argument("--inf_neighborhood", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_split", choices=["all", "half_a", "half_b"], default="all")
    parser.add_argument("--train_split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[14, 18, 22])
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--train_som", action="store_true", help="Train the SOM baseline before evaluation/plots.")
    parser.add_argument("--resume_som", action="store_true")
    parser.add_argument("--som_layer_indices", type=int, nargs="+", default=[22])
    parser.add_argument("--som_num_epochs", type=int, default=3)
    parser.add_argument("--som_samples_per_batch", type=int, default=4096)
    parser.add_argument("--som_batch_size", type=int, default=256)
    parser.add_argument("--som_max_batches", type=int, default=None)
    parser.add_argument("--som_unit_mm", type=float, default=1.0)

    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--eval_datasets", nargs="+", default=["vpnl", "biomotion", "kanwisher", "pitzalis"])
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--fwhm_mm", type=float, default=1.0)
    parser.add_argument("--resolution_mm", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = get_args()
    commands = []

    if args.stage in {"train", "all"}:
        commands.append(_train_command(args))
    if args.train_som:
        commands.append(_som_train_command(args))
    if args.stage in {"eval", "all"}:
        commands.append(_eval_command(args))
    if args.stage in {"plots", "all"} and args.plot_group != "none":
        commands.extend(_plot_commands(args))

    failures = []
    for command in commands:
        returncode = _run(command, dry_run=args.dry_run)
        if returncode != 0:
            failures.append((returncode, command))
            if args.fail_fast:
                break

    if failures:
        print("\nFailed commands:", flush=True)
        for returncode, command in failures:
            print(f"[{returncode}] {' '.join(command)}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
