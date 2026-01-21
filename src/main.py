import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.run is None:
        raise ValueError("run=<run_id> must be supplied")
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    results_dir = Path(cfg.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={results_dir}",
        f"mode={cfg.mode}",
    ]
    print("[main] Launching:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
