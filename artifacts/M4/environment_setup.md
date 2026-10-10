# M4 environment preparation

Completed on 2026-10-10 with the user's explicit authorization to modify the existing `lerobot` environment. This completes dependency preparation, not M4's training milestone.

## Changes

Installed Transformers 5.5.4, tokenizers 0.22.2, and eight supporting packages: regex, typer, annotated-doc, rich, Pygments, markdown-it-py, mdurl, and shellingham. Exact versions are in [added_requirements.txt](added_requirements.txt).

The local `/home/user/lerobot/pyproject.toml` declares `transformers>=5.4.0,<5.6.0` for the `pi` extra; its SciPy requirement and the training/dataset dependencies were already satisfied. No LeRobot reinstall, source update, PyTorch/CUDA replacement, or environment clone was needed. PEFT/LoRA dependencies have not been added because the training method has not yet been selected.

The resolver was constrained to every pre-existing package version. The reviewed dry run proposed only ten additions. Installation used those exact versions with the same constraints. All **111 pre-existing package versions were preserved**, and Isaac Lab's package/editable-link metadata fingerprint is unchanged. See [environment_changes.json](environment_changes.json).

## Verification

- `pip check`: no broken requirements before or after installation.
- π0 configuration, model, processor, Gemma and PaliGemma imports passed.
- `lerobot.scripts.lerobot_train --help` exited successfully.
- CUDA BF16 matrix multiplication passed on the RTX 5090 using the existing PyTorch 2.11.0+cu130.
- The M3 `ChunkDataset` still loads an observation of shape `[8]` and an action chunk of shape `[50,7]`.

See [environment_smoke.json](environment_smoke.json). Checks used the read-only runtime launcher with offline Hugging Face mode. Full local logs, pip resolver/install reports, and before/after package lists are retained here but ignored by Git.

## Commands used

After recording installed package versions in the local `preserve_installed.txt`:

```bash
/home/user/miniconda3/envs/lerobot/bin/python -B -m pip install \
  --dry-run --no-cache-dir --report artifacts/M4/pip_plan.json \
  --constraint artifacts/M4/preserve_installed.txt 'transformers>=5.4.0,<5.6.0'

# added_requirements.txt was generated from the reviewed resolver plan.
/home/user/miniconda3/envs/lerobot/bin/python -B -m pip install \
  --no-cache-dir --report artifacts/M4/pip_install.json \
  --constraint artifacts/M4/preserve_installed.txt \
  --requirement artifacts/M4/added_requirements.txt

/home/user/miniconda3/envs/lerobot/bin/python -B -m pip --no-cache-dir check
bash scripts/run_lerobot.sh -m lerobot.scripts.lerobot_train --help
```

The constraints describe this machine at installation time. Regenerate them from the current environment before applying this procedure to a different installation.

## Follow-up verification

At the end of dependency preparation, model acquisition and training were still pending. They have since passed: pretrained weights and the approved tokenizer were downloaded, the M3 preprocessing adapter was verified, and a two-episode overfit and checkpoint reload completed successfully. See the [M4 training report](report.md) for measurements and reproduction commands. The runtime launcher remains offline; downloads were performed separately.
