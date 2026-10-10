# Simulator launcher: bwrap UID-map permission denied

If the launcher prints `bwrap: setting up uid map: Permission denied`, the sandbox has failed before starting Python or Isaac Sim. Activating a different Conda environment does not address this error.

On this machine, the 2026-10-10 kernel audit log confirmed a transition from `unconfined` to `unprivileged_userns`, followed by denials of `setpcap` and writing the child's `uid_map`. `kernel.apparmor_restrict_unprivileged_userns` is enabled. The earlier agent tests ran under `vscode`, whose installed profile explicitly allows `userns`. Those tests therefore did not establish that a normal terminal could start the launcher.

Ubuntu documents per-application profiles for this restriction in its [24.04 release notes](https://discourse.ubuntu.com/t/ubuntu-24-04-lts-noble-numbat-release-notes/39890/1). The prepared [pivla-bwrap profile](../configs/apparmor/pivla-bwrap) allows user namespaces for `/usr/bin/bwrap`. This is a system policy change applying to that executable for all users; it does not install or modify Python packages or disable the system-wide user-namespace restriction. Bubblewrap still enforces the launcher's read-only mounts.

## One-time administrator step

Review the small profile, then run from the project root:

```bash
cd /home/user/pivla
sudo install -o root -g root -m 0644 configs/apparmor/pivla-bwrap /etc/apparmor.d/pivla-bwrap
sudo apparmor_parser -r /etc/apparmor.d/pivla-bwrap
```

These commands require administrator privileges because `/etc/apparmor.d` and the loaded AppArmor policy are system-managed. The profile has been syntax-checked locally; the agent has not installed or loaded it. If a system administrator already manages a profile attached to `/usr/bin/bwrap`, have them incorporate the permission there rather than loading a competing attachment.

Run the simulator as your normal user:

```bash
bash scripts/run_isaac.sh scripts/m1_hover.py --enable_cameras
```

Do not run Isaac Sim with `sudo`. The launcher now checks namespace availability first and links here on failure, while retaining its read-only protection. A GPU run succeeding inside an IDE or agent process does not verify permission in a separate terminal; retry from the terminal that originally failed.

## Remove the added permission

After stopping your simulator process, the added profile can be unloaded and removed:

```bash
sudo apparmor_parser -R /etc/apparmor.d/pivla-bwrap
sudo rm /etc/apparmor.d/pivla-bwrap
```

This restores the previous launcher restriction for new processes from an ordinary terminal.
