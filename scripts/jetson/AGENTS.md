# Jetson script rules

Scripts in this directory are review/dry-run material unless a human explicitly runs
them on the target device.

Automated agents must never execute `sudo`, `apt upgrade`, flashing, disk formatting,
power-mode changes, external-network actions, or Jetson SSH. Commands that could
change JetPack, L4T, CUDA, TensorRT, storage, or networking require a reviewed human
procedure.
