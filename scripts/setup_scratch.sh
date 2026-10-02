#!/usr/bin/env bash
# Set up one verified Azure local NVMe disk as disposable scratch storage.
# The disk is ephemeral: keep code, manifests, annotations, and results elsewhere.
set -euo pipefail

device=/dev/disk/azure/local/by-index/1
target=/mnt/ai-swarm-scratch

[[ -b "$device" ]] || { printf 'Azure local disk is missing.\n' >&2; exit 1; }
[[ "$(lsblk -dn -o MODEL "$device" | sed 's/[[:space:]]*$//')" == 'Microsoft NVMe Direct Disk v2' ]] || {
  printf 'Device is not an Azure local NVMe disk.\n' >&2; exit 1;
}
[[ "$(readlink -f "$device")" != "$(readlink -f /dev/disk/azure/os)" ]] || {
  printf 'Refusing to use the OS disk.\n' >&2; exit 1;
}

if mountpoint -q "$target"; then
  source_device=$(findmnt -n -o SOURCE --target "$target")
  [[ "$(readlink -f "$source_device")" == "$(readlink -f "$device")" ]] || {
    printf 'Scratch path is mounted from another device.\n' >&2; exit 1;
  }
else
  fs_type=$(lsblk -dn -o FSTYPE "$device" | tr -d '[:space:]')
  case "$fs_type" in
    '')
      signatures=$(sudo -n wipefs --no-act "$device")
      [[ -z "$signatures" ]] || {
        printf 'Refusing to format a disk with existing signatures.\n' >&2; exit 1;
      }
      sudo -n mkfs.ext4 -F -m 0 -L ai-swarm-scratch "$device" >/dev/null
      ;;
    ext4) ;;
    *) printf 'Unexpected filesystem on scratch disk.\n' >&2; exit 1 ;;
  esac
  sudo -n mkdir -p "$target"
  sudo -n mount -t ext4 "$device" "$target"
fi

sudo -n chown azureuser:azureuser "$target"
df -h "$target"
