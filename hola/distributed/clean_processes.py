#!/usr/bin/env python
"""
Utility script to kill any lingering processes from a distributed optimization run.

This script will find and kill any streamlit, ZMQ, and other related processes
that might have been left running from a previous optimization.
"""

import os
import sys
import subprocess
import argparse
import time
import signal

def find_processes(name_patterns):
    """Find processes matching any of the given name patterns."""
    all_pids = []

    for pattern in name_patterns:
        try:
            # Use ps command to find processes containing the pattern
            cmd = ["ps", "aux"]
            ps_output = subprocess.check_output(cmd, universal_newlines=True)

            # Process each line
            for line in ps_output.split('\n'):
                if pattern in line and "grep" not in line:
                    # Extract PID (second column in ps aux output)
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            all_pids.append((pid, line))
                        except ValueError:
                            pass
        except subprocess.SubprocessError as e:
            print(f"Error finding processes: {e}")

    return all_pids

def kill_processes(processes, dry_run=False, force=False):
    """Kill the given processes."""
    for pid, cmdline in processes:
        sig = signal.SIGKILL if force else signal.SIGTERM
        signal_name = "SIGKILL" if force else "SIGTERM"

        if dry_run:
            print(f"Would kill process {pid} with {signal_name}: {cmdline}")
        else:
            print(f"Killing process {pid} with {signal_name}: {cmdline}")
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                print(f"Process {pid} no longer exists")
            except PermissionError:
                print(f"Permission denied for process {pid}")

def main():
    parser = argparse.ArgumentParser(description="Kill ghost processes from optimization runs")
    parser.add_argument("--dry-run", action="store_true", help="Show processes without killing them")
    parser.add_argument("--force", action="store_true", help="Use SIGKILL instead of SIGTERM")
    args = parser.parse_args()

    # Define patterns for processes to kill
    patterns = [
        "streamlit",
        "zmq_monitor_example",
        "zmq_example.py",
        "/tmp/hola-optimization.ipc"
    ]

    print("Searching for processes to clean up...")
    processes = find_processes(patterns)

    if not processes:
        print("No matching processes found.")
        return

    print(f"Found {len(processes)} processes to clean up:")
    for pid, cmdline in processes:
        print(f"  PID {pid}: {cmdline}")

    kill_processes(processes, dry_run=args.dry_run, force=args.force)

    if not args.dry_run:
        # Check if any processes remain after a short delay
        time.sleep(1)
        remaining = find_processes(patterns)

        if remaining:
            print(f"\n{len(remaining)} processes still running after termination:")
            for pid, cmdline in remaining:
                print(f"  PID {pid}: {cmdline}")

            if not args.force:
                print("\nUse --force to forcefully kill these processes")
        else:
            print("\nAll processes successfully terminated")

    # Clean up socket file
    socket_file = "/tmp/hola-optimization.ipc"
    if os.path.exists(socket_file):
        if args.dry_run:
            print(f"Would remove socket file: {socket_file}")
        else:
            try:
                os.remove(socket_file)
                print(f"Removed socket file: {socket_file}")
            except Exception as e:
                print(f"Error removing socket file: {e}")

if __name__ == "__main__":
    main()