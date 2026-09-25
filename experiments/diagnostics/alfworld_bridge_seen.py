"""Validation-only ALFWorld bridge restricted to official valid_seen games."""

import argparse
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .alfworld_bridge_validation import EnvironmentDriver as BaseDriver
from .alfworld_bridge_validation import EnvironmentClient as BaseClient
from .alfworld_bridge_validation import open_game, serve


class EnvironmentDriver(BaseDriver):
    def __init__(self, train_root, factory=open_game):
        self.root = Path(train_root).resolve(strict=True)
        if self.root.name != "valid_seen":
            raise ValueError("Seen diagnostic only opens official valid_seen")
        self.factory = factory
        self.env = None
        self.done = True


class EnvironmentClient(BaseClient):
    def __init__(self, python, train_root, timeout=60):
        self.timeout = timeout
        self.sequence = 0
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            [str(python), "-u", "-m", "trustlab.alfworld_bridge_seen",
             "--train-root", str(train_root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            encoding="utf-8", bufsize=1,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True, type=Path)
    args = parser.parse_args()
    serve(EnvironmentDriver(args.train_root), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
