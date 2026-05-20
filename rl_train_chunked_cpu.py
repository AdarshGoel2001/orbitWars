"""Compatibility wrapper for :mod:`orbit_wars.cpu.rl_train_chunked`."""

from orbit_wars.cpu.rl_train_chunked import *  # noqa: F401,F403
from orbit_wars.cpu.rl_train_chunked import main


if __name__ == "__main__":
    main()
