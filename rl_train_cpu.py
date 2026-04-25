"""Compatibility wrapper for :mod:`orbit_wars.cpu.rl_train`."""

from orbit_wars.cpu.rl_train import *  # noqa: F401,F403
from orbit_wars.cpu.rl_train import main


if __name__ == "__main__":
    main()
