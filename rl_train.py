"""Compatibility wrapper for :mod:`orbit_wars.legacy.rl_train`."""

from orbit_wars.legacy.rl_train import *  # noqa: F401,F403
from orbit_wars.legacy.rl_train import main


if __name__ == "__main__":
    main()

