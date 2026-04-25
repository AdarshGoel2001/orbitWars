"""Compatibility wrapper for :mod:`orbit_wars.cpu.bench`."""

from orbit_wars.cpu.bench import *  # noqa: F401,F403
from orbit_wars.cpu.bench import main


if __name__ == "__main__":
    main()

