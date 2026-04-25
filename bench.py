"""Compatibility wrapper for :mod:`orbit_wars.legacy.bench`."""

from orbit_wars.legacy.bench import *  # noqa: F401,F403
from orbit_wars.legacy.bench import main


if __name__ == "__main__":
    main()

