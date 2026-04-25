"""Compatibility wrapper for :mod:`orbit_wars.apps.server`."""

from orbit_wars.apps.server import *  # noqa: F401,F403
from orbit_wars.apps.server import app, new_env


if __name__ == "__main__":
    new_env()
    app.run(host="127.0.0.1", port=5000, debug=False)

