#!/usr/bin/env sh
poetry run python main_SAC.py --no-actors 1 --no-cars 100 --run-name "no-actors 1, 100" &&
poetry run python main_SAC.py --no-actors 1 --no-cars 200 --run-name "no-actors 1, 200" &&
poetry run python main_SAC.py --no-actors 1 --no-cars 300 --run-name "no-actors 1, 300"
