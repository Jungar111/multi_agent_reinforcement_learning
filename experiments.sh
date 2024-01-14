#!/usr/bin/env sh
poetry run python main_SAC.py --no-actors 1 --no-cars 100 --run-name "no-actors 1, 100" &&
poetry run python main_SAC.py --no-actors 1 --no-cars 187 --run-name "no-actors 1, half cars" &&
poetry run python main_SAC.py --no-actors 1 --no-cars 200 --run-name "no-actors 1, 200" &&
poetry run python main_SAC.py --no-actors 1 --no-cars 300 --run-name "no-actors 1, 300" &&
poetry run python main_SAC.py --no-actors 1 --run-name "no-actors 1" &&
poetry run python main_SAC.py --no-actors 1 --no-cars 400 --run-name "no-actors 1, 400" &&
poetry run python main_SAC.py --no-cars 100 --run-name "no-cars 100" &&
poetry run python main_SAC.py --no-cars 200 --run-name "no-cars 200" &&
poetry run python main_SAC.py --no-cars 300 --run-name "no-cars 300" &&
poetry run python main_SAC.py --no-cars 400 --run-name "no-cars 400" &&
poetry run python main_SAC.py --run-name "Orignal number of cars" &&
poetry run python main.py
