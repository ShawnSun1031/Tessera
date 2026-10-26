#!/bin/bash

source .venv/bin/activate
ruff check ./backend --fix
pyright -v ./backend