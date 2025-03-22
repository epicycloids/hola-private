#!/bin/bash

cd "$(dirname "$0")/.."
poetry run streamlit run examples/streamlit_monitor.py