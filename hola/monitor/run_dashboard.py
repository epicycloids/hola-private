#!/usr/bin/env python
"""
Run the HOLA optimization dashboard.

This script launches the Streamlit dashboard for monitoring HOLA optimization.
"""

import os
import sys
import streamlit

# Add the current directory to Python path if not already there
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the dashboard module
from hola.monitor.streamlit_dashboard import main

if __name__ == "__main__":
    main()