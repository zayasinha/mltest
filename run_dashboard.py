#!/usr/bin/env python3
"""
ML Project Dashboard Runner
Run this script to start the Streamlit dashboard
"""

import subprocess
import sys
import os

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to dashboard.py
    dashboard_path = os.path.join(script_dir, 'dashboard.py')

    # Command to run streamlit
    cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path]

    print("Starting ML Project Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    print("Dashboard will be available at http://localhost:8501")

    try:
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main()