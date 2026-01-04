#!/usr/bin/env python3
"""
BMW Cars Data Exploration Dashboard Runner
Run this script to start the comprehensive data exploration dashboard
"""

import subprocess
import sys
import os

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to data exploration dashboard
    dashboard_path = os.path.join(script_dir, 'data_exploration_dashboard.py')

    # Command to run streamlit
    cmd = [sys.executable, '-m', 'streamlit', 'run', dashboard_path, '--server.headless', 'true']

    print("🚗 Starting BMW Cars Data Exploration Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    print("Dashboard will be available at http://localhost:8501")
    print("\n📊 This dashboard includes:")
    print("• Dataset Overview & Statistics")
    print("• Price Analysis & Distributions")
    print("• Model Insights & Comparisons")
    print("• Fuel & Transmission Analysis")
    print("• Year Trends & Time Analysis")
    print("• Feature Correlations")
    print("• Outlier Detection")
    print("• Key Business Insights")

    try:
        subprocess.run(cmd, cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main()