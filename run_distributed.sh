#!/bin/bash
# Run the distributed optimization example

# Default values
WORKERS=4
ITERATIONS=100
DASHBOARD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --dashboard)
      DASHBOARD=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build dashboard argument
DASHBOARD_ARG=""
if [ "$DASHBOARD" = true ]; then
  DASHBOARD_ARG="--dashboard"
fi

# Run the example
echo "Running distributed optimization with $WORKERS workers, $ITERATIONS iterations"
if [ "$DASHBOARD" = true ]; then
  echo "Dashboard will be launched at http://localhost:8501"
fi

poetry run python examples/distributed_example.py --workers $WORKERS --iterations $ITERATIONS $DASHBOARD_ARG