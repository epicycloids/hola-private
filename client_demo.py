import argparse
import json
import time
import requests
from typing import Dict, Any

"""
REST API client for optimization server.
This script demonstrates how to interact with the optimization server using HTTP requests.
"""

def main():
    parser = argparse.ArgumentParser(description='REST API client for optimization server')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Server hostname or IP address')
    parser.add_argument('--port', type=int, default=8000,
                        help='Server port number')
    args = parser.parse_args()

    # Base URL for the API
    base_url = f"http://{args.host}:{args.port}"
    print(f"Connecting to optimization server at {base_url}")

    # Get server status
    try:
        response = requests.get(f"{base_url}/status")
        status = response.json()
        print(f"Server status: {json.dumps(status, indent=2)}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    # Request optimization suggestions and submit results
    for i in range(5):  # Run 5 evaluations as an example
        print(f"\nEvaluation {i+1}:")

        # Get parameter suggestion
        try:
            response = requests.get(f"{base_url}/suggestion")
            data = response.json()

            if "error" in data and data["error"] is not None:
                print(f"Error getting suggestion: {data['error']}")
                break

            if data.get("parameters") is None:
                print("No more parameter suggestions available")
                break

            params = data["parameters"]
            print(f"Received parameters: {params}")

            # Simulate evaluation (in a real scenario, this would be your actual evaluation code)
            # Example function: objectives based on distance from origin
            x = params.get("x", 0)
            y = params.get("y", 0)

            # Calculate objectives (same as in the server example)
            import math
            import random

            # Simulate computation time
            time.sleep(random.uniform(0.5, 1.0))

            # Calculate objectives
            objective1 = math.exp(-(x**2 + y**2)/10)
            objective2 = (x - y)**2
            objective3 = math.sqrt(x**2 + y**2)

            objectives = {
                "objective1": objective1,
                "objective2": objective2,
                "objective3": objective3
            }

            print(f"Calculated objectives: {objectives}")

            # Submit results back to the server
            result = {
                "parameters": params,
                "objectives": objectives
            }

            response = requests.post(f"{base_url}/result", json=result)
            submit_result = response.json()

            if submit_result.get("success"):
                print("Result submitted successfully")
                if submit_result.get("is_best", False):
                    print("Found new best result!")
            else:
                print(f"Error submitting result: {submit_result.get('error')}")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            time.sleep(2)  # Wait before retrying

    # Get final status and best results
    try:
        response = requests.get(f"{base_url}/status")
        final_status = response.json()
        print("\nFinal server status:")
        print(json.dumps(final_status, indent=2))

        # Get top results
        response = requests.get(f"{base_url}/top?k=3")
        top_results = response.json()
        print("\nTop 3 results:")
        print(json.dumps(top_results, indent=2))
    except Exception as e:
        print(f"Error getting final results: {e}")

if __name__ == "__main__":
    main()