import sys
import glob
import site

print("Python site-packages directories:")
for path in site.getsitepackages():
    print(path)

print("\nAttempting imports:")
try:
    from websocket._app import WebSocketApp
    print("Successfully imported WebSocketApp from websocket._app")
except ImportError as e:
    print(f"Failed from websocket._app: {e}")

try:
    from websocket._core import WebSocketApp
    print("Successfully imported WebSocketApp from websocket._core")
except ImportError as e:
    print(f"Failed from websocket._core: {e}")

try:
    import websocket
    print("\nWebsocket import details:")
    print(f"  __version__: {getattr(websocket, '__version__', 'unknown')}")
    print(f"  __file__: {getattr(websocket, '__file__', 'unknown')}")
    print(f"  dir: {dir(websocket)}")

    if hasattr(websocket, "_app"):
        print("\nWebsocket has _app module:")
        print(f"  dir(websocket._app): {dir(websocket._app)}")
        if hasattr(websocket._app, "WebSocketApp"):
            print("  Found WebSocketApp in websocket._app!")
except Exception as e:
    print(f"Error importing websocket: {e}")