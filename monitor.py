import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
import json
import threading
import queue
import atexit
import os
import pickle

# Import WebSocketApp correctly
from websocket._app import WebSocketApp


class OptimizationMonitor:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.cache_file = ".monitor_cache.pkl"

        # WebSocket connection
        self.ws_url = f"ws://{api_base_url.split('//')[1]}/ws"
        self.ws = None
        self.ws_connected = False
        self.message_queue = queue.Queue()
        self.websocket_thread = None
        self.running = True
        self.thread_lock = threading.Lock()

        # Thread-safe data storage (not directly using session state from threads)
        self.local_history = []
        self.local_total_evaluations = 0
        self.local_active_workers = 0
        self.local_best_result = None
        self.local_connected = False
        self.local_connection_attempts = 0

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'connected' not in st.session_state:
            st.session_state.connected = False
        if 'total_evaluations' not in st.session_state:
            st.session_state.total_evaluations = 0
        if 'active_workers' not in st.session_state:
            st.session_state.active_workers = 0
        if 'best_result' not in st.session_state:
            st.session_state.best_result = None
        if 'connection_attempts' not in st.session_state:
            st.session_state.connection_attempts = 0

        # Load cached data if available
        self.load_cache()

        # Synchronize local data with session state
        self.sync_from_session_state()

        # Fetch initial status
        self.get_initial_status()

        # Start WebSocket connection
        self.connect_websocket()

        # Register cleanup function
        atexit.register(self.cleanup)

    def sync_to_session_state(self):
        """Synchronize local data to session state (thread-safe method)"""
        with self.thread_lock:
            st.session_state.history = self.local_history.copy()
            st.session_state.total_evaluations = self.local_total_evaluations
            st.session_state.active_workers = self.local_active_workers
            st.session_state.best_result = self.local_best_result
            st.session_state.connected = self.local_connected
            st.session_state.connection_attempts = self.local_connection_attempts

    def sync_from_session_state(self):
        """Synchronize session state to local data (thread-safe method)"""
        with self.thread_lock:
            self.local_history = st.session_state.history.copy()
            self.local_total_evaluations = st.session_state.total_evaluations
            self.local_active_workers = st.session_state.active_workers
            self.local_best_result = st.session_state.best_result
            self.local_connected = st.session_state.connected
            self.local_connection_attempts = st.session_state.connection_attempts

    def load_cache(self):
        """Load cached data from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                # Update session state with cached data if needed
                if not st.session_state.history and 'history' in cached_data:
                    st.session_state.history = cached_data.get('history', [])
                    print(f"Loaded {len(st.session_state.history)} items from cache")

                if st.session_state.total_evaluations == 0:
                    st.session_state.total_evaluations = cached_data.get('total_evaluations', 0)

                if not st.session_state.best_result:
                    st.session_state.best_result = cached_data.get('best_result')

                print(f"Cache loaded: {len(st.session_state.history)} history items")

                # Synchronize with local data
                self.sync_from_session_state()
        except Exception as e:
            print(f"Error loading cache: {e}")

    def save_cache(self):
        """Save current state to cache file"""
        try:
            # Ensure local data is up to date with session state
            self.sync_from_session_state()

            cache_data = {
                'history': self.local_history,
                'total_evaluations': self.local_total_evaluations,
                'best_result': self.local_best_result
            }

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"Cache saved: {len(self.local_history)} history items")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def cleanup(self):
        """Cleanup resources on shutdown"""
        print("Performing cleanup...")
        self.running = False

        # Save cache before exiting
        self.save_cache()

        # Close WebSocket connection
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.ws = None

    def get_initial_status(self):
        """Get the initial optimization status from the server"""
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=2)
            if response.status_code == 200:
                try:
                    status = response.json()

                    with self.thread_lock:
                        self.local_total_evaluations = status.get('total_evaluations', 0)
                        self.local_active_workers = status.get('active_workers', 0)

                        best_result = status.get('best_result')
                        if best_result and 'objectives' in best_result:
                            self.local_best_result = best_result

                        # Mark as connected if we got a successful response
                        self.local_connected = True

                    # Update session state with our thread-safe data
                    self.sync_to_session_state()
                    print(f"Initial status: {status}")

                except json.JSONDecodeError as e:
                    print(f"Failed to parse server response: {e}")
            else:
                print(f"Server returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to connect to server: {e}")

    def connect_websocket(self):
        """Establish WebSocket connection to the server"""
        def on_message(ws, message):
            if not self.running:
                return

            try:
                data = json.loads(message)
                self.message_queue.put(data)

                # Thread-safe update of connection status
                with self.thread_lock:
                    self.local_connected = True
                    self.local_connection_attempts = 0

                # Auto-save cache periodically when receiving data
                if len(self.local_history) % 10 == 0 and len(self.local_history) > 0:
                    self.save_cache()
            except json.JSONDecodeError as e:
                print(f"Failed to parse WebSocket message: {e}")

        def on_error(ws, error):
            if not self.running:
                return

            # Thread-safe update of connection status
            with self.thread_lock:
                self.local_connected = False

            print(f"WebSocket error: {error}")
            # Don't reconnect too rapidly
            time.sleep(0.5)

        def on_close(ws, close_status_code, close_msg):
            if not self.running:
                return

            # Thread-safe update of connection status
            with self.thread_lock:
                self.local_connected = False

            print(f"WebSocket closed: {close_status_code} - {close_msg}")

        def on_open(ws):
            if not self.running:
                return

            # Thread-safe update of connection status
            with self.thread_lock:
                self.local_connected = True
                self.local_connection_attempts = 0

            print("WebSocket connection established")

        def websocket_thread():
            while self.running:
                try:
                    if self.ws is None:
                        self.ws = WebSocketApp(
                            self.ws_url,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close,
                            on_open=on_open
                        )
                    # Run the WebSocket and reconnect if it fails
                    self.ws.run_forever(reconnect=5)

                    # If we get here, the connection was closed
                    if not self.running:
                        break

                    time.sleep(1)  # Add delay before reconnecting

                    # Thread-safe increment of connection attempts
                    with self.thread_lock:
                        self.local_connection_attempts += 1

                    # Reset the WebSocket object for reconnection
                    if self.ws:
                        try:
                            self.ws.close()
                        except:
                            pass
                    self.ws = None

                    # If too many consecutive failures, take a longer break
                    with self.thread_lock:
                        too_many_attempts = self.local_connection_attempts > 5

                    if too_many_attempts:
                        time.sleep(5)

                except Exception as e:
                    if not self.running:
                        break
                    print(f"WebSocket thread error: {e}")
                    time.sleep(1)

            print("WebSocket thread exiting")

        # Start WebSocket connection in a separate thread
        if self.websocket_thread is None or not self.websocket_thread.is_alive():
            self.websocket_thread = threading.Thread(target=websocket_thread)
            self.websocket_thread.daemon = True
            self.websocket_thread.start()

    def process_status_update(self, status_data: Dict):
        """Process a status update from the server"""
        with self.thread_lock:
            self.local_active_workers = status_data.get('active_workers', self.local_active_workers)
            self.local_total_evaluations = status_data.get('total_evaluations', self.local_total_evaluations)

            best_objectives = status_data.get('best_objectives')
            if best_objectives:
                if not self.local_best_result:
                    self.local_best_result = {"objectives": best_objectives}
                else:
                    self.local_best_result["objectives"] = best_objectives

    def process_result(self, result: Dict):
        """Process a result received from the server"""
        with self.thread_lock:
            # Add to history for plotting
            objectives = result.get('objectives', {})
            if objectives:
                # Add a timestamp if not present
                if 'timestamp' not in result:
                    result['timestamp'] = time.time()

                # Check if this is a new unique result
                unique = True
                for existing in self.local_history:
                    if existing.get('objectives') == objectives:
                        unique = False
                        break

                if unique:
                    self.local_history.append(result)
                    print(f"Added new result to history. Total: {len(self.local_history)}")

            # Update status information
            self.local_total_evaluations = result.get('total_evaluations', self.local_total_evaluations)
            self.local_active_workers = result.get('active_workers', self.local_active_workers)

            # Update best result if this is marked as best
            if result.get('is_best', False):
                self.local_best_result = {
                    "objectives": result.get('objectives', {})
                }

    def process_queued_messages(self):
        """Process any messages in the queue"""
        messages_processed = 0
        while not self.message_queue.empty() and messages_processed < 100:  # Limit processing to avoid UI freezing
            try:
                data = self.message_queue.get_nowait()
                message_type = data.get('type')

                print(f"Processing message of type: {message_type}")

                if message_type == 'result':
                    self.process_result(data.get('data', {}))
                elif message_type == 'status':
                    self.process_status_update(data.get('data', {}))

                messages_processed += 1

                # Thread-safe update of connection status
                with self.thread_lock:
                    self.local_connected = True

            except queue.Empty:
                break

        # After processing messages, sync to session state
        if messages_processed > 0:
            self.sync_to_session_state()

        return messages_processed

    def update_dashboard(self):
        """Update all dashboard components with the latest data"""
        # Sync from session state first (in case user cleared history via UI)
        self.sync_from_session_state()

        # Process any new messages from WebSocket
        messages = self.process_queued_messages()

        if messages > 0:
            print(f"Processed {messages} messages")
            # Save cache when new messages are processed
            self.save_cache()
            # Make sure session state is updated
            self.sync_to_session_state()

        # Display connection status
        if st.session_state.connected:
            st.success("Connected to optimization server")
        else:
            st.warning("Not connected to server - trying to reconnect...")
            # Try to reconnect if not connected
            if not self.ws:
                self.connect_websocket()

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Workers", st.session_state.active_workers)
        with col2:
            st.metric("Total Evaluations", st.session_state.total_evaluations)
        with col3:
            best_value = None
            if st.session_state.best_result:
                objectives = st.session_state.best_result.get('objectives', {})
                if isinstance(objectives, dict):
                    best_value = objectives.get('objective1')

            if best_value is not None:
                st.metric("Best Objective Value", f"{best_value:.4f}")

        # Plot optimization progress even if not connected (to show historic data)
        if st.session_state.history:
            try:
                data = []
                print(f"Building plot from {len(st.session_state.history)} history items")
                for i, result in enumerate(st.session_state.history, 1):
                    if isinstance(result, dict):
                        objectives = result.get('objectives', {})
                        if isinstance(objectives, dict):
                            value = objectives.get('objective1')
                            if value is not None:
                                data.append({
                                    'trial': i,
                                    'objective_value': value
                                })

                if data:
                    print(f"Created plot with {len(data)} data points")
                    df = pd.DataFrame(data)
                    fig = px.line(df, x='trial', y='objective_value',
                                title='Optimization Progress',
                                labels={'trial': 'Trial Number',
                                       'objective_value': 'Objective Value'})
                    st.plotly_chart(fig)

                    # Display history table
                    st.subheader("Optimization History")

                    # Convert to a more viewable format for the table
                    table_data = []
                    for i, result in enumerate(st.session_state.history[:20], 1):  # Show only most recent 20
                        if isinstance(result, dict):
                            objectives = result.get('objectives', {})
                            if isinstance(objectives, dict):
                                table_data.append({
                                    'Trial': i,
                                    'Objective': objectives.get('objective1'),
                                    'Is Best': result.get('is_best', False)
                                })

                    st.dataframe(pd.DataFrame(table_data))
                    st.text(f"Showing {len(table_data)} of {len(st.session_state.history)} total trials")
                else:
                    st.info("No optimization data available yet")
            except Exception as e:
                st.error(f"Error creating visualization: {e}")
                import traceback
                st.text(traceback.format_exc())
        else:
            st.info("No optimization data available yet")

    def run(self):
        st.title("Hyperparameter Optimization Monitor")

        # Sidebar controls
        st.sidebar.title("Controls")

        if st.sidebar.button("Clear History"):
            st.session_state.history = []
            # Clear cache too
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                    print("Cache file removed")
                except Exception as e:
                    print(f"Error removing cache file: {e}")
            # Sync with local data
            self.sync_from_session_state()

        if st.sidebar.button("Reconnect"):
            self.connect_websocket()

        if st.sidebar.button("Save Cache"):
            self.save_cache()
            st.sidebar.success(f"Saved {len(st.session_state.history)} items to cache")

        # Display debug info
        with st.sidebar.expander("Debug Info"):
            st.write(f"Connected: {st.session_state.connected}")
            st.write(f"History items: {len(st.session_state.history)}")
            st.write(f"Total evaluations: {st.session_state.total_evaluations}")
            st.write(f"Connection attempts: {st.session_state.connection_attempts}")
            st.write(f"Cache file exists: {os.path.exists(self.cache_file)}")

        # Main content area
        placeholder = st.empty()

        try:
            # Update dashboard periodically (much less frequently than polling)
            update_count = 0
            while self.running:
                with placeholder.container():
                    self.update_dashboard()

                # Save cache periodically
                update_count += 1
                if update_count % 10 == 0:
                    self.save_cache()

                time.sleep(0.5)  # Just to refresh UI, not for polling data

                # Use a try/except around rerun to handle any errors
                try:
                    st.rerun()
                except Exception as e:
                    print(f"Error on rerun: {e}")
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, cleaning up...")
            self.cleanup()

        except Exception as e:
            print(f"Error in run loop: {e}")
            self.cleanup()

def main():
    st.set_page_config(
        page_title="Optimization Monitor",
        page_icon="📈",
        layout="wide",
    )

    try:
        monitor = OptimizationMonitor()
        monitor.run()
    except KeyboardInterrupt:
        print("Main function: Keyboard interrupt received")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()