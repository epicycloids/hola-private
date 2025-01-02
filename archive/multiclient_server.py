from __future__ import annotations

import json
import os
import shutil
import uuid as uuid_gen
from pathlib import Path

import falcon
import pandas as pd

from hola.algorithm import HOLA
from hola.constants import OBJECTIVE_PREFIX, PARAM_PREFIX


def get_hola(session_path: str | Path) -> HOLA | None:
    """Reconstruct HOLA instance from session files.

    :param session_path: Path to session directory
    :type session_path: str | Path
    :return: Reconstructed HOLA instance, or None if invalid/missing files
    :rtype: HOLA | None
    :raises ValueError: If configuration files exist but are invalid
    """
    session_path = Path(session_path)
    if not session_path.is_dir():
        return None

    required_files = {
        "hola_params.json": "parameter configuration",
        "hola_objectives.json": "objective configuration",
        "hola_setup.json": "HOLA setup",
    }

    # Verify all required files exist
    missing_files = []
    for filename, description in required_files.items():
        if not (session_path / filename).is_file():
            missing_files.append(f"{description} ({filename})")

    if missing_files:
        raise ValueError(f"Missing required files for session: {', '.join(missing_files)}")

    try:
        with open(session_path / "hola_params.json", "r", encoding="utf-8") as f:
            params_config = json.load(f)

        with open(session_path / "hola_objectives.json", "r", encoding="utf-8") as f:
            obj_config = json.load(f)

        with open(session_path / "hola_setup.json", "r", encoding="utf-8") as f:
            setup_dict = json.load(f)

        hola = HOLA(
            params_config=params_config,
            objectives_config=obj_config,
            top_fraction=setup_dict["top_fraction"],
            min_samples=setup_dict["min_samples"],
            min_fit_samples=setup_dict["min_fit_samples"],
            n_components=setup_dict["n_components"],
            gmm_reg=setup_dict["gmm_reg"],
            gmm_sampler=setup_dict["gmm_sampler"],
            explore_sampler=setup_dict["explore_sampler"],
        )

        # Load results if they exist
        results_file = session_path / "hola_results.csv"
        if results_file.is_file():
            try:
                hola.load(results_file)
            except (ValueError, pd.errors.EmptyDataError) as e:
                raise ValueError(f"Invalid results file: {str(e)}") from e

        return hola

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration files: {str(e)}") from e
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {str(e)}") from e


class BaseHolaResource:
    """Base resource class for HOLA server endpoints.

    Provides common functionality for session validation.
    """

    def __init__(self, cpath: str | Path) -> None:
        """Initialize base resource.

        :param cpath: Path to directory containing optimization sessions
        :type cpath: str | Path
        """
        self.cpath = cpath

    def validate_session(
        self, resp: falcon.Response, uuid: str | None, endpoint_name: str
    ) -> tuple[str, bool]:
        """Validate session ID and return session path if valid.

        :param resp: HTTP response object to update on error
        :param uuid: Session identifier to validate
        :param endpoint_name: Name of endpoint for error message
        :return: Tuple of (session path, is_valid)
        """
        if uuid is None or uuid == "":
            resp.text = f"No session id provided. Please call this resource as /{endpoint_name}/{{session_id}}"
            resp.status = falcon.HTTP_400
            return "", False

        session_path = os.path.join(self.cpath, uuid)
        if not os.path.isdir(session_path):
            resp.text = f"There is no project with session_id {uuid}"
            resp.status = falcon.HTTP_400
            return session_path, False

        return session_path, True


class HomeResource(BaseHolaResource):
    """Homepage showing current optimization status and results.

    Displays the full leaderboard and up to 3 Pareto fronts, depending on
    whether there are multiple fronts of solutions.
    """

    def on_get(self, req: falcon.Request, resp: falcon.Response, uuid: str | None = None) -> None:
        """Handle GET request for homepage.

        :param req: HTTP request object
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier
        :type uuid: str | None

        **Response:**
            - 200: HTML page showing leaderboard and Pareto fronts
            - 400: If session ID is missing or invalid
        """
        session_path, valid = self.validate_session(resp, uuid, "home")
        if not valid:
            return

        try:
            with open(os.path.join(session_path, "hola_results.csv")) as f:
                df = pd.read_csv(f)
        except FileNotFoundError:
            resp.text = "No results available yet"
            resp.status = falcon.HTTP_400
            return

        # Build HTML response
        html = "<html><body>"

        # Show leaderboard
        html += "<h2>Full Leaderboard</h2>"
        html += df.to_html(classes="table table-striped", border=0)

        # Show Pareto front information
        fronts = (
            df.groupby("front")
            .apply(lambda x: x.sort_values("crowding_distance", ascending=False))
            .reset_index(drop=True)
        )

        if len(fronts.front.unique()) > 1:
            for front_idx in range(min(3, fronts.front.max() + 1)):
                front_df = fronts[fronts.front == front_idx]
                if not front_df.empty:
                    html += f"<h3>Pareto Front {front_idx}</h3>"
                    html += front_df.to_html(classes="table table-striped", border=0)

        html += "</body></html>"

        resp.text = html
        resp.content_type = falcon.MEDIA_HTML
        resp.status = falcon.HTTP_200


class ReportRequestResource(BaseHolaResource):
    """Resource for requesting parameter samples and reporting results.

    Handles both requesting new parameter samples and reporting back the
    objective values obtained from evaluating those parameters.
    """

    def on_get(
        self,
        req: falcon.Request,
        resp: falcon.Response,
        uuid: str | None = None,
        num_samples: int | None = None,
    ) -> None:
        """Handle GET request for parameter samples.

        :param req: HTTP request object
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier
        :type uuid: str | None
        :param num_samples: Number of samples to generate, defaults to 1
        :type num_samples: int | None

        **Response:**
            - 200: JSON containing parameter samples
            - 400: If session ID is missing or invalid
        """
        session_path, valid = self.validate_session(resp, uuid, "report_request")
        if not valid:
            return

        hola = get_hola(session_path)
        if hola is None:
            resp.text = f"Invalid session configuration."
            resp.status = falcon.HTTP_400
            return

        if num_samples is None:
            sample = hola.sample()
            resp.media = sample
        else:
            samples = []
            for _ in range(num_samples):
                samples.append(hola.sample())
                # Decrement count to avoid premature switch to exploitation
                hola.mixture_sampler.sample_count -= 1

            # Reset the actual count
            hola.mixture_sampler.sample_count += num_samples
            resp.media = samples

        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_200

    def on_post(
        self,
        req: falcon.Request,
        resp: falcon.Response,
        uuid: str | None = None,
        num_samples: int | None = None,
    ) -> None:
        """Handle POST request reporting objective values and requesting new samples.

        :param req: HTTP request object containing results data
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier
        :type uuid: str | None
        :param num_samples: Number of new samples to generate, defaults to 1
        :type num_samples: int | None

        **Expected Request Format:**
            Single result::

                {
                    "params": {"param1": value1, ...},
                    "objectives": {"obj1": value1, ...}
                }

            Multiple results::

                [
                    {
                        "params": {"param1": value1, ...},
                        "objectives": {"obj1": value1, ...}
                    },
                    ...
                ]

        **Response:**
            - 200: JSON containing new parameter samples
            - 400: If session ID is missing or invalid, or if data format is
                incorrect
        """
        session_path, valid = self.validate_session(resp, uuid, "report_request")
        if not valid:
            return

        hola = get_hola(session_path)
        if hola is None:
            resp.text = f"Invalid session configuration."
            resp.status = falcon.HTTP_400
            return

        try:
            data = json.load(req.bounded_stream)
            if isinstance(data, str):
                data = json.loads(data)
        except json.JSONDecodeError:
            resp.text = "Invalid JSON in request body"
            resp.status = falcon.HTTP_400
            return

        # Add results
        try:
            if isinstance(data, list):
                for result in data:
                    self._add_result(hola, result)
            elif isinstance(data, dict):
                self._add_result(hola, data)
            else:
                raise ValueError("Invalid data format")
        except (KeyError, ValueError) as e:
            resp.text = f"Invalid result data: {str(e)}"
            resp.status = falcon.HTTP_400
            return

        # Save updated results
        hola.save(os.path.join(session_path, "hola_results.csv"))

        # Generate requested number of new samples
        if num_samples is None:
            resp.media = hola.sample()
        else:
            resp.media = [hola.sample() for _ in range(num_samples)]

        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_200

    def _add_result(self, hola: HOLA, result: dict) -> None:
        """Add a single result to HOLA instance.

        :param hola: HOLA instance to update
        :param result: Result dictionary containing params and objectives
        :raises KeyError: If required fields are missing
        :raises ValueError: If data format is invalid
        """
        params = result.get("params")
        objectives = result.get("objectives")
        if params is None or objectives is None:
            raise KeyError("Result must contain 'params' and 'objectives'")
        hola.add(objectives, params)


class ExperimentResource(BaseHolaResource):
    """Resource for managing experiment configurations.

    Handles retrieving and updating objective configurations for optimization
    experiments. Parameter configurations are read-only after initialization.
    """

    def on_get(self, req: falcon.Request, resp: falcon.Response, uuid: str | None = None) -> None:
        """Handle GET request for experiment configuration.

        :param req: HTTP request object
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier
        :type uuid: str | None

        **Response:**
            - 200: JSON containing parameter and objective configurations
            - 400: If session ID is missing or invalid
        """
        session_path, valid = self.validate_session(resp, uuid, "experiment")
        if not valid:
            return

        try:
            with open(os.path.join(session_path, "hola_params.json"), "r", encoding="utf-8") as f:
                params_config = json.load(f)
            with open(
                os.path.join(session_path, "hola_objectives.json"), "r", encoding="utf-8"
            ) as f:
                obj_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            resp.text = f"Error reading configuration files: {str(e)}"
            resp.status = falcon.HTTP_400
            return

        resp.media = {"params_config": params_config, "objectives_config": obj_config}
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_200

    def on_post(self, req: falcon.Request, resp: falcon.Response, uuid: str | None = None) -> None:
        """Handle POST request to update objective configuration.

        :param req: HTTP request object containing new objective configuration
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier
        :type uuid: str | None

        **Expected Request Format:**
            .. code-block:: json

                {
                    "objectives_config": {
                        "objective1": {
                            "target": value,
                            "limit": value,
                            "comparison_group": int,
                            "priority": value
                        },
                        ...
                    }
                }

        **Response:**
            - 200: If configuration was updated successfully
            - 400: If session ID is missing/invalid or if configuration is invalid
        """
        session_path, valid = self.validate_session(resp, uuid, "experiment")
        if not valid:
            return

        hola = get_hola(session_path)
        if hola is None:
            resp.text = "Invalid session configuration"
            resp.status = falcon.HTTP_400
            return

        try:
            data = json.load(req.bounded_stream)
            new_obj_config = data["objectives_config"]
        except (json.JSONDecodeError, KeyError) as e:
            resp.text = f"Invalid request data: {str(e)}"
            resp.status = falcon.HTTP_400
            return

        try:
            hola.set_objectives_config(new_obj_config)
        except ValueError as e:
            resp.text = f"Invalid objective configuration: {str(e)}"
            resp.status = falcon.HTTP_400
            return

        # Save updated configuration
        try:
            with open(
                os.path.join(session_path, "hola_objectives.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(new_obj_config, f, indent=4)
            hola.save(os.path.join(session_path, "hola_results.csv"))
        except IOError as e:
            resp.text = f"Error saving configuration: {str(e)}"
            resp.status = falcon.HTTP_500
            return

        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_JSON


class ParamResource(BaseHolaResource):
    """Resource for retrieving optimal parameter configurations.

    Provides access to best parameters found during optimization. For
    multi-group problems, returns all parameters on the first Pareto front.
    """

    def on_get(self, req: falcon.Request, resp: falcon.Response, uuid: str | None = None) -> None:
        """Handle GET request for optimal parameters.

        :param req: HTTP request object
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier
        :type uuid: str | None

        **Response:**
            - 200: JSON containing optimal parameters
            - 400: If session ID is missing/invalid or if no results exist
        """
        session_path, valid = self.validate_session(resp, uuid, "param")
        if not valid:
            return

        results_path = os.path.join(session_path, "hola_results.csv")
        if not os.path.isfile(results_path):
            resp.text = "No results available yet"
            resp.status = falcon.HTTP_400
            return

        hola = get_hola(session_path)
        if hola is None:
            resp.text = "Invalid session configuration"
            resp.status = falcon.HTTP_400
            return

        results = hola.get_result()
        if not results.pareto_fronts:
            resp.text = "No valid results found"
            resp.status = falcon.HTTP_400
            return

        first_front = results.pareto_fronts[0]
        if len(first_front) == 1:
            resp.media = {
                "best_params": results.best_params,
                "best_objectives": results.best_objectives,
            }
        else:
            pareto_optimal = []
            for _, row in first_front.iterrows():
                params = {
                    k[len(PARAM_PREFIX) :]: v for k, v in row.items() if k.startswith(PARAM_PREFIX)
                }
                objectives = {
                    k[len(OBJECTIVE_PREFIX) :]: v
                    for k, v in row.items()
                    if k.startswith(OBJECTIVE_PREFIX)
                }
                pareto_optimal.append({"parameters": params, "objectives": objectives})
            resp.media = {"pareto_optimal": pareto_optimal}

        resp.status = falcon.HTTP_200
        resp.content_type = falcon.MEDIA_JSON


class CreateResource(BaseHolaResource):
    """Resource for creating new optimization sessions."""

    def on_post(self, req: falcon.Request, resp: falcon.Response) -> None:
        """Handle POST request to create new session.

        :param req: HTTP request containing configuration data
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response

        **Expected Request Format:**
            .. code-block:: json

                {
                    "params_file": {
                        "param1": {"type": "float", ...},
                        ...
                    },
                    "objectives_file": {
                        "obj1": {"target": value, ...},
                        ...
                    },
                    "hola_setup": {  # Optional
                        "top_fraction": float,
                        "min_samples": int,
                        ...
                    }
                }

        **Response:**
            - 201: JSON containing new session UUID if created successfully
            - 400: If configuration data is invalid
        """
        try:
            data = json.load(req.bounded_stream)
        except json.JSONDecodeError:
            resp.text = "Invalid JSON in request body"
            resp.status = falcon.HTTP_400
            return

        if not data.get("params_file") or not data.get("objectives_file"):
            resp.text = (
                "Must provide parameter configuration as params_file and "
                "objectives configuration as objectives_file"
            )
            resp.status = falcon.HTTP_400
            return

        # Generate unique session ID
        session_id = uuid_gen.uuid4()
        session_path = os.path.join(self.cpath, str(session_id))

        # Ensure no UUID collision
        while os.path.isdir(session_path):
            session_id = uuid_gen.uuid4()
            session_path = os.path.join(self.cpath, str(session_id))

        try:
            os.makedirs(session_path)

            # Save parameter configuration
            with open(os.path.join(session_path, "hola_params.json"), "w", encoding="utf-8") as f:
                json.dump(data["params_file"], f, indent=4)

            # Save objective configuration
            with open(
                os.path.join(session_path, "hola_objectives.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(data["objectives_file"], f, indent=4)

            # Save HOLA setup with defaults
            setup_dict = {
                "top_fraction": 0.2,
                "min_samples": None,
                "min_fit_samples": None,
                "n_components": 3,
                "gmm_reg": 0.0005,
                "gmm_sampler": "uniform",
                "explore_sampler": "sobol",
            }

            if "hola_setup" in data:
                for key, value in data["hola_setup"].items():
                    if key not in setup_dict:
                        resp.text = f"{key} is not a valid setup argument"
                        resp.status = falcon.HTTP_400
                        return
                    setup_dict[key] = value

            with open(os.path.join(session_path, "hola_setup.json"), "w", encoding="utf-8") as f:
                json.dump(setup_dict, f, indent=4)

        except IOError as e:
            # Clean up if any file operations fail
            if os.path.exists(session_path):
                shutil.rmtree(session_path)
            resp.text = f"Error creating session: {str(e)}"
            resp.status = falcon.HTTP_500
            return

        resp.media = {"uuid": str(session_id)}
        resp.content_type = falcon.MEDIA_JSON
        resp.status = falcon.HTTP_201


class DeleteResource(BaseHolaResource):
    """Resource for deleting optimization sessions."""

    def on_get(self, req: falcon.Request, resp: falcon.Response, uuid: str | None = None) -> None:
        """Handle GET request to delete session.

        :param req: HTTP request object
        :type req: falcon.Request
        :param resp: HTTP response object
        :type resp: falcon.Response
        :param uuid: Session identifier to delete
        :type uuid: str | None

        **Response:**
            - 200: If session was deleted successfully
            - 400: If session ID is missing or invalid
        """
        session_path, valid = self.validate_session(resp, uuid, "delete_session")
        if not valid:
            return

        try:
            shutil.rmtree(session_path)
        except IOError as e:
            resp.text = f"Error deleting session: {str(e)}"
            resp.status = falcon.HTTP_500
            return

        resp.text = "Session successfully deleted"
        resp.status = falcon.HTTP_200


def create_multiclient_server(dirpath: str | Path) -> falcon.App:
    """Create Falcon app for HOLA server.

    :param dirpath: Directory to store session data
    :type dirpath: str | Path
    :return: Configured Falcon application
    :rtype: falcon.App
    """
    server_dir = Path(dirpath).resolve()
    server_dir.mkdir(parents=True, exist_ok=True)

    # Create Falcon app with default middleware
    app = falcon.App(
        middleware=[falcon.CORSMiddleware(allow_origins=["*"], allow_credentials=True)]
    )

    # Mount routes
    routes = [
        ("/create_session", CreateResource),
        ("/delete_session/{uuid}", DeleteResource),
        ("/{uuid}", HomeResource),
        ("/report_request/{uuid}", ReportRequestResource),
        ("/report_request/{uuid}/{num_samples:int}", ReportRequestResource),
        ("/experiment/{uuid}", ExperimentResource),
        ("/param/{uuid}", ParamResource),
    ]

    for route, resource in routes:
        app.add_route(route, resource(server_dir))

    return app
