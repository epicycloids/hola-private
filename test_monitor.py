import marimo

__generated_with = "0.11.26"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    from hola.core.leaderboard import Leaderboard
    return (Leaderboard,)


@app.cell
def _():
    from hola.core.objectives import ObjectiveScorer, ObjectiveName
    return ObjectiveName, ObjectiveScorer


@app.cell
def _():
    from hola.core.parameters import ParameterName
    return (ParameterName,)


@app.cell
def _():
    from hola.core.leaderboard import Trial
    return (Trial,)


@app.cell
def _():
    from pathlib import Path
    return (Path,)


@app.cell
def _():
    from typing import Any
    return (Any,)


@app.cell
def _():
    import msgspec
    return (msgspec,)


@app.cell
def _(mo):
    mo.md(r"""# HOLA Dashboard""")
    return


@app.cell
def _(mo):
    filepath = mo.ui.text(
        value="optimization_results",
        placeholder="e.g., optimization_results",
        label="Results Directory: ",
        full_width=True
    )
    run_button = mo.ui.run_button(label="Fetch available results")
    mo.vstack([filepath, run_button])
    return filepath, run_button


@app.cell
def _(Path):
    def get_results(fp):
        try:
            path = Path(fp)
            folders = [item.name for item in path.iterdir() if item.is_dir()]
            return folders
        except:
            return []
    return (get_results,)


@app.cell
def _(filepath, get_results, mo, run_button):
    run_button.value
    result = mo.ui.dropdown(get_results(filepath.value), label="Select Result: ")
    return (result,)


@app.cell
def _(filepath, mo, run_button):
    def determine_available_results(fp):
        msg = mo.callout(value=mo.md("Enter the results directory and click 'Fetch available results'."), kind="warn")
        if run_button.value:
            try:
                return None
            except FileNotFoundError as _:
                return mo.callout(value=mo.md(f"Invalid directory: {fp}"), kind="danger")
        return msg

    msg = determine_available_results(filepath.value)
    return determine_available_results, msg


@app.cell
def _(msg, result):
    def display_available_results_dropdown(msg, result):
        if msg is not None:
            return msg
        elif result is not None:
            return result
        return None

    display_available_results_dropdown(msg, result)
    return (display_available_results_dropdown,)


@app.cell
def _(mo):
    refresh = mo.ui.refresh(label="Refresh Interval", options=["100ms", "1s", "5s"], default_interval="5s")
    return (refresh,)


@app.cell
def _(refresh, result):
    def display_refresh_option(result):
        if result is None:
            return None
        if result.value:
            return refresh
        return None

    display_refresh_option(result)
    return (display_refresh_option,)


@app.cell
def _(filepath, result):
    def component_filepaths(filepath, result):
        if result.value is None:
            return None, None
        scorer_fp = filepath.value + "/" + result.value + "/coordinator_state.json_components/objectives.json"
        leader_fp = filepath.value + "/" + result.value + "/coordinator_state.json_components/leaderboard.json"
        return scorer_fp, leader_fp

    scorer_fp, leader_fp = component_filepaths(filepath, result)
    return component_filepaths, leader_fp, scorer_fp


@app.cell
def _(Leaderboard, ObjectiveScorer, leader_fp, scorer_fp):
    def initialize_leaderboard(scorer_fp, leader_fp):
        if scorer_fp is None:
            return None
        scorer = ObjectiveScorer.load_from_file(scorer_fp)
        return Leaderboard.load_from_file(leader_fp, scorer)

    leaderboard = initialize_leaderboard(scorer_fp, leader_fp)
    return initialize_leaderboard, leaderboard


@app.cell
def _(Any, ObjectiveName, ParameterName, Trial, msgspec):
    def identify_missing_trials(leaderboard, leader_fp):
        missing_trials = []

        if leader_fp is None:
            return missing_trials

        with open(leader_fp, 'rb') as f:
            data = msgspec.json.decode(f.read(), type=dict[str, Any])

        for trial_data in data.get("trials", []):
            # Make sure all objective values are floats (no None values)
            if not "trial_id" in trial_data:
                continue

            if trial_data["trial_id"] in leaderboard._data:
                continue

            if "objectives" in trial_data:
                objectives_dict = {}
                for obj_name, obj_value in trial_data["objectives"].items():
                    # Convert None to infinity to ensure it's a float
                    if obj_value is None:
                        obj_value = float('inf')
                    objectives_dict[ObjectiveName(obj_name)] = float(obj_value)

                # Create the trial
                trial = Trial(
                    trial_id=trial_data["trial_id"],
                    objectives=objectives_dict,
                    parameters={ParameterName(k): v for k, v in trial_data.get("parameters", {}).items()},
                    is_feasible=trial_data.get("is_feasible", True),
                    metadata=trial_data.get("metadata", {})
                )

                # Add the trial to the leaderboard
                missing_trials.append(trial)

        return missing_trials
    return (identify_missing_trials,)


@app.cell
def _(identify_missing_trials, leader_fp, leaderboard, refresh):
    aux = refresh.value
    missing_trials = identify_missing_trials(leaderboard, leader_fp)
    return aux, missing_trials


@app.cell
def _(mo):
    tabs = mo.ui.tabs(
        {
            "Tables": "",
            "Plots": "",
        }
    )
    return (tabs,)


@app.cell
def _(result, tabs):
    def show_tabs(tabs, result):
        if result.value is None:
            return None
        return tabs

    show_tabs(tabs, result)
    return (show_tabs,)


@app.cell
def _(mo):
    ranked = mo.ui.switch(label="Filter and sort trials")
    return (ranked,)


@app.cell
def _(leaderboard, mo):
    available_params = leaderboard.get_parameter_names() if leaderboard is not None else []
    available_objs = leaderboard.get_objective_names() if leaderboard is not None else []
    available_cgs = leaderboard.get_comparison_group_ids() if leaderboard is not None else []

    p1val, p2val = None, None
    if len(available_params) > 0:
        p1val = available_params[0]
        p2val = available_params[-1]
    params1 = mo.ui.dropdown(available_params, value=p1val, label="Select first parameter for comparison: ")
    params2 = mo.ui.dropdown(available_params, value=p2val, label="Select second parameter for comparison: ")

    o1val, o2val, o3val = None, None, None
    if len(available_objs) > 0:
        o1val = available_objs[0]
        o2val = available_objs[-1]
        o3val = available_objs[0]
    objs1 = mo.ui.dropdown(available_objs, value=o1val, label="Select first objective for comparison: ")
    objs2 = mo.ui.dropdown(available_objs, value=o2val, label="Select second objective for comparison: ")
    objs3 = mo.ui.dropdown(available_objs, value=o3val, label="Select objective for history: ")

    c1val, c2val, c3val = None, None, None
    if len(available_cgs) > 1:
        c1val = available_cgs[0]
        c2val = available_cgs[-1]
        c3val = None
    cgs1 = mo.ui.dropdown(available_cgs, value=c1val, label="Select first comparison group for comparison: ")
    cgs2 = mo.ui.dropdown(available_cgs, value=c2val, label="Select second comparison group for comparison: ")
    cgs3 = mo.ui.dropdown(available_cgs, value=c3val, label="Select third comparison group for comparison: ")
    return (
        available_cgs,
        available_objs,
        available_params,
        c1val,
        c2val,
        c3val,
        cgs1,
        cgs2,
        cgs3,
        o1val,
        o2val,
        o3val,
        objs1,
        objs2,
        objs3,
        p1val,
        p2val,
        params1,
        params2,
    )


@app.cell
def _(
    cgs1,
    cgs2,
    cgs3,
    leaderboard,
    objs1,
    objs2,
    objs3,
    params1,
    params2,
    ranked,
    show_plots,
    show_tables,
):
    def show_thing(tabs):
        if tabs.value == "Tables":
            return show_tables(leaderboard, ranked)
        elif tabs.value == "Plots":
            return show_plots(leaderboard, params1, params2, objs1, objs2, objs3, cgs1, cgs2, cgs3)
    return (show_thing,)


@app.cell
def _(leaderboard, missing_trials, show_thing, tabs):
    for _missing_trial in missing_trials:
        leaderboard.add(_missing_trial)

    show_thing(tabs)
    return


@app.cell
def _(mo):
    def show_tables(leaderboard, ranked):
        if leaderboard is None:
            return None

        data_header = mo.md("## Trial Data")
        table = mo.ui.table(leaderboard.get_dataframe(ranked_only=ranked.value), show_column_summaries=False, selection=None)

        meta_header = mo.md("## Trial Metadata")
        meta_table = mo.ui.table(leaderboard.get_metadata(), show_column_summaries=False, selection=None)

        return mo.vstack([data_header, ranked, table, meta_header, meta_table])
    return (show_tables,)


@app.cell
def _(available_cgs, available_objs, mo):
    def show_plots(leaderboard, p1, p2, objs1, objs2, objs3, cgs1, cgs2, cgs3):
        if leaderboard is None:
            return None
        
        param_fig = None
        if p1.value is not None:
            param_fig = leaderboard.plot_parameters(p1.value, p2.value)
        param_stack = mo.vstack([p1, p2, param_fig])

        objs_compare_fig = None
        objs_history_fig = None
        if objs1.value is not None and objs2.value is not None:
            objs_compare_fig = leaderboard.plot_objectives(objs1.value, objs2.value)
        if objs3.value is not None:
            objs_history_fig = leaderboard.plot_objective_vs_trial(objs3.value)

        if len(available_objs) > 1:
            objs_stack = mo.vstack([objs1, objs2, objs_compare_fig, objs3, objs_history_fig])
        else:
            objs_stack = mo.vstack([objs3, objs_history_fig])

        cgs_compare_fig = None
        if cgs1.value is not None and cgs2.value is not None:
            if cgs3.value is not None:
                cgs_compare_fig = leaderboard.plot_comparison_groups_3d(cgs1.value, cgs2.value, cgs3.value)
            else:
                cgs_compare_fig = leaderboard.plot_comparison_groups(cgs1.value, cgs2.value)
        if len(available_cgs) > 2:
            cgs_stack = mo.vstack([cgs1, cgs2, cgs3, cgs_compare_fig])
        elif len(available_cgs) > 1:
            cgs_stack = mo.vstack([cgs1, cgs2, cgs_compare_fig])
        else:
            cgs_stack = mo.vstack([])

        return mo.vstack([param_stack, objs_stack, cgs_stack])
    return (show_plots,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
