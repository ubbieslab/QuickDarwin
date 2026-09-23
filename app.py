from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, List

import streamlit as st

from pydarwin_core import (
    ADMIN_FIRST_ORDER_ORAL,
    ADMIN_INFUSION,
    PROJECT_ROOT,
    CovariateInfo,
    MODEL_PARAMETER_ORDER,
    SearchSpace,
    UserChoices,
    available_bsv_parameters,
    available_model_parameters,
    default_algorithm_options,
    generate_py_darwin_files,
    has_occasion_column,
    infer_covariates,
    load_csv,
    load_csv_from_bytes,
    propose_search_space,
    validate_token_template_consistency,
)
from pydarwin_launcher import (
    build_run_command,
    format_windows_command,
    launch_in_windows_terminal,
    validate_run_environment,
)


# Session state keys for the configuration wizard
WIZARD_KEYS = (
    "wizard_compartments",
    "wizard_residual_error_models",
    "wizard_bsv_parameters",
    "wizard_bov_parameters",
    "wizard_selected_algos",
    "wizard_administration",
    "wizard_test_tlag",
    "wizard_use_effect_limit",
    "wizard_effect_limit",
    "wizard_working_dir",
    "wizard_output_dir",
    "wizard_temp_dir",
    "wizard_nmfe_path",
    "pydarwin_python_executable",
)
REPORTER_KEYS = (
    "reporter_mode",
    "reporter_project_dir",
    "reporter_working_dir",
    "reporter_output_dir",
    "reporter_key_models_dir",
    "reporter_non_dominated_models_dir",
    "reporter_r_path",
    "reporter_rscript_path",
    "last_generated_files",
    "last_generated_project_dir",
)

DEFAULT_WORKING_DIR = "{project_dir}/working"
DEFAULT_OUTPUT_DIR = "{working_dir}/output"
DEFAULT_TEMP_DIR = "{working_dir}/temp"
DEFAULT_NMFE_PATH = "c:/nm750/util/nmfe75.bat"
REPORTER_MODE_SINGLE = "single"
REPORTER_MODE_MOGA = "moga"
REPORTER_SCRIPT_PATH = PROJECT_ROOT / "reference" / "darwinreporter_shinyapp.R"


def _default_reporter_executable_paths(
    platform_name: str | None = None,
) -> tuple[str, str]:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        r_bin = r"C:\Program Files\R\R-4.4.1\bin\x64"
        return (rf"{r_bin}\R.exe", rf"{r_bin}\Rscript.exe")
    if platform_name == "darwin":
        return ("/usr/local/bin/R", "/usr/local/bin/Rscript")
    return (shutil.which("R") or "", shutil.which("Rscript") or "")


def _cov_state_key(param: str) -> str:
    return f"wizard_covs_{param.lower()}"


def _ensure_reporter_defaults(working_dir: str, output_dir: str) -> None:
    default_r_path, default_rscript_path = _default_reporter_executable_paths()
    st.session_state.setdefault("reporter_mode", REPORTER_MODE_SINGLE)
    st.session_state.setdefault("reporter_project_dir", "")
    st.session_state.setdefault("reporter_r_path", default_r_path)
    st.session_state.setdefault("reporter_rscript_path", default_rscript_path)
    if not st.session_state.get("reporter_working_dir"):
        st.session_state["reporter_working_dir"] = working_dir
    if not st.session_state.get("reporter_output_dir"):
        st.session_state["reporter_output_dir"] = output_dir
    if not st.session_state.get("reporter_key_models_dir") and working_dir:
        st.session_state["reporter_key_models_dir"] = str(Path(working_dir).expanduser() / "key_models")
    if not st.session_state.get("reporter_non_dominated_models_dir") and working_dir:
        st.session_state["reporter_non_dominated_models_dir"] = str(
            Path(working_dir).expanduser() / "non_dominated_models"
        )


def _set_reporter_defaults_from_generated(project_dir: Path, user_choices: UserChoices) -> None:
    project_dir_str = str(project_dir.resolve())
    working_dir = (user_choices.working_dir or DEFAULT_WORKING_DIR).strip()
    working_dir = working_dir.replace("{project_dir}", project_dir_str)
    output_dir = (user_choices.output_dir or DEFAULT_OUTPUT_DIR).strip()
    output_dir = output_dir.replace("{project_dir}", project_dir_str)
    output_dir = output_dir.replace("{working_dir}", working_dir)
    mode = REPORTER_MODE_MOGA if "MOGA" in user_choices.algorithms else REPORTER_MODE_SINGLE
    st.session_state["last_generated_project_dir"] = project_dir_str
    st.session_state["reporter_mode"] = mode
    st.session_state["reporter_project_dir"] = project_dir_str
    st.session_state["reporter_working_dir"] = working_dir
    st.session_state["reporter_output_dir"] = output_dir
    st.session_state["reporter_key_models_dir"] = str(Path(working_dir).expanduser() / "key_models")
    st.session_state["reporter_non_dominated_models_dir"] = str(
        Path(working_dir).expanduser() / "non_dominated_models"
    )


def _resolve_executable_path(user_value: str, default_name: str) -> str | None:
    cleaned = user_value.strip().strip('"').strip("'")
    if cleaned:
        candidate = Path(cleaned).expanduser()
        if candidate.is_file():
            return str(candidate)
        return None
    return shutil.which(default_name)


def _validate_reporter_inputs(
    mode: str,
    project_dir: str,
    working_dir: str,
    output_dir: str,
    key_models_dir: str,
    non_dominated_models_dir: str,
) -> list[str]:
    required_paths = [
        ("project_dir", project_dir),
        ("working_dir", working_dir),
        ("output_dir", output_dir),
    ]
    if mode == REPORTER_MODE_MOGA:
        required_paths.append(
            ("non_dominated_models_dir", non_dominated_models_dir)
        )
    else:
        required_paths.append(("key_models_dir", key_models_dir))

    errors: list[str] = []
    for label, value in required_paths:
        cleaned = value.strip()
        if not cleaned:
            errors.append(f"Please provide `{label}`.")
            continue
        path = Path(cleaned).expanduser()
        if not path.exists():
            errors.append(f"`{label}` does not exist: `{path}`")
        elif not path.is_dir():
            errors.append(f"`{label}` must be a directory: `{path}`")
    return errors


def _check_reporter_requirements(r_path_value: str, rscript_path_value: str) -> tuple[str | None, str | None, str | None]:
    if not REPORTER_SCRIPT_PATH.is_file():
        return (f"Missing DarwinReporter launcher script: `{REPORTER_SCRIPT_PATH}`", None, None)

    resolved_r = _resolve_executable_path(r_path_value, "R")
    if resolved_r is None:
        return (
            "R is not available on PATH. Provide the full path to `R` or install/add it to PATH.",
            None,
            None,
        )

    resolved_rscript = _resolve_executable_path(rscript_path_value, "Rscript")
    if resolved_rscript is None:
        return (
            "Rscript is not available on PATH. Provide the full path to `Rscript` or install/add it to PATH.",
            None,
            None,
        )

    check = subprocess.run(
        [resolved_rscript, "-e", "cat(requireNamespace('Certara.DarwinReporter', quietly = TRUE))"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check.returncode != 0:
        detail = check.stderr.strip() or check.stdout.strip() or "unknown error"
        return (f"Could not verify the R package `Certara.DarwinReporter`: {detail}", None, None)
    if check.stdout.strip().lower() != "true":
        return (
            "R package `Certara.DarwinReporter` is not installed. "
            "Install it in R, then try launching the reporter again.",
            None,
            None,
        )
    return (None, resolved_r, resolved_rscript)


def _build_reporter_command(
    r_path: str,
    mode: str,
    project_dir: str,
    working_dir: str,
    output_dir: str,
    key_models_dir: str,
    non_dominated_models_dir: str,
) -> list[str]:
    command = [
        r_path,
        "--quiet",
        "--no-save",
        "-f",
        str(REPORTER_SCRIPT_PATH),
        "--args",
        "--mode",
        mode,
        "--project-dir",
        project_dir,
        "--working-dir",
        working_dir,
        "--output-dir",
        output_dir,
    ]
    if mode == REPORTER_MODE_MOGA:
        command.extend(
            [
                "--non-dominated-models-dir",
                non_dominated_models_dir,
            ]
        )
    else:
        command.extend(["--key-models-dir", key_models_dir])
    return command


def _launch_reporter_in_terminal(
    r_path: str,
    mode: str,
    project_dir: str,
    working_dir: str,
    output_dir: str,
    key_models_dir: str,
    non_dominated_models_dir: str,
) -> None:
    command = _build_reporter_command(
        r_path,
        mode,
        project_dir,
        working_dir,
        output_dir,
        key_models_dir,
        non_dominated_models_dir,
    )

    if sys.platform == "darwin":
        command_str = " ".join(shlex.quote(part) for part in command)
        escaped_command = command_str.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "Terminal" to activate',
                "-e",
                f'tell application "Terminal" to do script "{escaped_command}"',
            ]
        )
        return

    if sys.platform.startswith("win"):
        subprocess.Popen(["cmd", "/c", "start", "", "cmd", "/k", *command])
        return

    raise RuntimeError(
        "DarwinReporter launch is currently supported on macOS and Windows only."
    )


def _validate_generated_file_edit(
    file_path: Path,
    content: str,
    generated: dict[str, List[str]],
) -> List[str]:
    """Validate edited configuration content before it replaces a generated file."""
    resolved_path = file_path.resolve()
    paths_by_kind = {
        kind: [Path(path).resolve() for path in paths]
        for kind, paths in generated.items()
    }
    known_paths = {
        path for paths in paths_by_kind.values() for path in paths
    }
    if resolved_path not in known_paths:
        return ["This file is not part of the current generated configuration."]

    parsed_json: Any = None
    if file_path.suffix.lower() == ".json":
        try:
            parsed_json = json.loads(content)
        except json.JSONDecodeError as exc:
            return [
                f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ]
        if not isinstance(parsed_json, dict):
            return ["The JSON file must contain an object at its top level."]

    template_paths = paths_by_kind.get("template", [])
    tokens_paths = paths_by_kind.get("tokens", [])

    try:
        if resolved_path in template_paths:
            if not tokens_paths:
                return ["Cannot validate template.txt because tokens.json is missing."]
            tokens = json.loads(tokens_paths[0].read_text(encoding="utf-8"))
            if not isinstance(tokens, dict):
                return ["The companion tokens.json must contain a JSON object."]
            _, errors = validate_token_template_consistency(content, tokens)
            return errors

        if resolved_path in tokens_paths:
            if not template_paths:
                return ["Cannot validate tokens.json because template.txt is missing."]
            template = template_paths[0].read_text(encoding="utf-8")
            _, errors = validate_token_template_consistency(template, parsed_json)
            return errors
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Could not validate the companion file: {exc}"]

    return []


def _save_generated_file_edit(
    file_path: Path,
    content: str,
    generated: dict[str, List[str]],
) -> List[str]:
    """Validate and save an edited generated file, returning any errors."""
    errors = _validate_generated_file_edit(file_path, content, generated)
    if errors:
        return errors
    try:
        file_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return [f"Could not save the file: {exc}"]
    return []


@st.dialog("Edit generated file", width="large")
def _render_generated_file_editor(
    file_path: str,
    generated: dict[str, List[str]],
) -> None:
    path = Path(file_path)
    editor_key = f"generated_file_editor::{path}"
    st.caption(str(path))

    if editor_key not in st.session_state:
        try:
            st.session_state[editor_key] = path.read_text(encoding="utf-8")
        except OSError as exc:
            st.error(f"Could not open the file: {exc}")
            return

    edited_content = st.text_area(
        "File contents",
        height=500,
        key=editor_key,
    )
    if st.button(
        "Save changes",
        type="primary",
        key=f"save_generated_file::{path}",
    ):
        errors = _save_generated_file_edit(path, edited_content, generated)
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.success(f"Saved {path.name}.")


def main() -> None:
    st.set_page_config(
        page_title="DarwinHub AI — pyDarwin Configuration Assistant",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp p,
        .stApp label,
        .stApp li,
        .stApp input,
        .stApp textarea,
        .stApp button {
            font-size: 1.1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("DarwinHub AI — pyDarwin Configuration Assistant")
    st.markdown(
        "Build `template.txt`, `tokens.json`, and `options.json` "
        "for pyDarwin based on your dataset."
    )

    st.subheader("1. Drag and drop a NONMEM-ready `.csv` data file")
    uploaded_file = st.file_uploader(
        "Upload a NONMEM-ready CSV data file",
        type=["csv"],
        label_visibility="collapsed",
    )

    # Persist the uploaded file across Streamlit reruns
    if uploaded_file is not None:
        st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
        st.session_state["uploaded_file_name"] = uploaded_file.name or "data.csv"

    # Use the persisted file when the uploader is empty after a rerun
    if uploaded_file is None:
        if st.session_state.get("uploaded_file_bytes"):
            name = st.session_state.get("uploaded_file_name", "data.csv")
            st.info(
                f"Using previously uploaded file: **{name}**. Upload a new file above to replace, or clear below."
            )
            df = load_csv_from_bytes(st.session_state["uploaded_file_bytes"])
            if st.button("Clear uploaded file and start over"):
                for key in list(st.session_state.keys()):
                    if key in {"uploaded_file_bytes", "uploaded_file_name"}:
                        st.session_state.pop(key, None)
                    elif (
                        key in WIZARD_KEYS
                        or key in REPORTER_KEYS
                        or key.startswith("wizard_covs_")
                        or key.startswith("ui_covs_")
                    ):
                        st.session_state.pop(key, None)
                st.rerun()
        else:
            st.info("Upload a `.csv` file to begin.")
            return
    else:
        df = load_csv(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.write(df.head())

    covariates = infer_covariates(df)
    has_occ = has_occasion_column(df)

    st.subheader("Covariate types")
    usable_covariates: dict[str, CovariateInfo] = {}
    for name, info in covariates.items():
        use_this = st.checkbox(
            f"Use {name} as covariate",
            value=True,
            key=f"{name}_use",
        )
        if not use_this:
            # Skip type selection; this covariate will not be considered downstream
            continue

        cov_type = st.selectbox(
            f"{name} type",
            options=["continuous", "categorical"],
            index=0 if info.cov_type == "continuous" else 1,
            key=f"{name}_cov_type",
        )
        info.cov_type = cov_type
        if cov_type == "categorical":
            levels = list(info.levels or [])
            info.is_binary = len(levels) == 2
            counts_text = ", ".join(
                f"{level}: n={int((info.level_counts or {}).get(str(level), 0))}"
                for level in levels
            )
            st.caption(
                f"Observed levels ({'binary' if info.is_binary else 'nominal'}): "
                f"{counts_text or 'none'}"
            )
            if levels:
                reference_index = (
                    levels.index(info.reference_level)
                    if info.reference_level in levels
                    else 0
                )
                info.reference_level = st.selectbox(
                    f"{name} reference level",
                    options=levels,
                    index=reference_index,
                    key=f"{name}_reference_level",
                    help="Defaults to the largest group by unique-subject count.",
                )
            if any(not isinstance(level, (int, float)) for level in levels):
                st.error(
                    f"{name} has non-numeric levels. Recode them to numeric values "
                    "before generating NONMEM files."
                )
            if not info.within_subject_consistent:
                st.warning(f"{name} changes within subject.")
                info.allow_time_varying = st.checkbox(
                    f"Allow {name} as a time-varying categorical covariate",
                    value=False,
                    key=f"{name}_allow_time_varying",
                )
        usable_covariates[name] = info

    # Only keep covariates the user chose to use
    covariates = usable_covariates

    st.subheader("2. Proposed search space from data")
    initial_space = propose_search_space(df, covariates)

    # Initialize wizard state from initial_space and keep the panel persistent.
    if "wizard_compartments" not in st.session_state:
        st.session_state["wizard_compartments"] = initial_space.compartments
        st.session_state["wizard_residual_error_models"] = initial_space.residual_error_models
        st.session_state["wizard_bsv_parameters"] = initial_space.bsv_parameters
        st.session_state["wizard_bov_parameters"] = initial_space.bov_parameters
        for param in MODEL_PARAMETER_ORDER:
            st.session_state[_cov_state_key(param)] = initial_space.covariates_by_param.get(param, [])
        st.session_state["wizard_selected_algos"] = ["GP"]
        st.session_state["wizard_administration"] = ADMIN_FIRST_ORDER_ORAL
        st.session_state["wizard_test_tlag"] = False
        st.session_state["wizard_use_effect_limit"] = False
        st.session_state["wizard_effect_limit"] = 4
        st.session_state["wizard_working_dir"] = DEFAULT_WORKING_DIR
        st.session_state["wizard_output_dir"] = DEFAULT_OUTPUT_DIR
        st.session_state["wizard_temp_dir"] = DEFAULT_TEMP_DIR
        st.session_state["wizard_nmfe_path"] = DEFAULT_NMFE_PATH

    st.session_state.setdefault("wizard_administration", ADMIN_FIRST_ORDER_ORAL)
    st.session_state.setdefault("wizard_test_tlag", False)
    st.session_state.setdefault("wizard_use_effect_limit", False)
    st.session_state.setdefault("wizard_effect_limit", 4)
    st.session_state.setdefault("wizard_working_dir", DEFAULT_WORKING_DIR)
    st.session_state.setdefault("wizard_output_dir", DEFAULT_OUTPUT_DIR)
    st.session_state.setdefault("wizard_temp_dir", DEFAULT_TEMP_DIR)
    st.session_state.setdefault("wizard_nmfe_path", DEFAULT_NMFE_PATH)

    cov_patterns: dict[str, dict[str, list[str]]] = {}
    categorical_additive_bounds: dict[str, dict[str, list[float]]] = {}

    # Valid covariate names after user filters
    all_cov_names = list(covariates.keys())

    for param in MODEL_PARAMETER_ORDER:
        sanitized = [
            x for x in st.session_state.get(_cov_state_key(param), [])
            if x in all_cov_names
        ]
        st.session_state[_cov_state_key(param)] = sanitized

    with st.expander("Search space (edit if needed)", expanded=True):
        current_administration = st.session_state.get("wizard_administration", ADMIN_FIRST_ORDER_ORAL)
        bsv_options = available_bsv_parameters(
            st.session_state.get("wizard_compartments", initial_space.compartments),
            current_administration,
        )
        bsv_default = [
            x for x in st.session_state.get("wizard_bsv_parameters", [])
            if x in bsv_options
        ]
        st.session_state["wizard_bsv_parameters"] = bsv_default
        cols = st.columns(3)
        with cols[0]:
            compartments = st.multiselect(
                "Compartments",
                options=[1, 2, 3],
                default=st.session_state["wizard_compartments"],
            )
        with cols[1]:
            residual_error_models = st.multiselect(
                "Residual error models",
                options=["additive", "proportional", "combined"],
                default=st.session_state["wizard_residual_error_models"],
            )
        with cols[2]:
            bsv_parameters = st.multiselect(
                "Between-subject variability on parameters",
                options=bsv_options,
                default=bsv_default,
            )

        bov_parameters: List[str] = []
        if has_occ:
            bov_parameters = st.multiselect(
                "Between-occasion variability (requires OCC column)",
                options=["CL", "V"],
                default=st.session_state["wizard_bov_parameters"],
            )
        else:
            st.caption("No `OCC`/`OCCASION` column detected, so BOV is disabled.")
            bov_parameters = []

        def _admin_label(x: str) -> str:
            if x == ADMIN_FIRST_ORDER_ORAL:
                return "First-order oral (ADVAN2/4/12 + TRANS; TVKA/KA)"
            return "Infusion (ADVAN1/3/11 + TRANS; no TVKA/KA)"

        administration = st.radio(
            "Drug input / absorption (sets route-specific template/tokens)",
            options=[ADMIN_FIRST_ORDER_ORAL, ADMIN_INFUSION],
            format_func=_admin_label,
            key="wizard_administration",
        )

        if administration == ADMIN_FIRST_ORDER_ORAL:
            test_tlag = st.checkbox(
                "Test lag time (ALAG1/TLAG)",
                value=st.session_state.get("wizard_test_tlag", False),
                key="wizard_test_tlag",
                help="If enabled, keep the ALAG token so pyDarwin can test no lag vs lag time.",
            )
        else:
            st.session_state["wizard_test_tlag"] = False
            test_tlag = False
            st.caption("Lag time testing is only available for first-order oral absorption.")

        active_bsv_options = available_bsv_parameters(compartments, administration)
        bsv_parameters = [x for x in bsv_parameters if x in active_bsv_options]

        active_covariate_params = available_model_parameters(compartments, administration)
        st.markdown("**Covariates by parameter (edit selections)**")
        covariates_by_param: dict[str, List[str]] = {}
        for param in active_covariate_params:
            widget_key = f"ui_covs_{param}"
            default_covs = st.session_state.get(_cov_state_key(param), [])
            if widget_key not in st.session_state:
                st.session_state[widget_key] = default_covs
            covariates_by_param[param] = st.multiselect(
                f"Covariates on {param}",
                options=all_cov_names,
                key=widget_key,
            )

        st.subheader("Covariate patterns by parameter")
        for param in active_covariate_params:
            cov_list = covariates_by_param.get(param, [])
            cov_patterns[param] = {}
            for cov in cov_list:
                cinfo = covariates.get(cov)
                if not cinfo:
                    continue
                if cinfo.cov_type == "continuous":
                    patterns = st.multiselect(
                        f"{cov} patterns on {param}",
                        options=["none", "power", "exponential"],
                        default=["none", "power", "exponential"],
                        key=f"{param}_{cov}_patterns",
                    )
                else:
                    patterns = st.multiselect(
                        f"{cov} categorical relationships on {param}",
                        options=["none", "proportional", "exponential", "additive"],
                        default=["none", "proportional", "exponential"],
                        key=f"{param}_{cov}_categorical_patterns",
                        help=(
                            "Additive is optional because it can permit non-positive "
                            "typical PK parameters. Separate typical estimates are "
                            "not included because they are nonhierarchical."
                        ),
                    )
                    if "additive" in patterns:
                        st.warning(
                            f"Choose explicit additive bounds for {param}~{cov}."
                        )
                        typical = {
                            "CL": 8.0,
                            "V": 80.0,
                            "KA": 0.5,
                        }.get(param, 1.0)
                        bound_cols = st.columns(3)
                        lower = bound_cols[0].number_input(
                            "Lower",
                            value=-0.5 * typical,
                            key=f"{param}_{cov}_additive_lower",
                        )
                        initial = bound_cols[1].number_input(
                            "Initial",
                            value=0.0,
                            key=f"{param}_{cov}_additive_initial",
                        )
                        upper = bound_cols[2].number_input(
                            "Upper",
                            value=0.5 * typical,
                            key=f"{param}_{cov}_additive_upper",
                        )
                        categorical_additive_bounds.setdefault(param, {})[cov] = [
                            float(lower),
                            float(initial),
                            float(upper),
                        ]
                cov_patterns[param][cov] = patterns

    # Keep wizard state in sync so user changes persist.
    st.session_state["wizard_compartments"] = compartments
    st.session_state["wizard_residual_error_models"] = residual_error_models
    st.session_state["wizard_bsv_parameters"] = bsv_parameters
    st.session_state["wizard_bov_parameters"] = bov_parameters
    for param, cov_list in covariates_by_param.items():
        st.session_state[_cov_state_key(param)] = cov_list

    administration = st.session_state.get("wizard_administration", ADMIN_FIRST_ORDER_ORAL)
    test_tlag = bool(st.session_state.get("wizard_test_tlag", False))

    st.subheader("3. Covariate centering (medians)")
    st.markdown(
        "For numeric covariates used in the model, confirm or adjust the median values "
        "used for centering."
    )
    used_covariate_names = sorted(
        {cov for covs in covariates_by_param.values() for cov in covs},
    )

    for name in used_covariate_names:
        info = covariates.get(name)
        if info is None or not info.is_numeric or info.median is None:
            continue
        with st.expander(f"{name} (median ≈ {info.median:.3g})", expanded=False):
            median_val = st.number_input(
                f"Median value for {name}",
                value=float(info.median),
            )
            center_on_one = st.checkbox(
                f"Center {name} on one (divide by median)",
                value=True,
                key=f"{name}_center_one",
            )
            center_on_zero = st.checkbox(
                f"Center {name} on zero (subtract median)",
                value=True,
                key=f"{name}_center_zero",
            )
            info.median = float(median_val)
            info.center_on_one = bool(center_on_one)
            info.center_on_zero = bool(center_on_zero)
            covariates[name] = info

    st.subheader("4. Machine learning algorithm options")
    algo_options = default_algorithm_options()
    selected_algos = st.multiselect(
        "Select one or more algorithms for pyDarwin to use",
        options=algo_options,
        default=st.session_state["wizard_selected_algos"],
    )
    st.session_state["wizard_selected_algos"] = selected_algos
    if not selected_algos:
        st.warning("Select at least one algorithm.")

    effect_limit_algorithms = {"GA", "MOGA"}
    uses_effect_limit = any(algo in effect_limit_algorithms for algo in selected_algos)
    effect_limit: int | None = None
    if uses_effect_limit:
        use_effect_limit = st.checkbox(
            "Use effect limit for GA/MOGA",
            value=st.session_state.get("wizard_use_effect_limit", False),
            key="wizard_use_effect_limit",
            help="If unchecked, generated GA/MOGA outputs will omit effect metadata and effect_limit.",
        )
        if use_effect_limit:
            effect_limit = int(
                st.number_input(
                    "Effect limit",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("wizard_effect_limit", 4)),
                    key="wizard_effect_limit",
                    help="Maximum number of enabled effects for algorithms that support effect_limit.",
                )
            )
        else:
            effect_limit = None
    else:
        st.session_state["wizard_use_effect_limit"] = False

    st.subheader("5. NONMEM launcher and run directories for `options.json`")
    nmfe_path = st.text_input(
        "NONMEM launcher path (`nmfe_path`)",
        key="wizard_nmfe_path",
        help=r"Example: C:\nm751\util\nmfe75.bat",
    )
    st.caption(
        "The defaults keep pyDarwin runtime files in separate subfolders. "
        "You can replace them with absolute Windows paths if needed."
    )
    working_dir = st.text_input(
        "Working directory",
        key="wizard_working_dir",
    )
    output_dir = st.text_input(
        "Output directory",
        key="wizard_output_dir",
    )
    temp_dir = st.text_input(
        "Temp directory",
        key="wizard_temp_dir",
    )
    _ensure_reporter_defaults(working_dir.strip(), output_dir.strip())

    search_space = SearchSpace(
        compartments=compartments,
        covariates_by_param=covariates_by_param,
        residual_error_models=residual_error_models,
        bsv_parameters=bsv_parameters,
        bov_parameters=bov_parameters,
        has_occasion=has_occ,
    )

    user_choices = UserChoices(
        search_space=search_space,
        covariates=covariates,
        algorithms=selected_algos,
        covariate_patterns_by_param=cov_patterns,
        categorical_additive_bounds_by_param=categorical_additive_bounds,
        administration=administration,
        test_tlag=test_tlag,
        effect_limit=effect_limit,
        working_dir=working_dir.strip(),
        output_dir=output_dir.strip(),
        temp_dir=temp_dir.strip(),
        nmfe_path=nmfe_path.strip(),
    )

    st.subheader("6. Generate pyDarwin configuration files")
    base_output_dir = PROJECT_ROOT / "generated_configs"

    if st.button("Generate `template.txt`, `tokens.json`, and `options.json`"):
        if not selected_algos:
            st.error("Please select at least one algorithm before generating files.")
        else:
            csv_bytes = (
                uploaded_file.getvalue()
                if uploaded_file is not None
                else st.session_state.get("uploaded_file_bytes")
            )
            csv_filename = (
                uploaded_file.name if uploaded_file is not None else st.session_state.get("uploaded_file_name", "data.csv")
            )
            generated = generate_py_darwin_files(
                csv_data=csv_bytes,
                csv_filename=csv_filename,
                user_choices=user_choices,
                base_output_dir=base_output_dir,
            )
            st.session_state["last_generated_files"] = generated
            generated_paths = generated.get("template", []) or generated.get("tokens", []) or generated.get("options", [])
            if generated_paths:
                _set_reporter_defaults_from_generated(Path(generated_paths[0]).parent, user_choices)
            st.rerun()

    generated = st.session_state.get("last_generated_files")
    if generated:
        st.success("pyDarwin configuration files generated.")
        st.markdown("**Generated files:**")
        for kind, paths in generated.items():
            for index, file_path in enumerate(paths):
                file_columns = st.columns([4, 1])
                with file_columns[0]:
                    st.write(f"- {kind}: `{file_path}`")
                with file_columns[1]:
                    if st.button(
                        "Open and edit",
                        key=f"open_generated_file::{kind}::{index}::{file_path}",
                    ):
                        path = Path(file_path)
                        editor_key = f"generated_file_editor::{path}"
                        try:
                            st.session_state[editor_key] = path.read_text(
                                encoding="utf-8"
                            )
                        except OSError as exc:
                            st.error(f"Could not open the file: {exc}")
                        else:
                            _render_generated_file_editor(file_path, generated)

        st.subheader("7. Run pyDarwin in a Windows terminal")
        st.caption(
            "This performs a preflight check, then opens the official pyDarwin command "
            "in a separate Windows terminal. It does not run NONMEM inside Streamlit."
        )
        python_executable = st.text_input(
            "Python executable containing pyDarwin",
            value=sys.executable,
            key="pydarwin_python_executable",
            help=r"Example: C:\pydarwin\.venv\Scripts\python.exe",
        )
        option_paths = generated.get("options", [])
        selected_options = st.selectbox(
            "Options file / algorithm to run",
            options=option_paths,
            format_func=lambda path: Path(path).name,
            key="pydarwin_selected_options",
        )
        template_paths = generated.get("template", [])
        tokens_paths = generated.get("tokens", [])
        if template_paths and tokens_paths and selected_options:
            run_command = build_run_command(
                python_executable,
                template_paths[0],
                tokens_paths[0],
                selected_options,
            )
            st.markdown("**Command preview:**")
            st.code(format_windows_command(run_command), language="bat")
            if not sys.platform.startswith("win"):
                st.info(
                    "This computer is not Windows, so the launcher will remain disabled. "
                    "Use this project copy on the Windows computer that has NONMEM."
                )
            if st.button(
                "Run pyDarwin in Terminal",
                disabled=not sys.platform.startswith("win"),
            ):
                preflight_errors = validate_run_environment(
                    python_executable,
                    template_paths[0],
                    tokens_paths[0],
                    selected_options,
                )
                if preflight_errors:
                    for error in preflight_errors:
                        st.error(error)
                else:
                    try:
                        launch_in_windows_terminal(
                            run_command,
                            Path(selected_options).parent,
                        )
                        st.success(
                            "pyDarwin was launched in a new terminal. "
                            "Keep that terminal open to monitor progress and errors."
                        )
                    except Exception as exc:
                        st.error(f"Failed to launch pyDarwin: {exc}")

    st.subheader("8. Visualization with DarwinReporter")
    st.caption(
        "Confirm the pyDarwin results directories below, then launch the DarwinReporter Shiny app. "
        "This step is optional and separate from file generation."
    )
    if not st.session_state.get("last_generated_project_dir"):
        st.info(
            "No generated project directory is stored yet. You can still paste an existing completed "
            "pyDarwin run below and launch DarwinReporter manually."
        )

    reporter_mode = st.radio(
        "Reporter search type",
        options=[REPORTER_MODE_SINGLE, REPORTER_MODE_MOGA],
        format_func=lambda value: "Single-objective" if value == REPORTER_MODE_SINGLE else "MOGA",
        key="reporter_mode",
        horizontal=True,
    )
    reporter_project_dir = st.text_input(
        "Reporter `project_dir`",
        key="reporter_project_dir",
    )
    reporter_working_dir = st.text_input(
        "Reporter `working_dir`",
        key="reporter_working_dir",
    )
    reporter_output_dir = st.text_input(
        "Reporter `output_dir`",
        key="reporter_output_dir",
    )
    reporter_r_path = st.text_input(
        "Path to `R` executable",
        key="reporter_r_path",
    )
    reporter_rscript_path = st.text_input(
        "Path to `Rscript` executable",
        key="reporter_rscript_path",
    )
    if reporter_mode == REPORTER_MODE_SINGLE:
        reporter_key_models_dir = st.text_input(
            "Reporter `key_models_dir`",
            key="reporter_key_models_dir",
        )
        reporter_non_dominated_models_dir = st.session_state.get(
            "reporter_non_dominated_models_dir",
            "",
        )
        st.caption(
            "Single-objective launch uses `project_dir`, `working_dir`, `output_dir`, "
            "and `key_models_dir`."
        )
    else:
        reporter_key_models_dir = st.session_state.get("reporter_key_models_dir", "")
        reporter_non_dominated_models_dir = st.text_input(
            "Reporter `non_dominated_models_dir`",
            key="reporter_non_dominated_models_dir",
        )
        st.caption(
            "MOGA launch uses `project_dir`, `working_dir`, `output_dir`, and `non_dominated_models_dir`."
        )

    if st.button("Launch DarwinReporter"):
        requirement_error, resolved_r_path, _ = _check_reporter_requirements(
            reporter_r_path,
            reporter_rscript_path,
        )
        if requirement_error:
            st.error(requirement_error)
        else:
            validation_errors = _validate_reporter_inputs(
                reporter_mode,
                reporter_project_dir,
                reporter_working_dir,
                reporter_output_dir,
                reporter_key_models_dir,
                reporter_non_dominated_models_dir,
            )
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
            else:
                try:
                    _launch_reporter_in_terminal(
                        resolved_r_path,
                        reporter_mode,
                        reporter_project_dir.strip(),
                        reporter_working_dir.strip(),
                        reporter_output_dir.strip(),
                        reporter_key_models_dir.strip(),
                        reporter_non_dominated_models_dir.strip(),
                    )
                    st.success(
                        "Launching DarwinReporter in a new terminal window. "
                        "Your browser should open when the Shiny app starts."
                    )
                except Exception as exc:
                    st.error(f"Failed to launch DarwinReporter: {exc}")


if __name__ == "__main__":
    main()
