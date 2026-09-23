"""
Shared logic for pyDarwin configuration generation.
Used by both the Streamlit UI and the local API.
Integrates rules from .cursor/skills/pydarwin-expert/SKILL.md.
"""
from __future__ import annotations

import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_ROOT / "reference"
SKILL_PATH = PROJECT_ROOT / ".cursor" / "skills" / "pydarwin-expert" / "SKILL.md"

PATTERN_PATH = PROJECT_ROOT / "tokens_covariate_pattern.json"

# Compartment limit per SKILL: only 1, 2, or 3
ALLOWED_COMPARTMENTS = (1, 2, 3)

# Input / absorption: picks reference/first_order vs reference/zero_order (template + tokens).
ADMIN_FIRST_ORDER_ORAL = "first_order_oral"
ADMIN_INFUSION = "infusion"
ADMINISTRATION_OPTIONS = (ADMIN_FIRST_ORDER_ORAL, ADMIN_INFUSION)
RESIDUAL_ERROR_MODEL_ORDER = ("additive", "proportional", "combined")
MODEL_PARAMETER_ORDER = ("CL", "V", "Q", "V2", "V3", "V4", "KA", "Q2", "Q3")
FIRST_ORDER_PERIPHERAL_PARAMS_BY_COMPARTMENT = {
    2: ["Q", "V3"],
    3: ["Q2", "Q3", "V3", "V4"],
}
ZERO_ORDER_PERIPHERAL_PARAMS_BY_COMPARTMENT = {
    2: ["Q", "V2"],
    3: ["Q2", "Q3", "V2", "V3"],
}
FIRST_ORDER_PERIPHERAL_BSV_TOKEN_MAP = {
    "Q": "BSVQ2",
    "Q2": "BSVQ2",
    "Q3": "BSVQ3",
    "V3": "BSVV3",
    "V4": "BSVV4",
}
ZERO_ORDER_PERIPHERAL_BSV_TOKEN_MAP = {
    "Q": "BSVQ2",
    "Q2": "BSVQ2",
    "Q3": "BSVQ3",
    "V2": "BSVV2",
    "V3": "BSVV3",
}

REFERENCE_OPTIONS_FOR_ALGORITHMS_DIR = REFERENCE_DIR / "options for algorithms"
FIRST_ORDER_REF_DIR = REFERENCE_DIR / "first_order"
ZERO_ORDER_REF_DIR = REFERENCE_DIR / "zero_order"

# Placeholders in template that are resolved at runtime (path, etc.), not by tokens.json
RESERVED_TEMPLATE_PLACEHOLDERS = frozenset({"data_dir"})

# Reference tokens use names like WT, SEX; UI/CSV may use WTKG, GENDER. Allow both.
COVARIATE_TOKEN_ALIASES: Dict[str, List[str]] = {
    "WT": ["WT", "WTKG", "WEIGHT"],
    "SEX": ["SEX", "GENDER"],
    "GENDER": ["GENDER", "SEX"],
    "AGE": ["AGE"],
    "SCR": ["SCR"],
    "BUN": ["BUN"],
    "CYP3": ["CYP3"],
    "RACE": ["RACE", "race"],
    "FORM": ["FORM"],
    "TRT": ["TRT"],
}


@dataclass
class CovariateInfo:
    name: str
    is_numeric: bool
    median: Optional[float] = None
    center_on_one: bool = False
    center_on_zero: bool = False
    cov_type: str = "continuous"      # "continuous" or "categorical"
    is_binary: Optional[bool] = None  # only meaningful if categorical
    levels: List[Any] | None = None
    level_counts: Dict[str, int] | None = None
    reference_level: Any = None
    within_subject_consistent: bool = True
    allow_time_varying: bool = False


@dataclass
class SearchSpace:
    compartments: List[int]
    covariates_by_param: Dict[str, List[str]]
    residual_error_models: List[str]
    bsv_parameters: List[str]
    bov_parameters: List[str]
    has_occasion: bool


@dataclass
class UserChoices:
    search_space: SearchSpace
    covariates: Dict[str, CovariateInfo]
    algorithms: List[str]
    covariate_patterns_by_param: Dict[str, Dict[str, List[str]]] | None = None
    categorical_additive_bounds_by_param: Dict[str, Dict[str, List[float]]] | None = None
    # Reference route: first-order oral (ADVAN2/4/12 + TRANS) vs infusion (ADVAN1/3/11 + TRANS).
    administration: str = ADMIN_FIRST_ORDER_ORAL
    # First-order only: if true, keep ALAG token so pyDarwin can test lag time.
    test_tlag: bool = False
    # Optional pyDarwin effect-limit for GA/MOGA. None means "not requested";
    # generated GA/MOGA options will write -1 for no limit.
    effect_limit: Optional[int] = None
    working_dir: Optional[str] = None
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    nmfe_path: Optional[str] = None


def _peripheral_params_by_compartment(administration: str) -> Dict[int, List[str]]:
    if administration == ADMIN_INFUSION:
        return ZERO_ORDER_PERIPHERAL_PARAMS_BY_COMPARTMENT
    return FIRST_ORDER_PERIPHERAL_PARAMS_BY_COMPARTMENT


def _peripheral_bsv_token_map(administration: str) -> Dict[str, str]:
    if administration == ADMIN_INFUSION:
        return ZERO_ORDER_PERIPHERAL_BSV_TOKEN_MAP
    return FIRST_ORDER_PERIPHERAL_BSV_TOKEN_MAP


def peripheral_parameter_names(
    compartments: List[int],
    administration: str,
) -> List[str]:
    """Return peripheral structural parameter names for the route and compartments."""
    params_by_compartment = _peripheral_params_by_compartment(administration)
    params: List[str] = []
    for compartment in (2, 3):
        if compartment in set(compartments):
            params.extend(params_by_compartment.get(compartment, []))
    ordered: List[str] = []
    for param in params:
        if param not in ordered:
            ordered.append(param)
    return ordered


def available_model_parameters(
    compartments: List[int],
    administration: str,
) -> List[str]:
    """Return parameter names that the UI/generator should expose."""
    params = ["CL", "V"]
    params.extend(peripheral_parameter_names(compartments, administration))
    if administration == ADMIN_FIRST_ORDER_ORAL:
        params.append("KA")
    ordered: List[str] = []
    for param in params:
        if param not in ordered:
            ordered.append(param)
    return ordered


def available_bsv_parameters(
    compartments: List[int],
    administration: str,
) -> List[str]:
    """Return parameter names that can have user-selectable BSV."""
    return available_model_parameters(compartments, administration)


def load_skill_content() -> str:
    """Load pyDarwin expert skill text from SKILL.md. Returns empty string if file missing."""
    if not SKILL_PATH.exists():
        return ""
    return SKILL_PATH.read_text()


def load_csv(file_like: BinaryIO) -> pd.DataFrame:
    """Load a CSV from a file-like object (e.g. uploaded file or BytesIO)."""
    return pd.read_csv(file_like)


def load_csv_from_bytes(data: bytes) -> pd.DataFrame:
    """Load a CSV from raw bytes (e.g. API upload)."""
    return pd.read_csv(io.BytesIO(data))


def load_covariate_patterns() -> Dict[str, Any]:
    return json.loads(PATTERN_PATH.read_text())


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _level_sort_key(value: Any) -> Tuple[int, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _infer_level_metadata(
    df: pd.DataFrame,
    column: Any,
) -> Tuple[List[Any], Dict[str, int], Any, bool]:
    values = [_python_scalar(value) for value in pd.unique(df[column].dropna())]
    levels = sorted(values, key=_level_sort_key)
    id_column = next(
        (
            candidate
            for candidate in df.columns
            if str(candidate).strip().upper() in {"ID", "#ID", "C ID"}
        ),
        None,
    )
    consistent = True
    if id_column is not None:
        pairs = df[[id_column, column]].dropna(subset=[id_column, column])
        per_subject_unique = pairs.groupby(id_column, dropna=False)[column].nunique()
        consistent = bool((per_subject_unique <= 1).all())
        representatives = pairs.drop_duplicates(subset=[id_column], keep="first")[column]
        counts_series = representatives.value_counts(dropna=True)
    else:
        counts_series = df[column].value_counts(dropna=True)
    counts = {
        str(_python_scalar(level)): int(counts_series.get(level, 0))
        for level in levels
    }
    reference = None
    if levels:
        reference = sorted(
            levels,
            key=lambda level: (-counts[str(level)], _level_sort_key(level)),
        )[0]
    return levels, counts, reference, consistent


def _non_reference_levels(info: CovariateInfo) -> List[Any]:
    levels = list(info.levels or [])
    return [level for level in levels if level != info.reference_level]


def _categorical_indicator_name(covariate: str, index: int) -> str:
    safe_covariate = re.sub(r"[^A-Za-z0-9_]", "_", covariate.upper())
    return f"{safe_covariate}_LVL{index}"


def _format_nonmem_level(level: Any) -> str:
    value = float(level)
    return f"{value:g}"


def infer_covariates(df: pd.DataFrame) -> Dict[str, CovariateInfo]:
    # Structural + SKILL: Dose, ID, CMT, TAD cannot be covariates (biological feasibility)
    known_struct_cols = {
        "ID",
        "TIME",
        "AMT",
        "CMT",
        "DV",
        "RATE",
        "MDV",
        "EVID",
        "SS",
        "II",
        "DOSE",
        "TAD",
        "#ID",
        "c ID",
        "OCC",
    }
    covariates: Dict[str, CovariateInfo] = {}
    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean.upper() in known_struct_cols:
            continue
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        median: Optional[float] = None
        if is_numeric:
            median = float(series.median(skipna=True))
        levels, level_counts, reference_level, consistent = _infer_level_metadata(
            df,
            col,
        )

        info = CovariateInfo(
            name=col_clean,
            is_numeric=is_numeric,
            median=median,
            levels=levels,
            level_counts=level_counts,
            reference_level=reference_level,
            within_subject_consistent=consistent,
        )

        likely_categorical = (
            not is_numeric
            or any(
                key in col_clean.upper()
                for key in ("SEX", "GENDER", "RACE", "ETHNIC", "FORM", "TRT")
            )
        )
        if likely_categorical:
            info.cov_type = "categorical"
            info.is_binary = len(levels) == 2
        else:
            info.cov_type = "continuous"

        covariates[col_clean] = info
    return covariates


def has_occasion_column(df: pd.DataFrame) -> bool:
    for col in df.columns:
        name = str(col).strip().upper()
        if name in {"OCC", "OCCASION"}:
            return True
    return False


def propose_search_space(
    df: pd.DataFrame, covariates: Dict[str, CovariateInfo]
) -> SearchSpace:
    has_occ = has_occasion_column(df)

    numeric_covs = [
        c.name for c in covariates.values() if c.cov_type == "continuous"
    ]
    cat_covs = [
        c.name for c in covariates.values() if c.cov_type == "categorical"
    ]

    cl_covs: List[str] = []
    v_covs: List[str] = []
    ka_covs: List[str] = []

    for name in numeric_covs:
        upper = name.upper()
        if "WT" in upper or "WEIGHT" in upper:
            cl_covs.append(name)
            v_covs.append(name)
        elif "AGE" in upper:
            cl_covs.append(name)
            v_covs.append(name)
            ka_covs.append(name)
        elif any(key in upper for key in ("SCR", "CREAT", "CRCL", "EGFR", "RENAL")):
            cl_covs.append(name)
        elif any(key in upper for key in ("BUN", "LFT", "ALT", "AST", "ALB")):
            cl_covs.append(name)
        elif any(key in upper for key in ("FORM", "TRT", "DOSE", "REGIMEN")):
            ka_covs.append(name)
        else:
            cl_covs.append(name)

    for name in cat_covs:
        upper = name.upper()
        if any(key in upper for key in ("SEX", "GENDER")):
            cl_covs.append(name)
            v_covs.append(name)
        elif "RACE" in upper or "ETHNIC" in upper:
            cl_covs.append(name)

    cl_covs = sorted(dict.fromkeys(cl_covs))
    v_covs = sorted(dict.fromkeys(v_covs))
    ka_covs = sorted(dict.fromkeys(ka_covs))

    return SearchSpace(
        compartments=[1, 2, 3],
        covariates_by_param={
            "CL": cl_covs,
            "V": v_covs,
            "Q": [],
            "KA": ka_covs,
            "V2": [],
            "V3": [],
            "V4": [],
            "Q2": [],
            "Q3": [],
        },
        residual_error_models=["additive", "proportional", "combined"],
        bsv_parameters=["CL", "V", "KA"],
        bov_parameters=["CL", "V"] if has_occ else [],
        has_occasion=has_occ,
    )


def default_algorithm_options() -> List[str]:
    return [
        "GA",
        "GBRT",
        "GP",
        "RF",
        "EXHAUSTIVE",
        "PSO",
        "MOGA",
    ]


def route_reference_dir(administration: str) -> Path:
    """
    Directory containing template.txt and tokens.json for the chosen input route.
    Not algorithm-specific — pyDarwin search algorithm comes from options JSON only.
    """
    if administration == ADMIN_INFUSION:
        return ZERO_ORDER_REF_DIR
    return FIRST_ORDER_REF_DIR


def load_reference_template_for_administration(administration: str) -> str:
    ref_dir = route_reference_dir(administration)
    template_path = ref_dir / "template.txt"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Missing template for administration={administration!r}: {template_path}"
        )
    return template_path.read_text()


def load_reference_options(algorithm_name: str) -> Dict[str, Any]:
    """
    Load pyDarwin options for the search algorithm from
    reference/options for algorithms/{ALGO}_options.json.
    """
    algo_upper = algorithm_name.strip().upper()
    options_path = REFERENCE_OPTIONS_FOR_ALGORITHMS_DIR / f"{algo_upper}_options.json"
    if not options_path.is_file():
        raise FileNotFoundError(
            f"Missing algorithm options file: {options_path} "
            f"(expected reference/options for algorithms/{algo_upper}_options.json)"
        )
    with options_path.open() as f:
        options = json.load(f)
    options["algorithm"] = algo_upper
    options.setdefault("author", "pyDarwin Agent")
    return options


def _normalize_options_dir_value(path_text: str) -> str:
    cleaned = str(path_text).strip().replace("\\", "/")
    cleaned = re.sub(r"/+", "/", cleaned)
    return cleaned.rstrip("/") or cleaned


def _apply_run_directories(
    options: Dict[str, Any],
    user_choices: "UserChoices",
) -> Dict[str, Any]:
    """Ensure generated options point to the selected run and bundled data directories."""
    updated = dict(options)
    updated["data_dir"] = "{project_dir}"
    updated["working_dir"] = _normalize_options_dir_value(
        user_choices.working_dir or "{project_dir}/working"
    )
    updated["output_dir"] = _normalize_options_dir_value(
        user_choices.output_dir or "{working_dir}/output"
    )
    updated["temp_dir"] = _normalize_options_dir_value(
        user_choices.temp_dir or "{working_dir}/temp"
    )
    return updated


def _option_with_effect_count(
    code: str,
    init: str,
    effect_count: int,
) -> List[str]:
    return [code, init, f" effects = {effect_count}"]


def _plain_option(
    code: str,
    init: str,
) -> List[str]:
    return [code, init]


def _format_generated_theta_init(
    pk_fragment: str,
    theta_init: str,
    relationship_label: str = "",
) -> str:
    """
    Add pyDarwin-recognizable THETA annotations to generated init lines based on the
    actual THETA(...) names present in the PK fragment.
    """
    theta_names = list(dict.fromkeys(re.findall(r"THETA\(([^)]+)\)", pk_fragment)))
    init_lines = [line.strip() for line in theta_init.splitlines() if line.strip()]
    if not theta_names or not init_lines:
        return theta_init

    if len(init_lines) == 1 and len(theta_names) > 1:
        init_lines = init_lines * len(theta_names)
    elif len(init_lines) < len(theta_names):
        init_lines.extend([init_lines[-1]] * (len(theta_names) - len(init_lines)))

    formatted: List[str] = []
    for theta_name, init_line in zip(theta_names, init_lines):
        suffix = f" \t; THETA({theta_name})"
        if relationship_label:
            suffix += f" {relationship_label}"
        formatted.append(f"{init_line}{suffix}")
    return "\n".join(formatted)


def _categorical_token_option(
    param: str,
    covariate: str,
    info: CovariateInfo,
    relationship: str,
    patterns: Dict[str, Any],
    additive_bounds: Optional[List[float]] = None,
) -> Tuple[str, str]:
    non_reference = _non_reference_levels(info)
    terms: List[str] = []
    theta_lines: List[str] = []
    relationship_spec = patterns["categorical"]["relationships"][relationship]
    for index, _ in enumerate(non_reference, start=1):
        indicator = _categorical_indicator_name(covariate, index)
        theta_name = f"{param}~{covariate}_L{index}"
        terms.append(f"THETA({theta_name})*{indicator}")
        if relationship == "additive":
            if additive_bounds is None:
                raise ValueError(
                    f"{param}~{covariate}: additive relationship requires bounds."
                )
            theta_init = f"({additive_bounds[0]:g},{additive_bounds[1]:g},{additive_bounds[2]:g})"
        else:
            theta_init = relationship_spec["theta_init"]
        theta_lines.append(
            f"{theta_init} \t; THETA({theta_name}) {relationship.upper()}"
        )
    joined = " + ".join(terms)
    if relationship == "additive":
        pk_fragment = f"+{joined}"
    elif relationship == "proportional":
        pk_fragment = f"*(1 + {joined})"
    elif relationship == "exponential":
        pk_fragment = f"*EXP({joined})"
    else:
        raise ValueError(f"Unsupported categorical relationship: {relationship}")
    return pk_fragment, "\n".join(theta_lines)


def _normalize_token_effect_metadata(
    token_name: str,
    token_options: List[Any],
) -> List[Any]:
    """
    Ensure each token option carries a trailing " effects = N" metadata string.
    Covariate effects and lag time count as 1 when active; structural and variance
    choices count as 0.
    """
    normalized: List[Any] = []
    for idx, option in enumerate(token_options):
        if not isinstance(option, list):
            normalized.append(option)
            continue
        option_list = list(option)

        existing_effect_line: Optional[str] = None
        cleaned_option_list: List[Any] = []
        for item in option_list:
            if isinstance(item, str) and "effects =" in item:
                existing_effect_line = item
                continue
            cleaned_option_list.append(item)

        effect_count = 0
        if "~" in token_name and idx > 0:
            effect_count = 1
        elif token_name == "ALAG" and idx > 0:
            effect_count = 1

        while len(cleaned_option_list) < 2:
            cleaned_option_list.append("")
        effect_line = existing_effect_line or f" effects = {effect_count}"
        cleaned_option_list.append(effect_line)
        normalized.append(cleaned_option_list)
    return normalized


def _inject_suffix_after_theta(line: str, suffix: str) -> str:
    """Insert suffix immediately after the first THETA(...) on the line."""
    if not suffix:
        return line
    upper = line.upper()
    idx = upper.find("THETA(")
    if idx == -1:
        return line + suffix
    start = idx + len("THETA(")
    depth = 1
    for i in range(start, len(line)):
        ch = line[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return line[: i + 1] + suffix + line[i + 1 :]
    return line + suffix


def _placeholder_suffix_for_covs(
    param: str,
    covs: List[str],
    token_index: int,
    user_choices: "UserChoices",
) -> str:
    """Build placeholder suffixes like {Q~WT[1]} or separate init lines [2]."""
    placeholders: List[str] = []
    for cov in dict.fromkeys(covs):
        if user_choices.covariates.get(cov) is None:
            continue
        if token_index == 1:
            placeholders.append(f" {{{param}~{cov}[1]}}")
        else:
            placeholders.append(f"{{{param}~{cov}[2]}}")
    if token_index == 1:
        return "".join(placeholders)
    if not placeholders:
        return ""
    return ("\n" if covs else "") + "\n".join(placeholders)


def _apply_effect_limit_option(
    options: Dict[str, Any],
    algorithm_name: str,
    effect_limit: Optional[int],
) -> Dict[str, Any]:
    """Set or remove effect_limit for GA/MOGA based on the UI checkbox."""
    algo_upper = algorithm_name.strip().upper()
    if algo_upper not in {"GA", "MOGA"}:
        return options

    updated = dict(options)
    if effect_limit is None:
        updated.pop("effect_limit", None)
    else:
        updated["effect_limit"] = int(effect_limit)
    return updated


def _uses_effect_metadata(algorithms: List[str], effect_limit: Optional[int]) -> bool:
    return effect_limit is not None and any(
        algo.strip().upper() in {"GA", "MOGA"} for algo in algorithms
    )


def _filter_advan_token_by_compartments(
    tokens_dict: Dict[str, Any],
    administration: str,
    compartments: List[int],
) -> None:
    """
    Keep only ADVAN token options that match the selected compartment counts.

    first_order_oral: 1 -> ADVAN2, 2 -> ADVAN4, 3 -> ADVAN12
    infusion:         1 -> ADVAN1, 2 -> ADVAN3, 3 -> ADVAN11
    """
    advan_options = tokens_dict.get("ADVAN")
    if not isinstance(advan_options, list) or not advan_options:
        return

    advan_by_number: Dict[int, Any] = {}
    for option in advan_options:
        if not isinstance(option, list) or not option or not isinstance(option[0], str):
            continue
        match = re.match(r"\s*ADVAN(\d+)\b", option[0], re.IGNORECASE)
        if not match:
            continue
        advan_by_number[int(match.group(1))] = option

    expected_by_compartment = (
        {1: 1, 2: 3, 3: 11}
        if administration == ADMIN_INFUSION
        else {1: 2, 2: 4, 3: 12}
    )

    ordered_compartments = list(dict.fromkeys(compartments))
    filtered_options: List[Any] = []
    missing: List[str] = []
    for comp in ordered_compartments:
        advan_number = expected_by_compartment.get(comp)
        option = advan_by_number.get(advan_number)
        if option is None:
            missing.append(f"ADVAN{advan_number} for {comp}-compartment")
            continue
        filtered_options.append(option)

    if missing:
        raise ValueError(
            "Reference ADVAN token is missing expected options: " + ", ".join(missing)
        )

    if filtered_options:
        tokens_dict["ADVAN"] = filtered_options


def _prune_peripheral_tokens_by_compartments(
    tokens_dict: Dict[str, Any],
    administration: str,
    compartments: List[int],
) -> None:
    """
    Remove peripheral-compartment tokens that are impossible for the selected
    search space, so generated tokens.json does not carry unused 2- or
    3-compartment structures.
    """
    selected = set(compartments)
    all_peripheral_bsv_tokens = {"BSVQ2", "BSVQ3", "BSVV2", "BSVV3", "BSVV4"}
    bsv_token_map = _peripheral_bsv_token_map(administration)
    allowed_params = set(peripheral_parameter_names(compartments, administration))
    allowed_tokens = {bsv_token_map[param] for param in allowed_params if param in bsv_token_map}

    for token_name in all_peripheral_bsv_tokens - allowed_tokens:
        tokens_dict.pop(token_name, None)

    if 2 not in selected and 3 not in selected:
        tokens_dict.pop("IOVQ2", None)


def _selected_peripheral_bsv_tokens(
    search_space: SearchSpace,
    administration: str,
) -> set[str]:
    """Return peripheral BSV token names required by the user's BSV selections."""
    selected_tokens: set[str] = set()
    selected_params = set(search_space.bsv_parameters or [])
    for param, token_name in _peripheral_bsv_token_map(administration).items():
        if param in selected_params:
            selected_tokens.add(token_name)
    return selected_tokens


def _prune_peripheral_bsv_tokens(
    tokens_dict: Dict[str, Any],
    search_space: SearchSpace,
    administration: str,
) -> None:
    """Remove peripheral BSV token keys that the user chose not to test."""
    bsv_token_map = _peripheral_bsv_token_map(administration)
    selected_tokens = _selected_peripheral_bsv_tokens(search_space, administration)
    for token_name in set(bsv_token_map.values()) - selected_tokens:
        tokens_dict.pop(token_name, None)


def _update_template_data_path(template: str, csv_name: str) -> str:
    lines = template.splitlines()
    new_lines: List[str] = []
    for line in lines:
        if "$DATA" in line and "{data_dir}" in line:
            parts = line.split("{data_dir}")
            prefix = parts[0]
            new_line = f"{prefix}{{data_dir}}/{csv_name} IGNORE=@"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines) + "\n"


# $INPUT should match the dataset header (CSV columns).
def _rewrite_template_input_from_columns(template: str, columns: List[str]) -> str:
    """
    Rewrite the $INPUT line to match the CSV header exactly.

    Note: This intentionally replaces any reference-specific extras (e.g. OCC, or token placeholders)
    so the generated template corresponds to the user's dataset.
    """
    def _normalize_input_column_name(name: str) -> str:
        cleaned = str(name).strip()
        compact = re.sub(r"[\s_]+", "", cleaned).upper()
        if compact in {"ID", "#ID", "CID"}:
            return "ID"
        if compact == "DATE":
            return "DATE=DROP"
        return cleaned

    normalized_cols = [_normalize_input_column_name(str(c)) for c in columns if str(c).strip()]
    if not normalized_cols:
        return template

    lines = template.splitlines()
    new_lines: List[str] = []
    replaced = False
    for line in lines:
        if not replaced and line.lstrip().startswith("$INPUT"):
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}$INPUT       " + " ".join(normalized_cols))
            replaced = True
        else:
            new_lines.append(line)
    return "\n".join(new_lines) + "\n"


def _append_report_tables(template: str, user_choices: "UserChoices") -> str:
    """Append darwinreporter-friendly $TABLE blocks based on selected covariates."""
    selected_covs: set[str] = set()
    for cov_list in user_choices.search_space.covariates_by_param.values():
        selected_covs.update(cov_list or [])

    continuous_covs: List[str] = []
    categorical_covs: List[str] = []
    for cov_name, info in user_choices.covariates.items():
        if cov_name not in selected_covs:
            continue
        if info.cov_type == "categorical":
            categorical_covs.append(cov_name)
        else:
            continuous_covs.append(cov_name)

    lines = [line for line in template.splitlines() if not line.lstrip().startswith("$TABLE")]
    while lines and not lines[-1].strip():
        lines.pop()

    table_lines = [
        "$TABLE ID TIME MDV IPRED PRED WRES CWRES NOPRINT FILE=sdtab ONEHEADER",
        "$TABLE ID ETAS(1:LAST) NOPRINT FILE=patab ONEHEADER",
    ]
    if continuous_covs:
        table_lines.append(
            "$TABLE ID "
            + " ".join(continuous_covs)
            + " NOPRINT FILE=cotab ONEHEADER"
        )
    if categorical_covs:
        table_lines.append(
            "$TABLE ID "
            + " ".join(categorical_covs)
            + " NOPRINT FILE=catab ONEHEADER"
        )

    return "\n".join(lines + [""] + table_lines) + "\n"


# When BOV is not selected (or data has no OCC/OCCASION), strip IOV from the
# generated template and tokens so no IOV placeholders or code remain.
IOV_TOKEN_NAMES = frozenset({"IOVV", "IOVCL", "IOVQ2"})


def _strip_occasion_from_input(template: str) -> str:
    """Remove OCC and OCCASION from $INPUT line when data has no occasion column."""
    lines = template.splitlines()
    new_lines: List[str] = []
    for line in lines:
        if line.strip().startswith("$INPUT"):
            # Remove OCC and OCCASION (case-insensitive, whole-word) from the line
            rest = line
            for col in ("OCC", "OCCASION"):
                rest = re.sub(rf"\b{re.escape(col)}\b", "", rest, flags=re.IGNORECASE)
            rest = re.sub(r"  +", " ", rest).strip()
            new_lines.append(rest)
        else:
            new_lines.append(line)
    return "\n".join(new_lines) + "\n"


def _enabled_iov_parameters(search_space: SearchSpace) -> set[str]:
    """Return the BOV/IOV-enabled parameters allowed by the UI state."""
    if not search_space.has_occasion:
        return set()
    return {param for param in (search_space.bov_parameters or []) if param in {"CL", "V"}}


def _strip_disabled_iov_from_template(template: str, enabled_iov_params: set[str]) -> str:
    """
    Remove IOV placeholders and PK terms for parameters that are not selected for
    BOV/IOV in the current search space.
    """
    lines = template.splitlines()
    new_lines: List[str] = []
    for line in lines:
        cleaned = line
        if "V" not in enabled_iov_params:
            cleaned = re.sub(r"\{IOVV\[\d+\]\}", "", cleaned)
            cleaned = re.sub(r"\s*\*EXP\(IOVV\)", "", cleaned, flags=re.IGNORECASE)
        if "CL" not in enabled_iov_params:
            cleaned = re.sub(r"\{IOVCL\[\d+\]\}", "", cleaned)
            cleaned = re.sub(r"\s*\*EXP\(IOVCL\)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).rstrip()
        new_lines.append(cleaned)
    return "\n".join(new_lines) + "\n"


def _strip_disabled_iov_from_token_text(text: str, enabled_iov_params: set[str]) -> str:
    """Remove IOV-related placeholders, code, and comments from a token string."""
    lines = text.splitlines()
    new_lines: List[str] = []
    for line in lines:
        cleaned = line
        if "V" not in enabled_iov_params:
            cleaned = re.sub(r"\{IOVV\[\d+\]\}", "", cleaned)
            cleaned = re.sub(r"\{IOVQ2\[\d+\]\}", "", cleaned)
            cleaned = re.sub(r"\s*\*EXP\(IOVV\)", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*\*EXP\(IOVQ2\)", "", cleaned, flags=re.IGNORECASE)
            if re.search(r"\bIOV(?:V|Q2[A-Z0-9]*)\b|IOV on (?:V|Q)\b", cleaned, re.IGNORECASE):
                continue
        if "CL" not in enabled_iov_params:
            cleaned = re.sub(r"\{IOVCL\[\d+\]\}", "", cleaned)
            cleaned = re.sub(r"\s*\*EXP\(IOVCL\)", "", cleaned, flags=re.IGNORECASE)
            if re.search(r"\bIOVCL[A-Z0-9]*\b|IOV on CL\b", cleaned, re.IGNORECASE):
                continue
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).rstrip()
        new_lines.append(cleaned)
    return "\n".join(new_lines)


def _strip_disabled_iov_from_tokens(
    tokens: Dict[str, Any],
    enabled_iov_params: set[str],
) -> Dict[str, Any]:
    """Remove disabled IOV token keys and clean nested ADVAN references."""
    remove_keys = set()
    if "CL" not in enabled_iov_params:
        remove_keys.add("IOVCL")
    if "V" not in enabled_iov_params:
        remove_keys.update({"IOVV", "IOVQ2"})

    cleaned_tokens: Dict[str, Any] = {}
    for name, value in tokens.items():
        if name in remove_keys:
            continue
        if isinstance(value, list):
            cleaned_value: List[Any] = []
            for option in value:
                if isinstance(option, list):
                    cleaned_option: List[Any] = []
                    for item in option:
                        if isinstance(item, str):
                            cleaned_option.append(
                                _strip_disabled_iov_from_token_text(item, enabled_iov_params)
                            )
                        else:
                            cleaned_option.append(item)
                    cleaned_value.append(cleaned_option)
                elif isinstance(option, str):
                    cleaned_value.append(
                        _strip_disabled_iov_from_token_text(option, enabled_iov_params)
                    )
                else:
                    cleaned_value.append(option)
            cleaned_tokens[name] = cleaned_value
        else:
            cleaned_tokens[name] = value
    return cleaned_tokens


def _filter_residual_error_tokens(
    tokens: Dict[str, Any],
    selected_models: List[str],
) -> Dict[str, Any]:
    """
    Restrict the RESERR token to the residual error models selected in the UI.
    Reference tokens are ordered as additive, proportional, combined.
    """
    if "RESERR" not in tokens or not isinstance(tokens["RESERR"], list):
        return tokens

    selected = [m for m in selected_models if m in RESIDUAL_ERROR_MODEL_ORDER]
    if not selected:
        return tokens

    model_to_index = {name: idx for idx, name in enumerate(RESIDUAL_ERROR_MODEL_ORDER)}
    filtered_reserr = [
        tokens["RESERR"][model_to_index[name]]
        for name in selected
        if model_to_index[name] < len(tokens["RESERR"])
    ]
    out = dict(tokens)
    out["RESERR"] = filtered_reserr
    return out


def _param_from_covariate_token(token_name: str) -> Optional[str]:
    """
    Map a token name like CL~WT or V4~AGE to the UI parameter key.
    Returns None if not a covariate-effect token.
    """
    if "~" not in token_name:
        return None
    param_part = token_name.split("~", 1)[0].upper()
    if param_part in MODEL_PARAMETER_ORDER:
        return param_part
    return None


def _filter_tokens_by_search_space(
    tokens: Dict[str, Any],
    search_space: SearchSpace,
    available_covariate_names: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Keep only tokens that are consistent with the UI search space AND the actual dataset:
    - Covariate tokens (PARAM~COV): keep only if (1) COV is in covariates_by_param for that param,
      AND (2) the dataset has a column that maps to COV (available_covariate_names).
    - IOV*: keep only if has_occasion and the corresponding param is in bov_parameters.
    - Structural tokens (ADVAN, RESERR, D1LAG, KAETA, ETAD1LAG, BSV*, etc.): keep.
    """
    covariates_by_param = search_space.covariates_by_param
    has_occasion = search_space.has_occasion
    bov_parameters = set(search_space.bov_parameters or [])
    available = available_covariate_names or set()

    filtered: Dict[str, Any] = {}
    for name, value in tokens.items():
        if not isinstance(value, list):
            continue
        # IOV: only if has_occasion and param in bov_parameters
        if name == "IOVCL":
            if has_occasion and "CL" in bov_parameters:
                filtered[name] = value
            continue
        if name == "IOVV":
            if has_occasion and "V" in bov_parameters:
                filtered[name] = value
            continue
        if name == "IOVQ2":
            if (
                has_occasion
                and "V" in bov_parameters
                and (2 in search_space.compartments or 3 in search_space.compartments)
            ):
                filtered[name] = value
            continue
        # Covariate-effect tokens (X~Y): keep only if BOTH user selected AND dataset has the column
        if "~" in name:
            param = _param_from_covariate_token(name)
            cov = name.split("~", 1)[1]
            param_list = covariates_by_param.get(param, [])
            aliases = COVARIATE_TOKEN_ALIASES.get(cov, [cov])
            user_selected = cov in param_list or any(a in param_list for a in aliases)
            in_dataset = cov in available or any(a in available for a in aliases)
            if param and user_selected and in_dataset:
                filtered[name] = value
            continue
        # Structural / other: keep (ADVAN, RESERR, D1LAG, KAETA, ETAD1LAG, BSV*, etc.)
        filtered[name] = value

    return filtered


def _strip_template_blocks_for_missing_columns(
    template: str, data_columns: set
) -> str:
    """
    Remove hardcoded blocks that reference columns not in the dataset.
    E.g. zipra template has IF (race.EQ.2) THEN ... ENDIF; remove when race not in data.
    """
    data_cols_upper = {str(c).strip().upper() for c in data_columns}
    lines = template.splitlines()
    out: List[str] = []
    skip_until_endif = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Skip block that starts with IF (col.EQ.x) when col not in data
        if re.match(r"IF\s*\(\s*(\w+)\s*\.EQ\.\s*\d+\s*\)\s*THEN", stripped, re.IGNORECASE):
            match = re.search(r"IF\s*\(\s*(\w+)\s*\.EQ\.", stripped, re.IGNORECASE)
            if match:
                col = match.group(1).upper()
                if col not in data_cols_upper:
                    skip_until_endif = True
                    i += 1
                    continue
        if skip_until_endif:
            if re.match(r"ENDIF", stripped, re.IGNORECASE):
                skip_until_endif = False
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def _remove_unused_placeholders(template: str, tokens: Dict[str, Any]) -> str:
    """
    Remove or replace every {TOKEN[i]} in template where TOKEN is not in tokens,
    so the template only references tokens that exist in the filtered token set.
    Replaced with empty string to avoid broken NONMEM.
    """
    token_keys = set(k for k in tokens if isinstance(tokens.get(k), list))
    token_keys.add("data_dir")  # reserved, never remove

    def repl(match: re.Match) -> str:
        full = match.group(0)
        inner = match.group(1)
        base = inner.split("[", 1)[0] if "[" in inner else inner
        if base in token_keys:
            return full
        return ""

    pattern = re.compile(r"\{([^}]+)\}")
    return pattern.sub(repl, template)


def _inject_covariate_centering(
    template: str,
    user_choices: "UserChoices",
) -> str:
    """
    Ensure $PK contains centering code for each continuous covariate used.
    Defines C{COV}ONE and/or C{COV}ZERO variables based on CovariateInfo.
    """
    lines = template.splitlines()
    out: List[str] = []
    in_pk = False

    # Build map from upper-cased covariate name to (one_var, zero_var, info)
    cov_centering: Dict[str, Tuple[str, str, CovariateInfo]] = {}
    categorical_indicators: Dict[str, List[Tuple[str, Any]]] = {}
    for cov_list in user_choices.search_space.covariates_by_param.values():
        for cov in cov_list:
            info = user_choices.covariates.get(cov)
            if not info:
                continue
            if info.cov_type == "categorical":
                categorical_indicators[cov] = [
                    (_categorical_indicator_name(cov, index), level)
                    for index, level in enumerate(
                        _non_reference_levels(info),
                        start=1,
                    )
                ]
                continue
            if info.median is None:
                continue
            cov_upper = cov.upper()
            cov_one_var = f"{cov_upper}ONE"
            cov_zero_var = f"{cov_upper}ZERO"
            cov_centering[cov_upper] = (cov_one_var, cov_zero_var, info)

    emitted = False

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("$PK"):
            in_pk = True
            out.append(line)
            # Immediately after $PK, emit centering code once.
            if not emitted and (cov_centering or categorical_indicators):
                for cov_upper, (cov_one_var, cov_zero_var, info) in cov_centering.items():
                    raw = cov_upper
                    if info.center_on_one:
                        out.append(f"  {cov_one_var} = {raw}/{info.median}")
                    if info.center_on_zero:
                        out.append(f"  {cov_zero_var} = ({raw} - {info.median})")
                for cov, indicators in categorical_indicators.items():
                    raw = cov.upper()
                    for indicator, level in indicators:
                        out.append(f"  {indicator} = 0")
                        out.append(
                            f"  IF ({raw}.EQ.{_format_nonmem_level(level)}) "
                            f"{indicator} = 1"
                        )
                emitted = True
            continue
        out.append(line)
    return "\n".join(out) + "\n"


# Reference templates (GA, etc.) embed legacy covariate placeholders like {CL~WT[1]}.
# We remove these before injecting placeholders from the UI so each covariate appears once.
_REFERENCE_COV_PLACEHOLDER_RE = re.compile(
    r"\s*\{[A-Za-z0-9_]+~[A-Za-z0-9_]+\[\d+\]\}",
    re.IGNORECASE,
)


def _strip_reference_covariate_placeholders(line: str) -> str:
    """Remove {PARAM~COV[n]} tokens from a line; collapse extra spaces."""
    cleaned = _REFERENCE_COV_PLACEHOLDER_RE.sub("", line)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.rstrip()


def _dedupe_pk_cov_placeholders_on_line(line: str) -> str:
    """
    On a single line, keep the first {PARAM~COV[1]} for each (PARAM,COV) pair;
    drop identical duplicates (case-insensitive), e.g. two {CL~AGE[1]}.
    """
    pattern = re.compile(r"\{[A-Za-z0-9_]+~[A-Za-z0-9_]+\[1\]\}", re.IGNORECASE)
    seen_upper: set[str] = set()
    parts: List[str] = []
    pos = 0
    for m in pattern.finditer(line):
        parts.append(line[pos : m.start()])
        tok = m.group(0)
        key = tok.upper()
        if key not in seen_upper:
            seen_upper.add(key)
            parts.append(tok)
        pos = m.end()
    parts.append(line[pos:])
    cleaned = "".join(parts)
    cleaned = re.sub(r"  +", " ", cleaned)
    return cleaned.rstrip()


def _inject_covariate_effect_placeholders(
    template: str,
    user_choices: "UserChoices",
) -> str:
    """
    Insert {PARAM~COV[i]} placeholders around TVCL/TVV/TVKA assignments so that
    covariate-effect tokens can act on CL, V, and KA.
    Only covariates selected in the UI are kept; reference {PARAM~COV[i]} on those
    lines are stripped first to avoid duplication.
    """
    lines = template.splitlines()
    out: List[str] = []

    cl_covs = user_choices.search_space.covariates_by_param.get("CL", [])
    v_covs = user_choices.search_space.covariates_by_param.get("V", [])
    ka_covs = user_choices.search_space.covariates_by_param.get("KA", [])
    theta_placeholder_params = ("V", "CL", "KA")

    def _suffix_for_covs(param: str, covs: List[str]) -> str:
        parts: List[str] = []
        # Preserve order, drop duplicate covariate names (e.g. from session state).
        for cov in dict.fromkeys(covs):
            info = user_choices.covariates.get(cov)
            if not info:
                continue
            # Any covariate that has a PARAM~COV token (continuous or categorical)
            # will use the same placeholder pattern here.
            parts.append(f" {{{param}~{cov}[1]}}")
        return "".join(parts)

    def _theta_lines_for_covs(param: str, covs: List[str]) -> List[str]:
        lines: List[str] = []
        for cov in dict.fromkeys(covs):
            info = user_choices.covariates.get(cov)
            if not info:
                continue
            lines.append(f"{{{param}~{cov}[2]}}")
        return lines

    cl_suffix = _suffix_for_covs("CL", cl_covs)
    v_suffix = _suffix_for_covs("V", v_covs)
    ka_suffix = _suffix_for_covs("KA", ka_covs)
    theta_lines_by_param = {
        "V": _theta_lines_for_covs("V", v_covs),
        "CL": _theta_lines_for_covs("CL", cl_covs),
        "KA": _theta_lines_for_covs("KA", ka_covs),
    }

    # Only inject on lines that *assign to* TVCL/TVV/TVKA (LHS), not lines like
    # CL=TVCL*EXP(...) or KA=TVKA which mention TVCL/TVKA on the RHS.
    _tvcl_lhs = re.compile(r"^\s*TVCL\s*=", re.IGNORECASE)
    _tvv_lhs = re.compile(
        r"^\s*TVV2\s*=|^\s*TVV\s*=",
        re.IGNORECASE,
    )
    _tvka_lhs = re.compile(r"^\s*TVKA\s*=", re.IGNORECASE)
    _theta_placeholder_line = re.compile(r"^\s*\{([A-Za-z0-9_]+)~([A-Za-z0-9_]+)\[2\]\}\s*$")
    central_theta_params = set(theta_placeholder_params)
    theta_buffer: List[str] | None = None

    def _flush_theta_buffer() -> None:
        nonlocal theta_buffer
        if theta_buffer is None:
            return
        cleaned_theta_lines: List[str] = []
        for theta_line in theta_buffer:
            match = _theta_placeholder_line.match(theta_line.strip())
            if match and match.group(1).upper() in central_theta_params:
                continue
            cleaned_theta_lines.append(theta_line)
        for param in theta_placeholder_params:
            cleaned_theta_lines.extend(theta_lines_by_param[param])
        out.extend(cleaned_theta_lines)
        theta_buffer = None

    for line in lines:
        new_line = line
        stripped = line.strip()
        if cl_suffix and _tvcl_lhs.match(stripped):
            base = _strip_reference_covariate_placeholders(line)
            new_line = _inject_suffix_after_theta(base, cl_suffix)
            new_line = _dedupe_pk_cov_placeholders_on_line(new_line)
        elif v_suffix and _tvv_lhs.match(stripped):
            base = _strip_reference_covariate_placeholders(line)
            new_line = _inject_suffix_after_theta(base, v_suffix)
            new_line = _dedupe_pk_cov_placeholders_on_line(new_line)
        elif ka_suffix and _tvka_lhs.match(stripped):
            base = _strip_reference_covariate_placeholders(line)
            new_line = _inject_suffix_after_theta(base, ka_suffix)
            new_line = _dedupe_pk_cov_placeholders_on_line(new_line)

        if stripped.startswith("$THETA"):
            theta_buffer = [new_line]
            continue
        if theta_buffer is not None:
            if stripped.startswith("$OMEGA"):
                _flush_theta_buffer()
                out.append(new_line)
            else:
                theta_buffer.append(new_line)
            continue
        out.append(new_line)

    _flush_theta_buffer()

    return "\n".join(out) + "\n"


def _inject_advan_covariate_placeholders(
    tokens: Dict[str, Any],
    user_choices: "UserChoices",
) -> Dict[str, Any]:
    """
    Rewrite ADVAN token strings so peripheral parameters use the covariates selected
    in the UI rather than only the legacy WT placeholders from the reference.
    """
    advan_options = tokens.get("ADVAN")
    if not isinstance(advan_options, list):
        return tokens

    peripheral_params = peripheral_parameter_names(
        user_choices.search_space.compartments,
        user_choices.administration,
    )
    covariates_by_param = user_choices.search_space.covariates_by_param
    assignment_patterns = {
        param: re.compile(rf"^\s*{re.escape(param)}\s*=", re.IGNORECASE)
        for param in peripheral_params
    }

    rewritten_options: List[Any] = []
    for option in advan_options:
        if not isinstance(option, list):
            rewritten_options.append(option)
            continue
        updated_option = list(option)

        if len(updated_option) > 2 and isinstance(updated_option[2], str):
            pk_lines: List[str] = []
            for line in updated_option[2].splitlines():
                base = _strip_reference_covariate_placeholders(line)
                stripped = line.strip()
                for param in peripheral_params:
                    covs = covariates_by_param.get(param, [])
                    if covs and assignment_patterns[param].match(stripped):
                        suffix = _placeholder_suffix_for_covs(param, covs, 1, user_choices)
                        base = _inject_suffix_after_theta(base, suffix)
                        base = _dedupe_pk_cov_placeholders_on_line(base)
                        break
                pk_lines.append(base)
            updated_option[2] = "\n".join(pk_lines)

        if len(updated_option) > 3 and isinstance(updated_option[3], str):
            theta_text = _strip_reference_covariate_placeholders(updated_option[3])
            extra_lines: List[str] = []
            for param in peripheral_params:
                extra = _placeholder_suffix_for_covs(
                    param,
                    covariates_by_param.get(param, []),
                    2,
                    user_choices,
                )
                if extra:
                    extra_lines.append(extra.lstrip("\n"))
            if extra_lines:
                theta_text = theta_text.rstrip() + "\n" + "\n".join(extra_lines)
            updated_option[3] = theta_text

        rewritten_options.append(updated_option)

    out = dict(tokens)
    out["ADVAN"] = rewritten_options
    return out


def _rewrite_advan_bsv_placeholders(
    tokens: Dict[str, Any],
    search_space: SearchSpace,
    administration: str,
) -> Dict[str, Any]:
    """
    Remove peripheral BSV placeholders from ADVAN options unless the user selected
    BSV on the corresponding parameter for that compartment structure.
    """
    advan_options = tokens.get("ADVAN")
    if not isinstance(advan_options, list):
        return tokens

    if administration == ADMIN_INFUSION:
        advan_param_map = {
            3: {"Q": "BSVQ2", "V2": "BSVV2"},
            11: {"Q2": "BSVQ2", "Q3": "BSVQ3", "V2": "BSVV2", "V3": "BSVV3"},
        }
    else:
        advan_param_map = {
            4: {"Q": "BSVQ2", "V3": "BSVV3"},
            12: {"Q2": "BSVQ2", "Q3": "BSVQ3", "V3": "BSVV3", "V4": "BSVV4"},
        }
    selected_params = set(search_space.bsv_parameters or [])

    rewritten_options: List[Any] = []
    for option in advan_options:
        if not isinstance(option, list) or not option or not isinstance(option[0], str):
            rewritten_options.append(option)
            continue
        match = re.match(r"\s*ADVAN(\d+)\b", option[0], re.IGNORECASE)
        if not match:
            rewritten_options.append(option)
            continue

        advan_number = int(match.group(1))
        param_tokens = advan_param_map.get(advan_number, {})
        disabled_tokens = {
            token_name
            for param, token_name in param_tokens.items()
            if param not in selected_params
        }
        updated_option = list(option)

        # Normalize stale first-order reference text so ADVAN4 uses V3/BSVV3 and
        # ADVAN12 uses V3/V4 with the matching BSV tokens/comments.
        if administration != ADMIN_INFUSION:
            replacement_map = {}
            if advan_number == 4:
                replacement_map = {
                    "{BSVV2[1]}": "{BSVV3[1]}",
                    "{BSVV2[2]}": "{BSVV3[2]}",
                    "ETA(BSVV2)": "ETA(BSVV3)",
                    "BSV on V2": "BSV on V3",
                }
            elif advan_number == 12:
                replacement_map = {
                    "{BSVV2[1]}": "{BSVV3[1]}",
                    "{BSVV2[2]}": "{BSVV3[2]}",
                    "ETA(BSVV2)": "ETA(BSVV3)",
                    "BSV on V2": "BSV on V3",
                }
            if replacement_map:
                for idx, item in enumerate(updated_option):
                    if not isinstance(item, str):
                        continue
                    normalized_item = item
                    for old, new in replacement_map.items():
                        normalized_item = normalized_item.replace(old, new)
                    updated_option[idx] = normalized_item

        if disabled_tokens:
            for idx, item in enumerate(updated_option):
                if not isinstance(item, str):
                    continue
                new_lines: List[str] = []
                for line in item.splitlines():
                    drop_line = False
                    cleaned = line
                    for token_name in disabled_tokens:
                        if re.search(rf"\{{{re.escape(token_name)}\[\d+\]\}}", cleaned):
                            if re.search(rf"\{{{re.escape(token_name)}\[2\]\}}", cleaned):
                                drop_line = True
                                break
                            cleaned = re.sub(
                                rf"\{{{re.escape(token_name)}\[\d+\]\}}",
                                "",
                                cleaned,
                            )
                    if drop_line:
                        continue
                    cleaned = re.sub(r"  +", " ", cleaned).rstrip()
                    new_lines.append(cleaned)
                updated_option[idx] = "\n".join(new_lines)
        rewritten_options.append(updated_option)

    out = dict(tokens)
    out["ADVAN"] = rewritten_options
    return out


def extract_template_token_names(template: str) -> List[str]:
    """Extract token names from template: {TOKEN_NAME} or {TOKEN_NAME[i]} -> TOKEN_NAME."""
    # Match {NAME} or {NAME[1]}, {NAME[2]}, etc.
    pattern = re.compile(r"\{([^}[\] ]+)(?:\[\d+\])?\}")
    names = pattern.findall(template)
    return sorted(dict.fromkeys(names))


def validate_token_template_consistency(
    template: str, tokens: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    SKILL: Every {TOKEN_NAME} or {TOKEN_NAME[i]} in template MUST have key in tokens.json.
    Reserved placeholders (e.g. data_dir) are excluded—they are resolved at runtime.
    Returns (ok, list of error messages).
    """
    token_keys = set(k for k in tokens if isinstance(tokens.get(k), list))
    errors: List[str] = []
    placeholder_pattern = re.compile(r"\{([^}[\] ]+)(?:\[(\d+)\])?\}")
    pending = [
        (name, int(index) if index else 1)
        for name, index in placeholder_pattern.findall(template)
        if name not in RESERVED_TEMPLATE_PLACEHOLDERS
    ]
    reachable: set[str] = set()
    checked: set[Tuple[str, int]] = set()
    while pending:
        name, index = pending.pop()
        if (name, index) in checked:
            continue
        checked.add((name, index))
        if name not in token_keys:
            errors.append(f"Placeholder {name}[{index}] has no token entry.")
            continue
        reachable.add(name)
        alternatives = tokens[name]
        for alternative_index, alternative in enumerate(alternatives):
            fragments = (
                [alternative]
                if isinstance(alternative, str)
                else list(alternative) if isinstance(alternative, list) else []
            )
            if index > len(fragments):
                errors.append(
                    f"Token {name} option {alternative_index} does not provide "
                    f"fragment [{index}]."
                )
                continue
            for fragment in fragments:
                if not isinstance(fragment, str):
                    continue
                pending.extend(
                    (nested_name, int(nested_index) if nested_index else 1)
                    for nested_name, nested_index in placeholder_pattern.findall(fragment)
                    if nested_name not in RESERVED_TEMPLATE_PLACEHOLDERS
                )
    unused = token_keys - reachable
    if unused:
        errors.append(f"Token entries not reachable from template: {sorted(unused)}.")
    return (len(errors) == 0, errors)


def validate_compartments(compartments: List[int]) -> Tuple[bool, List[str]]:
    """SKILL: Compartment only 1, 2, or 3. Returns (ok, list of error messages)."""
    errors: List[str] = []
    for c in compartments:
        if c not in ALLOWED_COMPARTMENTS:
            errors.append(
                f"Compartment {c} not allowed. Only {ALLOWED_COMPARTMENTS} are valid."
            )
    return (len(errors) == 0, errors)


def validate_categorical_configuration(user_choices: UserChoices) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    patterns_by_param = user_choices.covariate_patterns_by_param or {}
    additive_bounds = user_choices.categorical_additive_bounds_by_param or {}
    selected_pairs = {
        (param, covariate)
        for param, covariates in user_choices.search_space.covariates_by_param.items()
        for covariate in covariates
    }
    indicator_owners: Dict[str, str] = {}
    for param, covariate in sorted(selected_pairs):
        info = user_choices.covariates.get(covariate)
        if info is None or info.cov_type != "categorical":
            continue
        levels = list(info.levels or [])
        if len(levels) < 2:
            errors.append(f"{covariate}: categorical covariates need at least two levels.")
            continue
        if any(
            not isinstance(level, (int, float)) or isinstance(level, bool)
            for level in levels
        ):
            errors.append(
                f"{covariate}: categorical levels must be numeric NONMEM values."
            )
        if info.reference_level not in levels:
            errors.append(f"{covariate}: reference level is not an observed level.")
        if not info.within_subject_consistent and not info.allow_time_varying:
            errors.append(
                f"{covariate}: values change within subject; confirm time-varying use."
            )
        non_reference = _non_reference_levels(info)
        for index, _ in enumerate(non_reference, start=1):
            indicator = _categorical_indicator_name(covariate, index)
            if (
                indicator in indicator_owners
                and indicator_owners[indicator] != covariate
            ):
                errors.append(f"Duplicate categorical indicator name: {indicator}")
            indicator_owners[indicator] = covariate
        selected_patterns = patterns_by_param.get(param, {}).get(
            covariate,
            ["none", "proportional", "exponential"],
        )
        invalid_patterns = set(selected_patterns) - {
            "none",
            "additive",
            "proportional",
            "exponential",
        }
        if invalid_patterns:
            errors.append(
                f"{param}~{covariate}: unsupported patterns {sorted(invalid_patterns)}."
            )
        if "additive" in selected_patterns:
            bounds = additive_bounds.get(param, {}).get(covariate)
            if not isinstance(bounds, list) or len(bounds) != 3:
                errors.append(
                    f"{param}~{covariate}: additive requires [lower, initial, upper]."
                )
            elif (
                not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bounds)
                or not bounds[0] < bounds[1] < bounds[2]
            ):
                errors.append(
                    f"{param}~{covariate}: additive bounds must satisfy lower < initial < upper."
                )
    return (not errors, errors)


def save_csv_to_dir(data: bytes, filename: str, base_output_dir: Path) -> Path:
    """Write uploaded CSV bytes to base_output_dir/filename."""
    base_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_output_dir / filename
    csv_path.write_bytes(data)
    return csv_path


def generate_py_darwin_files(
    csv_data: bytes,
    csv_filename: str,
    user_choices: UserChoices,
    base_output_dir: Path,
) -> Dict[str, List[str]]:
    # SKILL: enforce compartment limit 1, 2, or 3
    comp_ok, comp_errors = validate_compartments(user_choices.search_space.compartments)
    if not comp_ok:
        raise ValueError("Invalid compartments: " + "; ".join(comp_errors))

    # Use the uploaded CSV header to adapt $INPUT.
    df = load_csv_from_bytes(csv_data)
    categorical_ok, categorical_errors = validate_categorical_configuration(
        user_choices
    )
    if not categorical_ok:
        raise ValueError(
            "Invalid categorical covariate configuration: "
            + "; ".join(categorical_errors)
        )
    primary_algo = user_choices.algorithms[0] if user_choices.algorithms else "GA"
    patterns = load_covariate_patterns()

    original_csv_name = Path(csv_filename).name or "data.csv"
    safe_csv_name = re.sub(r"[^A-Za-z0-9._-]+", "_", original_csv_name).strip("._")
    if not safe_csv_name:
        safe_csv_name = "data.csv"
    elif not safe_csv_name.lower().endswith(".csv"):
        safe_csv_name += ".csv"
    output_dir = base_output_dir / f"{Path(safe_csv_name).stem}_{primary_algo.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_csv_to_dir(csv_data, safe_csv_name, output_dir)

    generated_files: Dict[str, List[str]] = {"template": [], "tokens": [], "options": []}

    # Write template (route-specific: first_order vs zero_order; not algorithm-specific)
    route_dir = route_reference_dir(user_choices.administration)
    template = load_reference_template_for_administration(user_choices.administration)
    updated = _update_template_data_path(template, csv_path.name)
    updated = _rewrite_template_input_from_columns(updated, list(df.columns))
    has_occasion = user_choices.search_space.has_occasion
    enabled_iov_params = _enabled_iov_parameters(user_choices.search_space)

    if not has_occasion:
        updated = _strip_occasion_from_input(updated)
    updated = _strip_disabled_iov_from_template(updated, enabled_iov_params)

    # Ensure centering code and covariate placeholders are present in $PK
    updated = _inject_covariate_centering(updated, user_choices)
    updated = _inject_covariate_effect_placeholders(updated, user_choices)
    updated = _append_report_tables(updated, user_choices)

    # Load reference tokens for structural pieces; covariate-effect tokens will
    # be built from a generic pattern library and the user's covariate choices.
    tokens_path = route_dir / "tokens.json"
    if not tokens_path.is_file():
        raise FileNotFoundError(f"Missing tokens for route {route_dir.name}: {tokens_path}")
    with tokens_path.open() as f:
        tokens_dict = json.load(f)
    tokens_dict = _filter_tokens_by_search_space(tokens_dict, user_choices.search_space)
    tokens_dict = _filter_residual_error_tokens(
        tokens_dict,
        user_choices.search_space.residual_error_models,
    )

    # Drop any existing covariate-effect tokens (PARAM~COV) from reference.
    tokens_dict = {
        name: value for name, value in tokens_dict.items() if "~" not in name
    }
    _filter_advan_token_by_compartments(
        tokens_dict,
        user_choices.administration,
        user_choices.search_space.compartments,
    )
    _prune_peripheral_tokens_by_compartments(
        tokens_dict,
        user_choices.administration,
        user_choices.search_space.compartments,
    )
    _prune_peripheral_bsv_tokens(
        tokens_dict,
        user_choices.search_space,
        user_choices.administration,
    )
    tokens_dict = _strip_disabled_iov_from_tokens(tokens_dict, enabled_iov_params)

    # ALAG is only offered for first-order absorption when the user asks to test lag time.
    if (
        user_choices.administration != ADMIN_FIRST_ORDER_ORAL
        or not user_choices.test_tlag
    ):
        tokens_dict.pop("ALAG", None)
    tokens_dict = _rewrite_advan_bsv_placeholders(
        tokens_dict,
        user_choices.search_space,
        user_choices.administration,
    )
    tokens_dict = _inject_advan_covariate_placeholders(tokens_dict, user_choices)
    use_effect_metadata = _uses_effect_metadata(
        user_choices.algorithms,
        user_choices.effect_limit,
    )

    if use_effect_metadata:
        tokens_dict = {
            name: _normalize_token_effect_metadata(name, value) if isinstance(value, list) else value
            for name, value in tokens_dict.items()
        }

    # Build covariate-effect tokens for continuous covariates based on pattern library
    cov_patterns_by_param = user_choices.covariate_patterns_by_param or {}
    covariates_by_param = user_choices.search_space.covariates_by_param

    for param in available_model_parameters(
        user_choices.search_space.compartments,
        user_choices.administration,
    ):
        for cov in covariates_by_param.get(param, []):
            info = user_choices.covariates.get(cov)
            if not info:
                continue

            token_name = f"{param}~{cov}"
            token_options: List[List[Any]] = []
            # Always include a "no effect" option.
            if use_effect_metadata:
                token_options.append(_option_with_effect_count("", "", 0))
            else:
                token_options.append(_plain_option("", ""))

            cov_upper = cov.upper()

            if info.cov_type == "continuous":
                pats = cov_patterns_by_param.get(param, {}).get(
                    cov, ["none", "power", "exponential"]
                )

                cov_one_var = f"{cov_upper}ONE"
                cov_zero_var = f"{cov_upper}ZERO"

                if "power" in pats:
                    base = patterns["continuous"]["power"]
                    pk_fragment = base["pk_fragment"].format(
                        PARAM=param,
                        COV=cov_upper,
                        COV_ONE=cov_one_var,
                        COV_ZERO=cov_zero_var,
                    )
                    theta_init = _format_generated_theta_init(
                        pk_fragment,
                        base["theta_init"],
                        "POWER",
                    )
                    if use_effect_metadata:
                        token_options.append(_option_with_effect_count(pk_fragment, theta_init, 1))
                    else:
                        token_options.append(_plain_option(pk_fragment, theta_init))

                if "exponential" in pats:
                    base = patterns["continuous"]["exponential"]
                    pk_fragment = base["pk_fragment"].format(
                        PARAM=param,
                        COV=cov_upper,
                        COV_ONE=cov_one_var,
                        COV_ZERO=cov_zero_var,
                    )
                    theta_init = _format_generated_theta_init(
                        pk_fragment,
                        base["theta_init"],
                        "EXPONENTIAL",
                    )
                    if use_effect_metadata:
                        token_options.append(_option_with_effect_count(pk_fragment, theta_init, 1))
                    else:
                        token_options.append(_plain_option(pk_fragment, theta_init))

            elif info.cov_type == "categorical":
                selected_relationships = cov_patterns_by_param.get(param, {}).get(
                    cov,
                    [
                        "none",
                        *patterns["categorical"]["default_relationships"],
                    ],
                )
                additive_bounds = (
                    user_choices.categorical_additive_bounds_by_param or {}
                ).get(param, {}).get(cov)
                for relationship in (
                    "proportional",
                    "exponential",
                    "additive",
                ):
                    if relationship not in selected_relationships:
                        continue
                    pk_fragment, theta_init = _categorical_token_option(
                        param,
                        cov,
                        info,
                        relationship,
                        patterns,
                        additive_bounds,
                    )
                    if use_effect_metadata:
                        token_options.append(
                            _option_with_effect_count(
                                pk_fragment,
                                theta_init,
                                1,
                            )
                        )
                    else:
                        token_options.append(
                            _plain_option(pk_fragment, theta_init)
                        )

            # Only register tokens that have at least the no-effect and one effect option
            if len(token_options) > 1:
                tokens_dict[token_name] = token_options

    # Remove template placeholders for tokens we dropped (e.g. unused covariates)
    updated = _remove_unused_placeholders(updated, tokens_dict)

    template_path = output_dir / "template.txt"
    template_path.write_text(updated)
    generated_files["template"].append(str(template_path))

    out_tokens = output_dir / "tokens.json"
    with out_tokens.open("w") as f:
        json.dump(tokens_dict, f, indent=4)

    # SKILL: verify every {TOKEN} in template has a key in tokens.json
    tok_ok, tok_errors = validate_token_template_consistency(updated, tokens_dict)
    if not tok_ok:
        raise ValueError("Token-template consistency: " + "; ".join(tok_errors))

    generated_files["tokens"].append(str(out_tokens))

    # Write options
    for algo_name in user_choices.algorithms:
        options = load_reference_options(algo_name)
        options = _apply_run_directories(options, user_choices)
        options = _apply_effect_limit_option(options, algo_name, user_choices.effect_limit)
        if user_choices.nmfe_path:
            options["nmfe_path"] = user_choices.nmfe_path.strip()
        if len(user_choices.algorithms) == 1:
            out_path = output_dir / "options.json"
        else:
            out_path = output_dir / f"options_{algo_name.upper()}.json"
        with out_path.open("w") as f:
            json.dump(options, f, indent=4)
        generated_files["options"].append(str(out_path))

    meta = {
        "search_space": asdict(user_choices.search_space),
        "covariates": {
            name: asdict(info) for name, info in user_choices.covariates.items()
        },
        "algorithms": user_choices.algorithms,
        "covariate_patterns_by_param": user_choices.covariate_patterns_by_param or {},
        "categorical_additive_bounds_by_param": (
            user_choices.categorical_additive_bounds_by_param or {}
        ),
        "administration": user_choices.administration,
        "test_tlag": user_choices.test_tlag,
        "effect_limit": user_choices.effect_limit,
        "nmfe_path": user_choices.nmfe_path,
        "route_reference": route_dir.name,
        "csv_path": str(csv_path),
    }
    meta_path = output_dir / "agent_metadata.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return generated_files


def covariate_info_to_dict(c: CovariateInfo) -> Dict[str, Any]:
    return asdict(c)


def covariate_info_from_dict(d: Dict[str, Any]) -> CovariateInfo:
    levels = [_python_scalar(value) for value in d.get("levels", [])]
    reference_level = d.get("reference_level")
    if reference_level is None and levels:
        counts = d.get("level_counts") or {}
        reference_level = sorted(
            levels,
            key=lambda level: (-int(counts.get(str(level), 0)), _level_sort_key(level)),
        )[0]
    return CovariateInfo(
        name=d["name"],
        is_numeric=d["is_numeric"],
        median=d.get("median"),
        center_on_one=d.get("center_on_one", True),
        center_on_zero=d.get("center_on_zero", True),
        cov_type=d.get("cov_type", "continuous"),
        is_binary=(
            d.get("is_binary")
            if d.get("is_binary") is not None
            else len(levels) == 2
        ),
        levels=levels,
        level_counts={
            str(key): int(value)
            for key, value in (d.get("level_counts") or {}).items()
        },
        reference_level=reference_level,
        within_subject_consistent=bool(d.get("within_subject_consistent", True)),
        allow_time_varying=bool(d.get("allow_time_varying", False)),
    )


def search_space_to_dict(s: SearchSpace) -> Dict[str, Any]:
    return asdict(s)


def search_space_from_dict(d: Dict[str, Any]) -> SearchSpace:
    return SearchSpace(
        compartments=list(d["compartments"]),
        covariates_by_param=dict(d["covariates_by_param"]),
        residual_error_models=list(d["residual_error_models"]),
        bsv_parameters=list(d["bsv_parameters"]),
        bov_parameters=list(d["bov_parameters"]),
        has_occasion=bool(d["has_occasion"]),
    )


def user_choices_from_dict(d: Dict[str, Any]) -> UserChoices:
    covariates = {
        name: covariate_info_from_dict(c)
        for name, c in d["covariates"].items()
    }
    patterns = d.get("covariate_patterns_by_param") or {}
    adm = d.get("administration", ADMIN_FIRST_ORDER_ORAL)
    if adm not in ADMINISTRATION_OPTIONS:
        adm = ADMIN_FIRST_ORDER_ORAL
    test_tlag = bool(d.get("test_tlag", False))
    if adm != ADMIN_FIRST_ORDER_ORAL:
        test_tlag = False
    return UserChoices(
        search_space=search_space_from_dict(d["search_space"]),
        covariates=covariates,
        algorithms=list(d["algorithms"]),
        covariate_patterns_by_param=patterns,
        categorical_additive_bounds_by_param=d.get(
            "categorical_additive_bounds_by_param"
        ) or {},
        administration=adm,
        test_tlag=test_tlag,
        effect_limit=d.get("effect_limit"),
        working_dir=d.get("working_dir"),
        output_dir=d.get("output_dir"),
        temp_dir=d.get("temp_dir"),
        nmfe_path=d.get("nmfe_path"),
    )
