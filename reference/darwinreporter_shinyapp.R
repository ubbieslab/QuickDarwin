get_arg <- function(args, flag) {
  idx <- match(flag, args)
  if (is.na(idx) || idx >= length(args)) {
    return(NULL)
  }
  args[idx + 1]
}

ensure_dir <- function(path_value, label) {
  if (is.null(path_value) || !nzchar(path_value)) {
    stop(sprintf("Missing required argument: %s", label), call. = FALSE)
  }
  if (!dir.exists(path_value)) {
    stop(sprintf("Directory does not exist for %s: %s", label, path_value), call. = FALSE)
  }
}

args <- commandArgs(trailingOnly = TRUE)
mode <- get_arg(args, "--mode")
project_dir <- get_arg(args, "--project-dir")
working_dir <- get_arg(args, "--working-dir")
output_dir <- get_arg(args, "--output-dir")
key_models_dir <- get_arg(args, "--key-models-dir")
non_dominated_models_dir <- get_arg(args, "--non-dominated-models-dir")

if (!requireNamespace("Certara.DarwinReporter", quietly = TRUE)) {
  stop(
    "Package 'Certara.DarwinReporter' is not installed. Install it in R, then relaunch the reporter.",
    call. = FALSE
  )
}

library(Certara.DarwinReporter)

ensure_dir(project_dir, "--project-dir")
ensure_dir(working_dir, "--working-dir")
ensure_dir(output_dir, "--output-dir")

options(shiny.launch.browser = TRUE)

if (identical(mode, "moga")) {
  ensure_dir(non_dominated_models_dir, "--non-dominated-models-dir")
  ddb <- darwin_data(
    project_dir = project_dir,
    working_dir = working_dir,
    output_dir = output_dir,
    non_dominated_models_dir = non_dominated_models_dir
  )
} else if (identical(mode, "single")) {
  ensure_dir(key_models_dir, "--key-models-dir")
  ddb <- darwin_data(
    project_dir = project_dir,
    working_dir = working_dir,
    output_dir = output_dir
  ) |>
    import_key_models(dir = key_models_dir)
} else {
  stop("Unsupported or missing `--mode`. Use `single` or `moga`.", call. = FALSE)
}

darwinReportUI(ddb)
