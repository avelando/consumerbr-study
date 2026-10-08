from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Stage:
    command: str
    name: str
    module: str
    function_name: str


STAGES = [
    Stage(
        "download",
        "Download ConsumerBR corpus",
        "data.download",
        "download_corpus",
    ),
    Stage(
        "extract",
        "Extract ConsumerBR corpus",
        "data.extract",
        "extract_corpus",
    ),
    Stage(
        "convert",
        "Convert ConsumerBR CSV to Parquet",
        "data.convert",
        "convert_corpus_to_parquet",
    ),
    Stage(
        "modeling-base",
        "Build binary modeling base",
        "data.modeling_base",
        "build_modeling_base",
    ),
    Stage(
        "clean",
        "Clean modeling base",
        "data.clean",
        "clean_modeling_base",
    ),
    Stage(
        "features",
        "Build pre-response features",
        "data.features",
        "build_feature_base",
    ),
    Stage(
        "dataset-integrity",
        "Validate dataset integrity",
        "data.dataset_integrity",
        "validate_dataset_integrity",
    ),
    Stage(
        "characterize",
        "Characterize experimental dataset",
        "data.characterize",
        "characterize_dataset",
    ),
    Stage(
        "selection-bias",
        "Describe outcome observation",
        "data.selection_bias",
        "analyze_outcome_observation",
    ),
    Stage(
        "temporal-protocol",
        "Build and audit the single temporal split",
        "experiments.temporal_protocol",
        "build_temporal_protocol",
    ),
    Stage("register-execution", "Register and verify execution inputs",
          "experiments.reproducibility", "register_execution"),
    Stage("company-baseline", "Evaluate smoothed historical company rate",
          "baselines", "evaluate_historical_baselines"),
    Stage("export-reports", "Export aggregate reports for publication",
          "experiments.report_export", "export_reports"),
]


def execute_stage(stage_number, stage):
    print()
    print(f"Running stage {stage_number:02d}: {stage.name}")
    print()

    module = import_module(
        f"consumerbr_resolution.{stage.module}"
    )
    function = getattr(module, stage.function_name)
    function()


def run_stage_by_number(stage_number):
    if not 1 <= stage_number <= len(STAGES):
        raise ValueError("Invalid stage number.")

    execute_stage(stage_number, STAGES[stage_number - 1])


def run_stage_by_command(command):
    for stage_number, stage in enumerate(STAGES, start=1):
        if stage.command == command:
            execute_stage(stage_number, stage)
            return

    raise ValueError(f"Unknown stage command: {command}")


def run_all():
    print(f"Running {len(STAGES)} implemented stages in sequence.")
    for stage_number, stage in enumerate(STAGES, start=1):
        execute_stage(stage_number, stage)
    print("Implemented stages completed. TF-IDF and transformer experiments are pending.")