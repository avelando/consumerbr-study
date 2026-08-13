from dataclasses import dataclass
from typing import Callable

from consumerbr_resolution.tfidf import (
    fit_tfidf_vectorizers,
)
from consumerbr_resolution.baselines import (
    evaluate_historical_baselines,
)
from consumerbr_resolution.characterize import (
    characterize_dataset,
)
from consumerbr_resolution.clean import (
    clean_modeling_base,
)
from consumerbr_resolution.convert import (
    convert_corpus_to_parquet,
)
from consumerbr_resolution.download import (
    download_corpus,
)
from consumerbr_resolution.extract import (
    extract_corpus,
)
from consumerbr_resolution.features import (
    build_feature_base,
)
from consumerbr_resolution.modeling_base import (
    build_modeling_base,
)
from consumerbr_resolution.selection_bias import (
    analyze_outcome_observation,
)
from consumerbr_resolution.temporal_protocol import (
    build_temporal_protocol,
)


@dataclass(frozen=True)
class Stage:
    command: str
    name: str
    function: Callable[[], None]


STAGES = [
    Stage(
        command="download",
        name="Download ConsumerBR corpus",
        function=download_corpus,
    ),
    Stage(
        command="extract",
        name="Extract ConsumerBR corpus",
        function=extract_corpus,
    ),
    Stage(
        command="convert",
        name="Convert ConsumerBR CSV to Parquet",
        function=convert_corpus_to_parquet,
    ),
    Stage(
        command="modeling-base",
        name="Build binary modeling base",
        function=build_modeling_base,
    ),
    Stage(
        command="clean",
        name="Clean modeling base",
        function=clean_modeling_base,
    ),
    Stage(
        command="features",
        name="Build deterministic pre-response features",
        function=build_feature_base,
    ),
    Stage(
        command="characterize",
        name="Characterize experimental dataset",
        function=characterize_dataset,
    ),
    Stage(
        command="selection-bias",
        name="Analyze outcome observation patterns",
        function=analyze_outcome_observation,
    ),
    Stage(
        command="temporal-protocol",
        name="Build temporal evaluation protocol",
        function=build_temporal_protocol,
    ),
    Stage(
        command="baselines",
        name="Evaluate historical baselines",
        function=evaluate_historical_baselines,
    ),
    Stage(
        command="tfidf",
        name="Fit fold-specific TF-IDF vectorizers",
        function=fit_tfidf_vectorizers,
    ),
]


def execute_stage(stage_number, stage):
    print()
    print(f"Running stage {stage_number}: {stage.name}")
    print()

    stage.function()


def run_stage_by_number(stage_number):
    if stage_number < 1 or stage_number > len(STAGES):
        raise ValueError("Invalid stage number.")

    stage = STAGES[stage_number - 1]
    execute_stage(stage_number, stage)


def run_stage_by_command(command):
    for stage_number, stage in enumerate(STAGES, start=1):
        if stage.command == command:
            execute_stage(stage_number, stage)
            return

    raise ValueError(f"Unknown stage command: {command}")


def run_all():
    for stage_number, stage in enumerate(STAGES, start=1):
        execute_stage(stage_number, stage)