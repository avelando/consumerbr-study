from dataclasses import dataclass
from typing import Callable

from consumerbr_resolution.clean import clean_modeling_base
from consumerbr_resolution.convert import convert_corpus_to_parquet
from consumerbr_resolution.download import download_corpus
from consumerbr_resolution.extract import extract_corpus
from consumerbr_resolution.modeling_base import build_modeling_base
from consumerbr_resolution.features import build_feature_base


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