from dataclasses import dataclass
from typing import Callable

from consumerbr_resolution.risk_calibration_analysis import (
    analyze_risk_and_calibration,
)
from consumerbr_resolution.generalization_analysis import (
    analyze_generalization,
)
from consumerbr_resolution.bertimbau_catboost_fusion import (
    evaluate_bertimbau_catboost_fusion,
)
from consumerbr_resolution.bertimbau_finetuning import (
    evaluate_bertimbau,
)
from consumerbr_resolution.bertimbau_tokens import (
    build_bertimbau_token_cache,
)
from consumerbr_resolution.bertimbau_assets import (
    prepare_bertimbau_assets,
)
from consumerbr_resolution.tabular_catboost import (
    evaluate_catboost,
)
from consumerbr_resolution.tfidf_complement_nb import (
    evaluate_tfidf_complement_nb,
)
from consumerbr_resolution.tfidf_metadata_history_sgd import (
    evaluate_tfidf_metadata_history_sgd,
)
from consumerbr_resolution.company_history import (
    build_company_history_features,
)
from consumerbr_resolution.tfidf_metadata_sgd import (
    evaluate_tfidf_metadata_sgd,
)
from consumerbr_resolution.metadata_sgd import (
    evaluate_metadata_sgd,
)
from consumerbr_resolution.metadata import (
    fit_metadata_preprocessors,
)
from consumerbr_resolution.tfidf_sgd import (
    evaluate_tfidf_sgd,
)
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
    Stage(
        command="tfidf-sgd",
        name="Evaluate TF-IDF with SGD",
        function=evaluate_tfidf_sgd,
    ),
    Stage(
        command="metadata",
        name="Fit fold-specific metadata preprocessors",
        function=fit_metadata_preprocessors,
    ),
    Stage(
        command="metadata-sgd",
        name="Evaluate metadata with SGD",
        function=evaluate_metadata_sgd,
    ),
    Stage(
        command="tfidf-metadata-sgd",
        name="Evaluate TF-IDF with metadata using SGD",
        function=evaluate_tfidf_metadata_sgd,
    ),
    Stage(
        command="company-history",
        name="Build causal company history features",
        function=build_company_history_features,
    ),
    Stage(
        command="tfidf-metadata-history-sgd",
        name="Evaluate TF-IDF metadata and company history with SGD",
        function=evaluate_tfidf_metadata_history_sgd,
    ),
    Stage(
        command="tfidf-complement-nb",
        name="Evaluate TF-IDF with ComplementNB",
        function=evaluate_tfidf_complement_nb,
    ),
    Stage(
        command="catboost",
        name="Evaluate CatBoost tabular model",
        function=evaluate_catboost,
    ),
    Stage(
        command="bertimbau-assets",
        name="Prepare BERTimbau pretrained assets",
        function=prepare_bertimbau_assets,
    ),
    Stage(
        command="bertimbau-tokens",
        name="Build BERTimbau token cache",
        function=build_bertimbau_token_cache,
    ),
    Stage(
        command="bertimbau",
        name="Evaluate BERTimbau temporal fine-tuning",
        function=evaluate_bertimbau,
    ),
    Stage(
        command="bertimbau-catboost-fusion",
        name="Evaluate BERTimbau and CatBoost late fusion",
        function=evaluate_bertimbau_catboost_fusion,
    ),
    Stage(
        command="generalization",
        name="Analyze temporal and company generalization",
        function=analyze_generalization,
    ),
    Stage(
        command="risk-calibration",
        name="Analyze risk ranking and calibration",
        function=analyze_risk_and_calibration,
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