from consumerbr_resolution.download import download_corpus


STAGES = [
    ("Download ConsumerBR corpus", download_corpus),
]


def run_stage(stage_number):
    if stage_number < 1 or stage_number > len(STAGES):
        raise ValueError("Invalid stage number.")

    stage_name, stage_function = STAGES[stage_number - 1]

    print()
    print(f"Running stage {stage_number}: {stage_name}")
    print()

    stage_function()


def run_all():
    for stage_number, (stage_name, stage_function) in enumerate(
        STAGES,
        start=1,
    ):
        print()
        print(f"Running stage {stage_number}: {stage_name}")
        print()

        stage_function()