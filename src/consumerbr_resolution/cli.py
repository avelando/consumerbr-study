import argparse

from consumerbr_resolution.pipeline import (
    STAGES,
    run_all,
    run_stage_by_command,
    run_stage_by_number,
)


def show_menu():
    while True:
        print()
        print("ConsumerBR Resolution Prediction")
        print()

        for stage_number, stage in enumerate(STAGES, start=1):
            print(f"{stage_number}. {stage.name}")

        print("A. Run all available stages")
        print("0. Exit")
        print()

        option = input("Select an option: ").strip().lower()

        if option == "0":
            return

        if option == "a":
            run_all()
            continue

        if option.isdigit():
            stage_number = int(option)

            if 1 <= stage_number <= len(STAGES):
                run_stage_by_number(stage_number)
                continue

        print("Invalid option.")


def main():
    stage_commands = [stage.command for stage in STAGES]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        nargs="?",
        choices=["menu", "all", *stage_commands],
        default="menu",
    )

    args = parser.parse_args()

    if args.command == "menu":
        show_menu()
    elif args.command == "all":
        run_all()
    else:
        run_stage_by_command(args.command)