import argparse

from consumerbr_resolution.pipeline import STAGES, run_all, run_stage


def show_menu():
    while True:
        print()
        print("ConsumerBR Resolution Prediction")
        print()

        for stage_number, (stage_name, _) in enumerate(STAGES, start=1):
            print(f"{stage_number}. {stage_name}")

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
                run_stage(stage_number)
                continue

        print("Invalid option.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        nargs="?",
        choices=["menu", "download", "all"],
        default="menu",
    )

    args = parser.parse_args()

    if args.command == "menu":
        show_menu()
    elif args.command == "download":
        run_stage(1)
    elif args.command == "all":
        run_all()