"""
Entry point that runs the full GoEmoX pipeline:
"""

from download_data import main as download_data_main
from train import main as train_main
from predict import main as predict_main
from analyze_results import main as analyze_results_main
from visualize_results import main as visualize_results_main


def main(
    run_download: bool = True,
    run_train: bool = True,
    run_predict: bool = True,
    run_analyze: bool = True,
    run_visualize: bool = True,
) -> None:
    """Run the full pipeline with optional steps."""
    if run_download:
        print("[PIPELINE] Step 1/5: Downloading datasets...")
        download_data_main()
    else:
        print("[PIPELINE] Step 1/5: Skipping download step.")

    if run_train:
        print("[PIPELINE] Step 2/5: Training model...")
        train_main()
    else:
        print("[PIPELINE] Step 2/5: Skipping training step.")

    if run_predict:
        print("[PIPELINE] Step 3/5: Running predictions...")
        predict_main()
    else:
        print("[PIPELINE] Step 3/5: Skipping prediction step.")

    if run_analyze:
        print("[PIPELINE] Step 4/5: Analyzing results...")
        analyze_results_main()
    else:
        print("[PIPELINE] Step 4/5: Skipping analysis step.")

    if run_visualize:
        print("[PIPELINE] Step 5/5: Generating visualizations...")
        visualize_results_main()
    else:
        print("[PIPELINE] Step 5/5: Skipping visualization step.")

    print("[PIPELINE] All requested steps completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the GoEmoX emotion classification pipeline end-to-end."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading datasets (if they already exist in data/).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training (if a trained model already exists in results/).",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip running predictions.",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip analysis of training results.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip generating visualizations.",
    )

    args = parser.parse_args()

    main(
        run_download=not args.skip_download,
        run_train=not args.skip_train,
        run_predict=not args.skip_predict,
        run_analyze=not args.skip_analyze,
        run_visualize=not args.skip_visualize,
    )
