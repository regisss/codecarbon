import argparse
from time import sleep

from codecarbon import EmissionsTracker
from codecarbon.emissions_tracker import TaskEmissionsTracker
from datasets import load_dataset
from transformers import pipeline


def inference(model, data, inference_task_name):
    print(f"Running inference {inference_task_name} on {model} {data}")
    sleep(5)
    return "inference_result"


def main():
    # Arguments management
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained model (on the HF Hub or locally).",
    )
    parser.add_argument(
        "--task",
        default="feature-extraction",
        type=str,
        help="Task to perform.",
    )
    parser.add_argument(
        "--dataset_name",
        default="imdb",
        type=str,
        help="Name of a dataset of the HF Hub.",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Number of inference iterations for benchmarking.",
    )
    args = parser.parse_args()

    try:
        tracker = EmissionsTracker(
            project_name="model_inference", measure_power_secs=10
        )
        tracker.start()

        with TaskEmissionsTracker("Load dataset", tracker=tracker):
            dataset = load_dataset(args.dataset_name, split="test")

        with TaskEmissionsTracker("Build model", tracker=tracker):
            model = pipeline(model=args.model_name_or_path, task="image-classification")

        for i, d in enumerate(dataset[: args.n_iterations]["text"]):
            inference_task_name = "Inference" + str(i + 1)
            with TaskEmissionsTracker(inference_task_name, tracker=tracker):
                model(d)
    finally:
        emissions = tracker.stop()

    print(f"Emissions : {1000 * emissions} g CO₂")
    for task_name, task in tracker._tasks.items():
        print(
            f"Emissions : {1000 * task.emissions_data.emissions} g CO₂ for task {task_name}"
        )
        print(
            f"\tEnergy : {1000 * task.emissions_data.cpu_energy} Wh {1000 * task.emissions_data.gpu_energy} Wh RAM{1000 * task.emissions_data.ram_energy}Wh"
        )
        print(
            f"\tPower CPU:{ task.emissions_data.cpu_power:.0f}W GPU:{ task.emissions_data.gpu_power:.0f}W RAM{ task.emissions_data.ram_power:.0f}W"
            + f" during {task.emissions_data.duration} seconds."
        )


if __name__ == "__main__":
    main()
