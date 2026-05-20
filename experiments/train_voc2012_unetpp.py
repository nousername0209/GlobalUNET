from segmentation_benchmark.cli import parse_experiment_args


if __name__ == "__main__":
    config = parse_experiment_args(dataset="voc2012", model="unetpp")
    from segmentation_benchmark.train import run_experiment

    run_experiment(config)
