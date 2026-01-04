import nemo_run as run

if __name__ == "__main__":
    training_job = run.Script(
        inline="""
# This string will get saved to a sh file and executed with bash
# Run any preprocessing commands

# Run the training command
python mock_train.py --device-number 8 --nnodes 1

# Run any post processing commands
"""
    )

    # Run it locally
    executor = run.LocalExecutor(ntasks_per_node=8)

    with run.Experiment("nemo_2.0_training_experiment", log_level="INFO") as exp:
        exp.add(training_job, executor=executor, tail_logs=True, name="training")
        # Add more jobs as needed

        # Run the experiment
        exp.run(detach=False)
