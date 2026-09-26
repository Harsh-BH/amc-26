Master here. The user rebooted the laptop (~12:30) to clear the GPU power throttle. Every local process died; any AWS
job keeps running (check its status with aws sagemaker describe-training-job --profile harsh). Check which outputs on
disk are complete (verify files open, row counts), redo only what is missing, and continue your task. Rules unchanged:
never end your turn to wait; detach long jobs; poll in the foreground with bounded waits.
