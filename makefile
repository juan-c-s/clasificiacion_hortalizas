.PHONY: run-mlflow
run-mlflow:
	uv run mlflow server --host 127.0.0.1 --port 8080 