if [ -z "$MLFLOW_TRACKING_URI" ]; then
  echo "MLFLOW_TRACKING_URI is not set. Please set it before running this script."
  exit 1
fi
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri $MLFLOW_TRACKING_URI --artifacts-destination $MLFLOW_TRACKING_URI
