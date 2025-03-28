#!/bin/bash

WORKER_COUNT=${1:-3}
echo "Scaling Celery workers to $WORKER_COUNT..."

celery_mod=${CELERY_MODULE:-3}

# force wait for redis to start a small startup delay
sleep 5
# use wait-for-it to ensure redis is active on port 6379
/wait-for-it.sh redis:6379 --timeout=5 -- echo "Redis is up for Celery to start..."

for i in $(seq 1 $WORKER_COUNT); do
  WORKER_NAME="worker$i"

  echo "Starting worker: $WORKER_NAME"
  celery -A ${celery_mod}.worker.celery_app worker -n $WORKER_NAME -Q general_tasks --loglevel=info &

  sleep 1
done


# feed workers start up
FEED_WORKERS=${2:-3}
echo "Scaling Feed workers to $FEED_WORKERS..."

for i in $(seq 1 $FEED_WORKERS); do
  WORKER_NAME="feed_worker$i"

  echo "Starting worker: $WORKER_NAME"
  celery -A ${celery_mod}.feed_worker.feed_worker_app worker -n $WORKER_NAME -Q feed_tasks --loglevel=info &

  sleep 0.5
done


MODEL_WORKERS=${3:-3}
echo "Scaling Model workers to $MODEL_WORKERS..."

for i in $(seq 1 $MODEL_WORKERS); do
  WORKER_NAME="model_worker$i"

  echo "Starting worker: $WORKER_NAME"
  # since so many tasks, --loglevel=warning to reduce log spam
  celery -A ${celery_mod}.model_worker.model_worker_app worker -n $WORKER_NAME -Q model_tasks --loglevel=warning &

  sleep 0.5
done

STREAM_WORKERS=${4:-3}
echo "Scaling Stream workers to $STREAM_WORKERS..."

for i in $(seq 1 $STREAM_WORKERS); do
  WORKER_NAME="stream_worker$i"

  echo "Starting worker: $WORKER_NAME"
  # since so many tasks, --loglevel=warning to reduce log spam
  celery -A ${celery_mod}.stream_worker.stream_worker_app worker -n $WORKER_NAME -Q stream_tasks --loglevel=warning &

  sleep 0.5
done

wait