#!/bin/bash
ENV=$1
echo "Deploying app to $ENV environment..."

if [ "$ENV" == "staging" ]; then
  kubectl apply -f k8s/staging-deployment.yaml
elif [ "$ENV" == "production" ]; then
  kubectl apply -f k8s/prod-deployment.yaml
else
  echo "Unknown environment"
  exit 1
fi
echo "Deployment to $ENV completed."