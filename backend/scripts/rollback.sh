#!/bin/bash
echo "Rollback triggered due to failure or high risk."
# Example rollback command; customize as per deployment environment
kubectl rollout undo deployment/my-app -n production
echo "Rollback completed."