#!/usr/bin/env bash

MODEL_NAME=pornilarity-model-v1

ENDPOINT_CONFIG_NAME=pornilarity-model-v1-config

ENDPOINT_NAME=pornilarity-v1-endpoint

PRODUCTION_VARIANTS="VariantName=Default,ModelName=${MODEL_NAME},"\
"InitialInstanceCount=1,InstanceType=ml.c5.large"

aws sagemaker create-endpoint-config --endpoint-config-name ${ENDPOINT_CONFIG_NAME} \
--production-variants ${PRODUCTION_VARIANTS}

aws sagemaker create-endpoint --endpoint-name ${ENDPOINT_NAME} \
--endpoint-config-name ${ENDPOINT_CONFIG_NAME}