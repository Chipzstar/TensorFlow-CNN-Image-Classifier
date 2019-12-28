FROM tensorflow/serving:latest

# RUN pip install sagemaker-containers

# Installing NGINX, used to reverse proxy the predictions from SageMaker to TF Serving
RUN apt-get update && apt-get install -y --no-install-recommends nginx git

# Copy our model folder to the container 
# NB: Tensorflow serving requires you manually assign version numbering to models e.g. model_path/1/
# see below links: 

# https://stackoverflow.com/questions/45544928/tensorflow-serving-no-versions-of-servable-model-found-under-base-path
# https://github.com/aws/sagemaker-python-sdk/issues/599
COPY pornilarity_model /opt/ml/model/export/Servo/1/

# Copy NGINX configuration to the container
COPY nginx.conf /opt/ml/code/nginx.conf

# Copies the hosting code inside the container
# COPY serve.py /opt/ml/code/serve.py

# Defines serve.py as script entrypoint
# ENV SAGEMAKER_PROGRAM serve.py

# starts NGINX and TF serving pointing to our model
ENTRYPOINT service nginx start | tensorflow_model_server --port=8500 \
 --rest_api_port=8501 \
 --model_name=pornilarity_model \
 --model_base_path=/opt/ml/model/export/Servo/