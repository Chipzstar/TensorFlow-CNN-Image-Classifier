
%%sh

# The name of our algorithm
ecr_repo=sagemaker-tf-2-serving # PREVIOUS REPO: sagemaker-tf-serving
docker_image=sagemaker-tf-2-serving # PREVIOUS IMAGE: sagemaker-tf-serving

cd container

# chmod a+x container/serve.py

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-eu-west-2}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${ecr_repo}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${ecr_repo}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${ecr_repo}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email --registry-ids=763104351884)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# docker build -t ${docker_image} .
# docker tag ${docker_image}:latest ${fullname}

# Pull docker image for tensorflow 2.0 GPU inference 

image_url="763104351884.dkr.ecr.${region}.amazonaws.com/tensorflow-inference:2.0.0-gpu-py36-cu100-ubuntu18.04"

docker pull ${image_url}
docker tag ${image_url} ${fullname}
docker push ${fullname}