import io, re, json, boto3, base64, time, requests
import numpy as np
import PIL
from PIL import Image

JPEG_CONTENT_TYPE = 'image/jpeg'
PNG_CONTENT_TYPE = 'image/png'
IMG_SIZE = 100
CLIENT = boto3.client('logs')
LOG_GROUP_NAME='/aws/sagemaker/Endpoints/pornilarity-v1-endpoint'
LOG_STREAM_NAME = f"[{time.strftime('%Y/%M/%d - %H/%M/%S')}] <inference_logs>"

response = CLIENT.create_log_stream(
    logGroupName=LOG_GROUP_NAME,
    logStreamName=LOG_STREAM_NAME
)

def getToken():
    response = CLIENT.describe_log_streams(
        logGroupName=LOG_GROUP_NAME,
        logStreamNamePrefix=LOG_STREAM_NAME,
    )
    try:
        token = response['logStreams'][0]['uploadSequenceToken']
    except:
        token = None
    return token
       
def get_current_timestamp():
    timestamp = int(time.time()*1000)
    return timestamp
    
    
def get_payload(event):
    data = json.loads(json.dumps(event))
    payload =  data['instances']
    return payload


def process_image_pillow(image_data):
    image_content = image_data.read()
    dataBytesIO = io.BytesIO(image_content)
    pillow_array = np.frombuffer(dataBytesIO.read(), np.uint8) # reads in byte pixel data and converts to numeric numpy array
    image = Image.open(dataBytesIO) # converts pixel data into Pillow image format
    gray = image.convert('L') # grayscales the image (removes RGB channels)
    image = gray.resize((IMG_SIZE, IMG_SIZE)) # resizes image to correct image size
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # reshapes the image into accepted dimensions (4D tensor)
    image = image.astype('float32') # change data type of image to float32
    image = image / 255.0 # Normalisation of RGB pixel values
    return image
    
    
def convert_json_to_array(response_body):
    body = " ".join(response_body.split())
    regex = re.compile("(?![e])[a-z\"\}\{:\s\[\]]*")
    predictions = regex.sub("", body).split(',')
    predictions = np.array(predictions)
    predictions = predictions.astype(np.float64)
    return predictions

    
def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/x-image':
        processed_image = process_image_pillow(data)
        image_tensor = np.asarray(processed_image).tolist()
        payload = {"instances": image_tensor}
        return json.dumps(payload)
    else:
        _return_error(415, 'Unsupported content type "{}"'.format(context.request_content_type or 'Unknown'))


def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))
    response_content_type = context.accept_header
    response_body = response.content.decode("utf-8")
    predictions = convert_json_to_array(response_body)
    predictions = predictions.tolist()
    if not getToken():
        response = CLIENT.put_log_events(
           logGroupName=LOG_GROUP_NAME,
           logStreamName=LOG_STREAM_NAME,
           logEvents=[
               {
                   'timestamp': get_current_timestamp(),
                   'message': response_body
               },
           ]
        )
    # result = dict({'predictions': predictions})
    return json.dumps(predictions), response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))