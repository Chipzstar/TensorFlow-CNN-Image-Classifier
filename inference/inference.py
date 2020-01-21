JPEG_CONTENT_TYPE = 'image/jpeg'
PNG_CONTENT_TYPE = 'image/png'
       
def get_payload(event):
    data = json.loads(json.dumps(event))
    payload =  data['instances']
    return payload


def process_image(image_file):
    np_array = np.fromstring(image_file, np.uint8)
    image_tensor = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2GRAY) # grayscale the image
    image = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) # resize, as our model is expecting images in 32x32.
    image = np.array(image).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # reshapes the image into accepted dimensions (4D tensor)
    image = image.astype('float32') # converts datatype to 32-bit float
    image = image / 255.0 # Normalisation of RGB pixel values
    return image
    
    
def convert_bytes_to_array(response_body):
    body = " ".join(response_body.split())
    regex = re.compile("(?![e])[a-z\"\}\{:\s\[\]]*")
    predictions = regex.sub("", body).split(',')
    predictions = np.array(predictions)
    predictions = predictions.astype(np.float64)
    return predictions
    
    
def get_top5(predictions, labels):
    top_5 = []
    count = 0
    while count < 5:
        max_index = np.argmax(predictions)
        class_label = labels[max_index]
        top_5.append(class_label)
        predictions = np.delete(predictions, max_index)
        count+=1
    return top_5
    
def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/x-image':
        print(f"Base64 image: {data}")
        payload = data.read()
        print(f"Payload: {payload})
        encoded_image = base64.b64encode(payload).decode('utf-8')
        print(f"Encoded Image: {encoded_image}")
        instance = [{"b64": encoded_image}]
        return json.dumps({"instances": instance})
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
    prediction = response.content
    return prediction, response_content_type


def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))