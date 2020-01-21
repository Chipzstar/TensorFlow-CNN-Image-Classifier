/*
Copyright 2017 - 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
    http://aws.amazon.com/apache2.0/
or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
*/


//var environment = process.env.ENV;
var region = process.env.REGION;
//var storageBucketName = process.env.STORAGE_PORNILARITYSTORAGE_BUCKETNAME;
var functionName = process.env.FUNCTION_NAME;

/* Amplify Params - DO NOT EDIT */

const express = require('express');
const bodyParser = require('body-parser');
const awsServerlessExpressMiddleware = require('aws-serverless-express/middleware');
const AWS = require('aws-sdk');


// declare a new express app
const app = express();
const router = express.Router();
router.use(bodyParser.json({limit: '10mb'}));
router.use(awsServerlessExpressMiddleware.eventContext());

// declare lambda object to invoke second lambda function
var lambda = new AWS.Lambda({
	apiVersion: '2015-03-31',
	region: region //change to your region
});

// Enable CORS for all methods
router.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token");
  next();
});

router.post('/classify', async function(req, res) {
  var params = {
    FunctionName: functionName, /* required */
    InvocationType: 'RequestResponse',
    LogType: 'Tail',
    Payload: JSON.stringify(req, null, 2) // pass params
  };
  lambda.invoke(params, function(error, data) {
    if (error) {
      context.done('Error', error);
    }
    if(data.Payload){
      context.succeed(data.Payload);
    }
  });
  res.status(200).json({
    message: "POST request was a success!"
  });
});


// The aws-serverless-express library creates a server and listens on a Unix
// Domain Socket for you, so you can remove the usual call to app.listen.
// app.listen(3000)
app.use('/', router);

// Export your express server so you can import it in the lambda function.
module.exports = app;
