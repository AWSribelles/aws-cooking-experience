import base64
import json
import logging
import random
import os

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class Claude3Wrapper:
    """Encapsulates Claude 3 model invocations using the Amazon Bedrock Runtime client."""

    def __init__(self, client=None):
        """
        :param client: A low-level client representing Amazon Bedrock Runtime.
                       Describes the API operations for running inference using Bedrock models.
                       Default: None
        """
        self.client = client

    def invoke_claude_3_with_text(self, prompt):
        """
        Invokes Anthropic Claude 3 Haiku to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude 3 to complete.
        :return: Inference response from the model.
        """

        # Initialize the Amazon Bedrock runtime client
        client = self.client or boto3.client(
            service_name="bedrock-runtime", region_name="us-west-2"
        )

        # Invoke Claude 3 with the text prompt
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"

        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                    }
                ),
            )

            # Process and print the response
            result = json.loads(response.get("body").read())
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            output_list = result.get("content", [])

            print("Invocation details:")
            print(f"- The input length is {input_tokens} tokens.")
            print(f"- The output length is {output_tokens} tokens.")

            print(f"- The model returned {len(output_list)} response(s):")
            for output in output_list:
                print(output["text"])

            return result

        except ClientError as err:
            logger.error(
                "Couldn't invoke Claude 3 Haiku. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise

    # snippet-end:[python.example_code.bedrock-runtime.InvokeAnthropicClaude3Text]

    # snippet-start:[python.example_code.bedrock-runtime.InvokeAnthropicClaude3Multimodal]

    def invoke_claude_3_multimodal(self, prompt, s3_bucket, s3_key):
        """
        Invokes Anthropic Claude 3 Haiku to run a multimodal inference using the input
        provided in the request body.

        :param prompt:    The prompt that you want Claude 3 to use.
        :param s3_bucket: The name of the S3 bucket where the image is stored.
        :param s3_key:    The key (path) of the image in the S3 bucket.
        :return: Inference response from the model.
        """

        # Initialize the Amazon Bedrock runtime client
        client = self.client or boto3.client(
            service_name="bedrock-runtime", region_name="us-west-2"
        )

        # Initialize the S3 client
        s3_client = boto3.client("s3")

        # Fetch the image data from S3
        image_data = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)["Body"].read()
        base64_image_data = base64.b64encode(image_data).decode("utf8")

        # Invoke the model with the prompt and the encoded image
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image_data,
                            },
                        },
                    ],
                }
            ],
        }

        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
            )

            # Process and print the response
            result = json.loads(response.get("body").read())
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            output_list = result.get("content", [])

            print("Invocation details:")
            print(f"- The input length is {input_tokens} tokens.")
            print(f"- The output length is {output_tokens} tokens.")

            print(f"- The model returned {len(output_list)} response(s):")
            for output in output_list:
                print(output["text"])

            return result
        except ClientError as err:
            logger.error(
                "Couldn't invoke Claude 3 Haiku. Here's why: %s: %s",
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise


def lambda_handler(event, context):
    # Retrieve the prompt and S3 details from the event
    print(event)
    body = json.loads(event['body'])
    prompt = body["prompt"]
    s3_bucket = os.environ.get('INPUT_BUCKET')
    s3_key = body["s3_key"]

    client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    dynamodb = boto3.client('dynamodb')
    wrapper = Claude3Wrapper(client)

    # Invoke Claude 3 with a multimodal prompt (text and image)
    print(f"Invoking Claude 3 Haiku with '{prompt}' and '{s3_bucket}/{s3_key}'...")
    response_content = wrapper.invoke_claude_3_multimodal(prompt, s3_bucket, s3_key)

    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(response_content)
    }
    '''
    text = response_content["content"][0]["text"]
    id = str(random.randint(100, 999))
    item = {
    'ID': {'S': id},
    'S3Bucket': {'S': s3_bucket},
    'S3Key': {'S': s3_key},
    'PromptOut': {'S': text}
    }
    table_name = os.environ.get('DYNAMODB_TABLE_NAME')
    dynamodb.put_item(
        TableName=table_name,
        Item=item
    )
    response = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": "Successful"
    }
    '''
    return response
    #return response