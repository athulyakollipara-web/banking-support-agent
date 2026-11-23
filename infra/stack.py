"""AWS CDK stack: the agent as a container Lambda behind an HTTP API.

    Cognito user pool for customers; API Gateway checks their token (JWT authorizer) before the Lambda runs,
        and the agent takes the customer id from the verified token, never from the request body.
    DynamoDB sessions table with TTL (conversations expire after 24 hours).
    Lambda container: FastAPI app + intent model trained at image build; Bedrock access for wording replies only.
"""
from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_cognito as cognito
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class SupportAgentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        sessions = dynamodb.Table(
            self, "Sessions", partition_key=dynamodb.Attribute(name="session_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST, time_to_live_attribute="ttl",
            encryption=dynamodb.TableEncryption.AWS_MANAGED, removal_policy=RemovalPolicy.DESTROY)

        customers = cognito.UserPool(self, "Customers", self_sign_up_enabled=False, sign_in_aliases=cognito.SignInAliases(email=True),
                                     mfa=cognito.Mfa.OPTIONAL, mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
                                     removal_policy=RemovalPolicy.DESTROY)
        app_client = customers.add_client("MobileApp", auth_flows=cognito.AuthFlow(user_srp=True), generate_secret=False)

        fn = lambda_.DockerImageFunction(
            self, "AgentFn", code=lambda_.DockerImageCode.from_image_asset(str(ROOT), file="Dockerfile"),
            memory_size=1536, timeout=Duration.seconds(29),
            log_group=logs.LogGroup(self, "AgentLogs", retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.DESTROY),
            environment={"BSA_MODE": "aws", "SESSIONS_TABLE": sessions.table_name, "BEDROCK_MODEL_ID": MODEL_ID, "BSA_LLM": "on"})
        sessions.grant_read_write_data(fn)
        fn.add_to_role_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel", "bedrock:Converse"],
            resources=["arn:aws:bedrock:*::foundation-model/anthropic.*", f"arn:aws:bedrock:*:{self.account}:inference-profile/*"]))

        auth = authorizers.HttpUserPoolAuthorizer("CustomerAuth", customers, user_pool_clients=[app_client])
        api = apigw.HttpApi(self, "Api", api_name="banking-support-agent")
        integration = integrations.HttpLambdaIntegration("AgentIntegration", fn)
        api.add_routes(path="/v1/{proxy+}", methods=[apigw.HttpMethod.ANY], integration=integration, authorizer=auth)
        api.add_routes(path="/healthz", methods=[apigw.HttpMethod.GET], integration=integration)
        stage = api.default_stage.node.default_child
        stage.default_route_settings = apigw.CfnStage.RouteSettingsProperty(throttling_burst_limit=100, throttling_rate_limit=50)

        CfnOutput(self, "ApiUrl", value=api.api_endpoint)
        CfnOutput(self, "UserPoolId", value=customers.user_pool_id)
        CfnOutput(self, "AppClientId", value=app_client.user_pool_client_id)
        CfnOutput(self, "SessionsTable", value=sessions.table_name)
