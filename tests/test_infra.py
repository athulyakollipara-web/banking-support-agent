"""CDK assertions (skipped if aws-cdk-lib is not installed)."""
import sys
from pathlib import Path

import pytest

cdk = pytest.importorskip("aws_cdk")
from aws_cdk.assertions import Match, Template  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra"))
from stack import SupportAgentStack  # noqa: E402


@pytest.fixture(scope="module")
def template():
    return Template.from_stack(SupportAgentStack(cdk.App(), "Test"))


def test_api_requires_customer_token(template):
    template.has_resource_properties("AWS::ApiGatewayV2::Authorizer", {"AuthorizerType": "JWT"})
    template.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey": "ANY /v1/{proxy+}", "AuthorizationType": "JWT"})


def test_sessions_expire(template):
    template.has_resource_properties("AWS::DynamoDB::Table", {"TimeToLiveSpecification": {"AttributeName": "ttl", "Enabled": True}})


def test_lambda_in_aws_mode_with_bedrock_only_for_anthropic_models(template):
    template.has_resource_properties("AWS::Lambda::Function", {
        "PackageType": "Image", "Environment": {"Variables": Match.object_like({"BSA_MODE": "aws"})}})
    template.has_resource_properties("AWS::IAM::Policy", {"PolicyDocument": {"Statement": Match.array_with([
        Match.object_like({"Action": ["bedrock:InvokeModel", "bedrock:Converse"]})])}})
