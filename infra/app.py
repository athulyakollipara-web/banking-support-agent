#!/usr/bin/env python3
import aws_cdk as cdk

from stack import SupportAgentStack

app = cdk.App()
SupportAgentStack(app, "BankingSupportAgent")
app.synth()
