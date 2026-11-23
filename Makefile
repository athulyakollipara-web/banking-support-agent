.PHONY: setup train test demo chat serve eval docker deploy

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
train:
	bsa train
test:
	python -m pytest
demo:
	bsa demo
chat:
	BSA_DEMO_OTP=123456 bsa chat
serve:
	bsa serve
eval:
	python eval/intent_eval.py && python eval/run_scenarios.py
docker:
	docker build -t banking-support-agent . && docker run --rm -p 8080:8080 banking-support-agent
deploy:        ## needs AWS credentials, Docker and the CDK CLI
	pip install -e ".[infra]" && cd infra && cdk deploy
