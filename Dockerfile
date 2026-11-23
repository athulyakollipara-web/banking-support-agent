# One image for local Docker and AWS Lambda (via the AWS Lambda Web Adapter extension).
# The intent model is trained at build time, so the image is self-contained and every deploy is reproducible
# from the pinned datasets.
FROM public.ecr.aws/docker/library/python:3.12-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/healthz PYTHONUNBUFFERED=1 \
    BSA_HOME=/opt/bsa BSA_MODEL_DIR=/opt/bsa/model

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bsa ./bsa
RUN pip install --no-cache-dir . && bsa train && rm -rf /opt/bsa/data && useradd --create-home agent
USER agent

EXPOSE 8080
CMD ["uvicorn", "bsa.api:app", "--host", "0.0.0.0", "--port", "8080"]
