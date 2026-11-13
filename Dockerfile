FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY relay ./relay
RUN pip install --no-cache-dir .
COPY config.yaml ./
ENV RELAY_CONFIG=/app/config.yaml
EXPOSE 8080
CMD ["python", "-m", "relay.app"]
