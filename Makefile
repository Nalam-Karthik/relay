.PHONY: install test run loadtest docker

install:
	python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

test:
	.venv/bin/python -m pytest -q

run:
	.venv/bin/python -m relay.app

loadtest:
	.venv/bin/python scripts/loadtest.py --requests 2000 --concurrency 50 --unique 100

docker:
	docker build -t relay-gateway . && docker run --rm -p 8080:8080 -e ANTHROPIC_API_KEY relay-gateway
