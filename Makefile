.PHONY: install test demo compile run docker-build clean

install:
	pip install -r requirements.txt

install-optional:
	pip install -r requirements-optional.txt

compile:
	python -m compileall -q growthagent examples tests

test: compile
	python -m unittest discover -s tests -v

demo: compile
	python examples/run_growth_agent.py

run:
	uvicorn growthagent.app.main:app --reload

docker-build:
	docker build -f deploy/Dockerfile -t growthagent:latest .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs .pytest_cache
