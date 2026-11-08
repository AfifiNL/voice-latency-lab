# Everything here runs offline and costs nothing, except `record-ttft`, which calls a live model
# on purpose. `make check` is exactly what CI runs.
UV  ?= uv
RUN := $(UV) run --all-extras
CLI := $(RUN) voice-lab
TTFT := fixtures/live-ttft.jsonl
# Live TTFT recording: needs Gemini credentials (Vertex AI: ADC + GOOGLE_CLOUD_PROJECT), costs
# money, and is capped at LIVE_CALLS calls per thinking level.
LIVE_CALLS ?= 24
LIVE ?= VOICE_LAB_GEMINI_VERTEX=true VOICE_LAB_GEMINI_LOCATION=eu \
        VOICE_LAB_GEMINI_MODEL=gemini-3.6-flash VOICE_LAB_MAX_LIVE_CALLS=$(LIVE_CALLS)

.PHONY: help install lint fmt typecheck test bench bench-baseline ttft-check detector-check \
        record-ttft demo check

help:            ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

install:         ## create .venv with all extras and dev tools (locked)
	$(UV) sync --all-extras

lint:            ## ruff lint + format check
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

fmt:             ## apply ruff formatting and safe fixes
	$(RUN) ruff format src tests scripts
	$(RUN) ruff check --fix src tests scripts

typecheck:       ## mypy (strict)
	$(RUN) mypy

test:            ## pytest (offline)
	$(RUN) pytest

bench:           ## 1,000-turn bench (real policy + responder, recorded TTFT) vs the baseline
	$(CLI) simulate --turns 1000 --seed 7 --ttft-from $(TTFT) \
	  --ttft-config gemini:gemini-3.6-flash:thinking-minimal --out reports/bench.jsonl > /dev/null
	$(CLI) compare bench/baseline.jsonl reports/bench.jsonl

bench-baseline:  ## re-record bench/baseline.jsonl (review the diff before committing)
	$(CLI) simulate --turns 1000 --seed 7 --ttft-from $(TTFT) \
	  --ttft-config gemini:gemini-3.6-flash:thinking-minimal --out bench/baseline.jsonl > /dev/null

ttft-check:      ## the recorded live TTFT run still matches its committed summary
	$(CLI) ttft-report $(TTFT) --review fixtures/live-ttft-review.json \
	  --check fixtures/live-ttft.summary.json

detector-check:  ## detector + request accuracy on the labelled corpora, against frozen floors
	$(CLI) eval-detector --floors evals/detector_floors.json

record-ttft:     ## LIVE: re-record TTFT at thinking minimal and low (costs money)
	$(LIVE) $(CLI) record-ttft --thinking minimal --out $(TTFT)
	$(LIVE) $(CLI) record-ttft --thinking low --out $(TTFT)

demo:            ## offline demo: language policy, speculative turn, simulated latency
	$(CLI) demo

check: lint typecheck test bench ttft-check detector-check  ## everything CI runs
