PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

UNAME_S := $(shell uname -s)
DEFAULT_HOLO_USE_ACCELERATE := 0
ifeq ($(UNAME_S),Darwin)
DEFAULT_HOLO_USE_ACCELERATE := 1
endif
HOLO_USE_ACCELERATE ?= $(DEFAULT_HOLO_USE_ACCELERATE)

C_MAKE := $(MAKE) -C c HOLO_USE_ACCELERATE=$(HOLO_USE_ACCELERATE) PYTHON=$(PYTHON)
C_ENV := HOLOSTUFF_USE_C=1 HOLOSTUFF_C_STRICT=1

.PHONY: help all deps check-experiment-deps c c-test c-bench c-ci-evidence test test-py benchmark benchmark-c ablations ablations-c stress stress-c experiments experiments-c demos clean

help:
	@printf '%s\n' \
	  'Targets:' \
	  '  make c              build the C kernel shared/static library' \
	  '  make c-test         run C kernel tests' \
	  '  make c-bench        compare Python/NumPy vs C trace, bind_fixed, and VSA program kernels' \
	  '  make c-ci-evidence  compile CI evidence that scalar C trace beats NumPy' \
	  '  make deps           install base + experiment Python dependencies' \
	  '  make test           build C kernel, then run pytest' \
	  '  make benchmark      run benchmark_holographic.py with NumPy core' \
	  '  make benchmark-c    run benchmark_holographic.py with C core' \
	  '  make experiments    run benchmark, ablations, and stress with NumPy core' \
	  '  make experiments-c  run benchmark, ablations, stress, and trace bench with C core' \
	  '  make demos          run the guided tour'

all: c

deps:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-experiments.txt

check-experiment-deps:
	@$(PYTHON) -c "import importlib.util, sys; missing = [m for m in ('matplotlib', 'pandas', 'scipy', 'sklearn') if importlib.util.find_spec(m) is None]; print('Missing experiment dependencies: ' + ', '.join(missing)) if missing else None; sys.exit(1 if missing else 0)" || \
	  (echo "Missing experiment dependencies; run: make deps PYTHON=$(PYTHON)" && exit 1)

c:
	$(C_MAKE) all

c-test:
	$(C_MAKE) test

c-bench:
	$(C_MAKE) bench-compare

c-ci-evidence:
	$(PYTHON) c/benchmarks/ci_evidence.py

test: c
	$(PYTEST)

test-py:
	$(PYTEST)

benchmark: check-experiment-deps
	$(PYTHON) benchmark_holographic.py

benchmark-c: c check-experiment-deps
	$(C_ENV) $(PYTHON) benchmark_holographic.py

ablations:
	$(PYTHON) holographic_ablate.py

ablations-c: c
	$(C_ENV) $(PYTHON) holographic_ablate.py

stress:
	$(PYTHON) stress_holographic.py

stress-c: c
	$(C_ENV) $(PYTHON) stress_holographic.py

experiments: benchmark ablations stress

experiments-c: c benchmark-c ablations-c stress-c c-bench

demos:
	$(PYTHON) tour.py

clean:
	$(C_MAKE) clean
	rm -f benchmark_report.md bench_*.png
