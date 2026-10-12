PYTHON          := python
PACKAGE_MANAGER := uv
RUNNER          := $(PACKAGE_MANAGER) run
SRC_DIR         := src
CACHE_DIRS      := __pycache__ $(SRC_DIR)/__pycache__ .mypy_cache
FLAGS           := --warn-return-any \
                        --warn-unused-ignores \
                        --ignore-missing-imports \
                        --disallow-untyped-defs \
                        --check-untyped-defs

.PHONY: all install run debug clean lint lint-strict

all: run

install:
	$(PACKAGE_MANAGER) sync

run:
	$(RUNNER) $(PYTHON) -m $(SRC_DIR)

debug:
	$(RUNNER) $(PYTHON) -m pdb -m $(SRC_DIR)

clean:
	rm -rf $(CACHE_DIRS)

lint:
	$(RUNNER) flake8 $(SRC_DIR)
	$(RUNNER) mypy $(FLAGS) $(SRC_DIR)

lint-strict:
	$(RUNNER) flake8 $(SRC_DIR)
	$(RUNNER) mypy --strict $(SRC_DIR)
