# Blaze Compiler – Phase 0 Makefile

.PHONY: build test clean

build:
	@echo "Build not implemented yet."

test:
	@python3.14 test.py

clean:
	@rm -rf __pycache__ *.pyc .cache target out *.out tmp_*
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
