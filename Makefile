.PHONY: install test test-cpu bench clean

install:
	pip install -e ".[dev]"

# Run the suite normally: kernel tests use the GPU, or skip if none is present.
test:
	pytest -v

# Validate the actual Triton kernels on CPU via the interpreter -- no GPU needed.
# Slower than a real GPU, but exercises kernel logic (indexing, masks, reductions).
test-cpu:
	TRITON_INTERPRET=1 pytest -q

bench:
	python benchmarks/bench_rmsnorm.py --M 8192 --dtype fp16 --out assets/rmsnorm_bench.png

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
