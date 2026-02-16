PYTHON ?= python3

.PHONY: test-all test-cpu-fast test-mps test-cuda

test-all:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

test-cpu-fast:
	$(PYTHON) -m unittest -v \
		tests.test_data_cpu \
		tests.test_kv_cache_cpu \
		tests.test_models_cpu \
		tests.test_optim \
		tests.test_tokenizer \
		tests.test_train_cli_cpu \
		tests.test_train_integration_cpu

test-mps:
	$(PYTHON) -m unittest -v \
		tests.test_mps_smoke \
		tests.test_mps_gated

test-cuda:
	$(PYTHON) -m unittest -v \
		tests.test_cuda_gated
