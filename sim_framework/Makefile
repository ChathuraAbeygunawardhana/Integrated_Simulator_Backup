# Top-level orchestration. Each target wraps a script under scripts/.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: install check_env lint smoke full_test matrix timing_closure report all clean

install:
	./scripts/install_deps.sh

check_env:
	./scripts/check_env.sh

lint:
	$(MAKE) -C $(ROOT)/rtl -f Makefile.lint lint

smoke:
	PYTHONPATH=$(ROOT) $(ROOT)/.venv/bin/python -m pytest tb/integration/test_single_dnn_smoke.py tb/integration/test_tf_golden_conv2d.py -v

full_test:
	PYTHONPATH=$(ROOT) $(ROOT)/.venv/bin/python -m pytest tb/ -v

matrix:
	PYTHONPATH=$(ROOT) $(ROOT)/.venv/bin/python scripts/iterate_until_pass.py

timing_closure:
	@if command -v vivado >/dev/null 2>&1; then \
	    vivado -mode batch -source timing/vivado/synth_single_dnn.tcl -tclargs $(ROOT); \
	    vivado -mode batch -source timing/vivado/synth_multi_dnn.tcl  -tclargs $(ROOT); \
	else \
	    echo '[skip] vivado not installed; cycle-budget tests still ran'; \
	fi

report:
	PYTHONPATH=$(ROOT) $(ROOT)/.venv/bin/python scripts/gen_report.py

all: check_env lint full_test matrix timing_closure report

clean:
	rm -rf build/* docs/verification_matrix.md docs/final_report.md
