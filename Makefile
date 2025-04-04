.PHONY: test
test:
    PYTHONPATH=. pytest --maxfail=1 --disable-warnings -q
