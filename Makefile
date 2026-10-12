.PHONY: install test serve generate e2e clean
install:   ; pip install -e ".[llm,dev]"
test:      ; pytest
serve:     ; lighthouse serve --reload
generate:  ; lighthouse generate --n 5000 --db data/bulk.db
e2e:       ; python tests/e2e_browser.py http://127.0.0.1:8765 shots/
clean:     ; rm -rf data/*.db* shots .pytest_cache
