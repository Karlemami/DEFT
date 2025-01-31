VENV := venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
SRC_DIR = src

all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

venv: $(VENV)/bin/activate

run: venv
	@$(PYTHON) $(SRC_DIR)/main.py

fmt:
	@black $(SRC_DIR)

PAPER_PATH = paper

paper:
	@make open -C $(PAPER_PATH) > /dev/null

clean:
	@echo "Removing venv..."
	@rm -rf $(VENV)
	@echo "Removing all pycache..."
	@find . -type f -name '*.pyc' -delete
	@echo "Project cleaned!"

.PHONY: all venv run fmt paper clean
