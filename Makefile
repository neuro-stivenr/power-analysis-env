default: env

env:
	python3 -m venv env
	env/bin/pip install --upgrade pip setuptools wheel
	env/bin/pip install -r requirements.txt

.PHONY: clean, lab

clean:
	rm -rf env
	rm -rf __pycache__
	rm -rf .ipynb_checkpoints

lab:
	env/bin/jupyter lab


