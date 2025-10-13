Setup for Windows:
==================


- Launch the terminal application

- Download and install uv

`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

- Restart terminal

- Install Python 3.12
`uv python install 3.12`

- Create virtual environment
`uv venv --python 3.12`

- Activate virtual environment
`.venv\Scripts\activate`

- Install requirements
`uv pip install numpy pandas matplotlib seaborn jupyterlab scikit-learn jax statsmodels`

- Launch jupyterlab
`jupyter lab`

