# Activating the environment
# cd into the project folder and enter the below command
. .venv/bin.activate

# Then install the project in a dev mode
pip3 install -e .

pip3 list

# Start the dev server: Note that the project entry point is run.py that's why run is twice
flask --app run run
