# Empowerment Lander

Codebase built on work from https://github.com/rddy/deepassist. 

To set up required packages, use `environment.yml` with conda or `requirements.txt` with pip.
If using conda, activate the environment using `conda activate lander`. Then install OpenAI Baselines
using `pip install -e baselines`.

To only train simulated pilots, run `train_sim_pilots.py`.
To train simulated pilots and assistive copilots, run `train_copilot.py` with the argument `--empowerment` for adjusting empowerment coefficient (0 for no empowerment).
This will also automatically run the cross evaluation tests. 

To replay policies, run `run_rollouts.py` and replace the copilots with the saved policies.

To play the game (human trials), run the script `human_exp.sh`. This runs through the 
scripts `run_scripts/human_solo.py`, which launches the vanilla game for getting accustomed to the controls,
then `python run_scripts/human_emp.py --empowerment`, which launches the game with a copilot using empowerment, 
then `python run_scripts/human_emp.py`, launches the game with a copilot without empowerment. Results will 
be saved under a empowerment_lander/data folder. 

