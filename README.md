# QML-CyberDefence-COMP4900

All the details about the code and various scenarios described below can be found in the report: https://github.com/TimXia7/QML-CyberDefence-COMP4900/blob/main/Final%20Report.pdf

# Virtual Env Setup
1. In Command Prompt, navigate to the root directory of the project folder, and run these commands, one at a time:

```
pip install virtualenv
python -m venv venv
source venv/bin/activate or .\venv\Scripts\Activate.ps1 for windows 
pip install -r requirements.txt
```

# Running The Various Scenarios For The Simple Track
1. For the Simple Track scenario where Train 1 and 2 positions reset: python src/single_track_simulations/main_reset.py
2. For the Simple Track scenario where Train 2 position does not reset: python src/single_track_simulations/main.py
3.  For the Simple Track control scenario where Train 1 and 2 positions reset: python src/single_track_simulations/main_control_reset.py
4.   For the Simple Track control scenario Where Train 1 and 2 positions do not reset: python src/single_track_simulations/main_control.py

# Running The Various Scenarios For The Intermediate Track
1. For the Intermediate Track scenario where Train 1 and 2 positions reset: python src/intermediate_track_simulations/main_intermediate_track_reset.py
2. For the Intermediate Track scenario where Train 2 position does not reset: python src/intermediate_track_simulations/main_intermediate_track.py
3.  For the Intermediate Track control scenario where Train 1 and 2 positions reset: python src/intermediate_track_simulations/main_intermediate_track_reset_control.py
4.   For the Intermediate Track control scenario Where Train 1 and 2 positions do not reset: python src/intermediate_track_simulations/main_intermediate_track_control.py
    
