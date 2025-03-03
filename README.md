# QML-CyberDefence-COMP4900

# Virtual Env Setup
1. In Command Prompt, navigate to the root directory of the project folder, and run these commands, one at a time:

```
pip install virtualenv
python -m venv venv
source venv/bin/activate or .\venv\Scripts\Activate.ps1 for windows
pip install -r requirements.txt
```

# Running Files (from the base of the project, folder "QML-CyberDefence-COMP4900" )
1. To run the main sequence/simulation, enter:

    python .\src\QML.py

As of 2024-03-02, it is not complete, and just outputs a test value.


2. To run all automated tests, use:

    python -m unittest discover unit_tests

To run a specific test, use  (replace, "test_variational_circuit.py"  with the unit test file):

    python .\unit_tests\test_variational_circuit.py