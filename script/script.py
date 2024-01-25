import subprocess

subprocess.run(["pip", "install", "-r", "../requirements.txt"])
subprocess.run(["python", "../collect_dataset.py", "."])
subprocess.run(["python", "../split_dataset.py"])
