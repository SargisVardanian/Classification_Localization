import subprocess

script_path = 'Object_Detection_EfficientNetB0.py'
n = 5

subprocess.call(['python', script_path] * n)
