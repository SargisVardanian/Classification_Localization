import subprocess

script_path = 'C:\\Users\\User\\PycharmProjects\\Classification_Localization\\Object_Detection_EfficientNetB0.py'
n = 5

for _ in range(n):
    subprocess.call(['python', script_path])
