import subprocess

if __name__ == '__main__':

    subprocess.run("python3 ilqr_evaluation.py", shell=True)
    subprocess.run("python3 ddpg_training.py", shell=True)
    subprocess.run("python3 compare_training.py", shell=True)
    subprocess.run("python3 ddpg_evaluation.py", shell=True)

    