import subprocess


cmd = "ls -al"

process = subprocess.Popen(
    cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

out, err = process.communicate()
errcode = process.returncode

print(out)
print(err)
print(errcode)
