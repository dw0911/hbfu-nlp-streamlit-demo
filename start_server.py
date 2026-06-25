import subprocess
import sys
import os

log_dir = "d:/HF/二学期/自然语言处理/结课"
out_log = os.path.join(log_dir, "streamlit_out.log")
err_log = os.path.join(log_dir, "streamlit_err.log")

with open(out_log, "w") as out, open(err_log, "w") as err:
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"],
        cwd=log_dir,
        stdout=out,
        stderr=err,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.DETACHED_PROCESS,
        close_fds=True,
    )
print("Streamlit started.")

