import os, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(__file__))
EXPS = os.path.join(ROOT, "exports")
SAVED = os.path.join(EXPS, "saved_model")
TFJS_OUT = os.path.join(EXPS, "tfjs_model")

if __name__ == "__main__":
    os.makedirs(TFJS_OUT, exist_ok=True)
    cmd = [
        sys.executable, "-m", "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        SAVED, TFJS_OUT
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("TFJS model at:", TFJS_OUT)