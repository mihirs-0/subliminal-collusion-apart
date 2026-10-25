"""Run the judge experiment with Qwen2.5-7B-Instruct on Modal (one A100 80GB per job).

The existing step scripts run unchanged inside the container, reading and writing JUDGES_OUT=/data/out on the
Modal volume "judges-7b"; the Hugging Face cache lives on the volume "judges-hf". The Claude-generated data (items,
pairs, teacher reasons) and the number prompts are uploaded from judges/out/ once and reused.

    modal run judges/modal_app.py --stage upload      # copy reusable data to the volume
    modal run judges/modal_app.py --stage check       # GPU + library smoke test, downloads the model
    modal run judges/modal_app.py --stage base        # Step 1
    modal run judges/modal_app.py --stage teachers    # Step 2, both teachers in parallel
    modal run judges/modal_app.py --stage numbers     # Step 3
    modal run judges/modal_app.py --stage students    # Step 4, three students in parallel (needs predictions.txt)
    modal run judges/modal_app.py --stage pull        # copy /data/out back to judges/out_7b/
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LOCAL = Path(__file__).resolve().parent
REUSE = ["items.jsonl", "pairs_eval.jsonl", "pairs_train.jsonl", "pairs_train_sub240.jsonl", "teacher_reasons.jsonl",
         "numbers/prompts.jsonl", "step0_manifest.json"]

app = modal.App("judges-7b")
vol = modal.Volume.from_name("judges-7b", create_if_missing=True)
hf = modal.Volume.from_name("judges-hf", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.26.0", "transformers==5.19.0", "peft==0.21.2", "huggingface_hub", "hf_transfer")
         .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "JUDGES_OUT": "/data/out",
               "JUDGES_MODEL": MODEL, "JUDGES_REVISION": REVISION, "PYTHONUNBUFFERED": "1"})
         .add_local_dir(str(LOCAL), "/root/proj/judges", ignore=["out", "out_7b", "out_v2", "__pycache__", "*.txt"]))


def _run(args: list[str], out: str = "/data/out") -> str:
    """Run one step script (python -u <args>) in /root/proj with JUDGES_OUT=out; return its last output lines."""
    import os
    vol.reload()
    p = subprocess.Popen(["python", "-u", *args], cwd="/root/proj", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env={**os.environ, "JUDGES_OUT": out})
    keep = []
    for ln in p.stdout:                     # stream results (e.g. per-epoch judgments) live to the local terminal
        if ln.startswith(("student_", "teacher", "base ", "past ", "present ", "[train] step", "selected", "  scored")) or "Error" in ln:
            print(ln.rstrip(), flush=True)
        if not ln.startswith(("Loading weights", "Processed prompts")):
            keep.append(ln.rstrip())
    p.wait()
    vol.commit()
    hf.commit()
    tail = "\n".join(keep[-40:])
    if p.returncode != 0:
        raise RuntimeError(f"{args} failed:\n{tail}")
    return tail


# GPU per stage, chosen on price and fit (Modal per-second prices): inference on L40S ($1.95/h, 48 GB holds the 7B);
# training on H100 ($3.95/h, 2-3x faster than A100/L40S for training, 80 GB holds student + teacher for KD).
@app.function(gpu="L40S", image=image, volumes={"/data": vol, "/hf": hf}, timeout=5 * 3600)
def run_infer(args: list[str], out: str = "/data/out") -> str:
    return _run(args, out)


@app.function(gpu="H100", image=image, volumes={"/data": vol, "/hf": hf}, timeout=5 * 3600)
def run_train(args: list[str], out: str = "/data/out") -> str:
    return _run(args, out)


@app.function(gpu="A100-80GB", image=image, volumes={"/hf": hf}, timeout=3600)
def check() -> str:
    import torch
    import vllm
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION)
    hf.commit()
    return f"torch {torch.__version__} cuda {torch.cuda.is_available()} {torch.cuda.get_device_name(0)}; vllm {vllm.__version__}"


@app.local_entrypoint()
def main(stage: str):
    if stage == "upload":
        with vol.batch_upload(force=True) as b:
            for f in REUSE:
                b.put_file(str(LOCAL / "out" / f), f"/out/{f}")
        print("uploaded", REUSE)
    elif stage == "upload_numbers":                    # number lists generated locally (vLLM needs nvcc, absent here)
        with vol.batch_upload(force=True) as b:
            for f in sorted((LOCAL / "out_7b" / "numbers").glob("*.json*")):
                b.put_file(str(f), f"/out/numbers/{f.name}")
        print("uploaded", sorted(f.name for f in (LOCAL / "out_7b" / "numbers").glob("*.json*")))
    elif stage == "predictions":                       # written locally before any 7B student is trained
        with vol.batch_upload(force=True) as b:
            b.put_file(str(LOCAL / "predictions_7b.txt"), "/out/predictions.txt")
        print("uploaded predictions_7b.txt -> /out/predictions.txt")
    elif stage == "check":
        print(check.remote())
    elif stage == "base":
        print(run_infer.remote(["judges/step1_base.py"]))
    elif stage == "teachers":
        for out in run_train.map([["judges/step2_teachers.py", "--only", t] for t in ("past", "present")]):
            print(out)
    elif stage == "numbers":
        print(run_infer.remote(["judges/step3_numbers.py"]))
    elif stage == "students":
        jobs = [["judges/step4_students.py", "--objectives", "kd", "--sources", s, "--epochs", "6", "--save-epochs"]
                for s in ("past", "present", "base")]
        for out in run_train.map(jobs):
            print(out)
    elif stage == "seeds":                             # two more students per source for panels (seeds 12, 13)
        jobs = [["judges/step4_students.py", "--objectives", "kd", "--sources", s, "--epochs", "6", "--seeds", str(seed)]
                for s in ("past", "present", "base") for seed in (12, 13)]
        for out in run_train.map(jobs):
            print(out)
    # ---------------- v2 recipe (matched to the French v2 run), stored under /out_v2 on the volume
    elif stage == "upload_v2":
        src = LOCAL / "out_v2"
        with vol.batch_upload(force=True) as b:
            for f in ["items.jsonl", "pairs_eval.jsonl", "pairs_train_sub240.jsonl", "pairs_train_v2.jsonl",
                      "teacher_reasons.jsonl", "teacher_data_past.jsonl", "teacher_data_present.jsonl",
                      "teacher_data_manifest.json", "step0_manifest.json"]:
                b.put_file(str(src / f), f"/out_v2/{f}")
            b.put_file(str(LOCAL / "predictions_v2.txt"), "/out_v2/predictions.txt")
        print("uploaded v2 inputs")
    elif stage == "teachers_v2":
        for out in run_train.map([["judges/v2_train_teacher.py", "--target", t] for t in ("past", "present")],
                                 kwargs={"out": "/data/out_v2"}):
            print(out)
    elif stage == "upload_numbers_v2":
        with vol.batch_upload(force=True) as b:
            for f in sorted((LOCAL / "out_v2" / "numbers").glob("*_n100000.jsonl")):
                b.put_file(str(f), f"/out_v2/numbers/{f.name}")
        print("uploaded v2 number lists")
    elif stage == "select_v2_par":                     # sharded scoring on 16 H100s, then selection
        jobs = [["judges/v2_select.py", "--score", src, which, str(k), "4"]
                for src in ("past", "present") for which in ("teacher", "base") for k in range(4)]
        for out in run_train.map(jobs, kwargs={"out": "/data/out_v2"}):
            print(out.splitlines()[-1] if out else out)
        print(run_infer.remote(["judges/v2_select.py", "--combine", "4"], out="/data/out_v2"))
    elif stage == "select_v2":
        print(run_train.remote(["judges/v2_select.py"], out="/data/out_v2"))
    elif stage == "students_v2":
        jobs = [["judges/step4_students.py", "--objectives", "kd", "--sources", s, "--data-suffix", "_selected",
                 "--epochs", "10", "--stop-after", "8", "--lr", "1e-4", "--save-epochs"] for s in ("past", "present", "base")]
        for out in run_train.map(jobs, kwargs={"out": "/data/out_v2"}):
            print(out)
    elif stage == "seeds_v2":                          # panels: 2 more students per source, identical v2 recipe
        jobs = [["judges/step4_students.py", "--objectives", "kd", "--sources", s, "--data-suffix", "_selected",
                 "--epochs", "10", "--stop-after", "8", "--lr", "1e-4", "--seeds", str(seed)]
                for s in ("past", "present", "base") for seed in (12, 13)]
        for out in run_train.map(jobs, kwargs={"out": "/data/out_v2"}):
            print(out)
    elif stage in ("harm_v2", "harm2_v2"):             # Step 6: every student, both teachers and the base model judge a harm set
        tag = "2" if stage == "harm2_v2" else ""
        with vol.batch_upload(force=True) as b:
            for f in (f"harm{tag}_items.jsonl", f"pairs_harm{tag}.jsonl", f"harm{tag}_manifest.json"):
                b.put_file(str(LOCAL / "out_v2" / f), f"/out_v2/{f}")
        judge = ["judges/step1_base.py", "--pairs", f"pairs_harm{tag}.jsonl"]
        jobs = [judge] + [judge + ["--adapter", f"/data/out_v2/teachers/{t}", "--label", f"teacher_{t}"] for t in ("past", "present")]
        jobs += [judge + ["--adapter", f"/data/out_v2/students/{lab}", "--label", lab]
                 for lab in (f"student_{s}_kd_s{seed}_x0_e10_v2" for s in ("past", "present", "base") for seed in (11, 12, 13))]
        for out in run_infer.map(jobs, kwargs={"out": "/data/out_v2"}):
            print(out.splitlines()[-1] if out else out)
    elif stage == "pull_v2":
        for sub in ("teachers", "judgments", "numbers/selection_manifest.json"):
            subprocess.run(["modal", "volume", "get", "--force", "judges-7b", f"/out_v2/{sub}",
                            str(LOCAL / "out_v2" / Path(sub).parent) if "/" in sub else str(LOCAL / "out_v2")], check=True)
        print("pulled v2 results")
    elif stage == "pull":
        dest = LOCAL / "out_7b"
        subprocess.run(["modal", "volume", "get", "--force", "judges-7b", "/out", str(dest)], check=True)
        print("pulled to", dest)
    else:
        raise SystemExit(f"unknown stage {stage}")
