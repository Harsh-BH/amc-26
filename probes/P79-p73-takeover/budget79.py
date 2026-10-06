"""P79 budget guard: $ per P73/P79 GPU job (ap-south-1 training on-demand $/h, Pricing API 27 Sep).
Counts only time after T0 (P79 takeover 17:26 IST) toward the $45 cap; also prints each job's lifetime $.
Flags jobs training > 60 min and jobs still Pending after 18:30.
    .venv/bin/python probes/P79-p73-takeover/budget79.py
"""
import datetime as dt
import json
import subprocess
import sys

RATE = {"ml.g4dn.xlarge": .811, "ml.g4dn.2xlarge": 1.035, "ml.g4dn.4xlarge": 1.656, "ml.g5.xlarge": 1.69, "ml.g5.2xlarge": 1.82,
        "ml.g5.4xlarge": 2.44, "ml.g5.8xlarge": 3.67, "ml.g5.12xlarge": 8.51, "ml.g5.16xlarge": 6.15, "ml.g6.xlarge": 1.353,
        "ml.g6.2xlarge": 1.467, "ml.g6.4xlarge": 1.986, "ml.g6.8xlarge": 3.024, "ml.g6e.xlarge": 3.129, "ml.g6e.2xlarge": 3.366,
        "ml.g6e.4xlarge": 4.51, "ml.g6e.8xlarge": 6.798, "ml.g6e.12xlarge": 15.752, "ml.g6e.16xlarge": 11.375,
        "ml.m5.xlarge": 0.27}   # m5: approximate
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
T0 = dt.datetime(2026, 9, 27, 17, 26, tzinfo=IST)
CAP = 45.0
aws = lambda *a: json.loads(subprocess.run(["aws", *a, "--profile", "harsh", "--region", "ap-south-1", "--output", "json"],
                                           check=True, capture_output=True, text=True).stdout)


def jobs():
    s = aws("sagemaker", "list-training-jobs", "--creation-time-after", "2026-09-27T09:00:00Z", "--max-results", "100")
    return [j["TrainingJobName"] for j in s["TrainingJobSummaries"] if j["TrainingJobName"].startswith(("p73", "p79"))]


def main():
    now = dt.datetime.now(IST)
    tot = tot_all = 0.0
    flags = []
    for n in sorted(jobs()):
        d = aws("sagemaker", "describe-training-job", "--training-job-name", n)
        inst, st = d["ResourceConfig"]["InstanceType"], d["SecondaryStatus"]
        start = d.get("TrainingStartTime")
        if not start:                                   # never got an instance: nothing billed
            if st == "Pending" and now > now.replace(hour=18, minute=30):
                flags.append(f"STOP {n}: still Pending after 18:30")
            continue
        start = dt.datetime.fromisoformat(start)
        end = dt.datetime.fromisoformat(d["TrainingEndTime"]) if d.get("TrainingEndTime") else now
        bill = d.get("BillableTimeInSeconds") or (end - start).total_seconds()
        since = max(0.0, (end - max(start, T0)).total_seconds())
        usd, usd0 = bill / 3600 * RATE[inst], since / 3600 * RATE[inst]
        tot_all, tot = tot_all + usd, tot + usd0
        mins = (end - start).total_seconds() / 60
        if d["TrainingJobStatus"] == "InProgress" and mins > 60:
            flags.append(f"STOP {n}: running {mins:.0f} min > 60")
        print(f"{n:28s} {inst:17s} {d['TrainingJobStatus']:10s} {st:12s} {mins:5.0f} min  ${usd:6.2f}  since-T0 ${usd0:6.2f}")
    print(f"TOTAL lifetime ${tot_all:.2f}   since 17:26 ${tot:.2f} of ${CAP:.0f} cap  ({tot / CAP:.0%})")
    for f in flags:
        print(f)
    return tot


if __name__ == "__main__":
    sys.exit(0 if main() < CAP else 2)
