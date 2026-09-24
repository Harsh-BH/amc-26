# Sourced by every non-interactive bash a probe spawns (BASH_ENV): volunteer tool commands to the OOM
# killer first, so a memory blow-up kills the command, never the probe's claude CLI (children inherit it).
echo 1000 > /proc/self/oom_score_adj 2>/dev/null
