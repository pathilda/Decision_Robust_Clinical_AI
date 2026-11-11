#!/bin/bash
# Checks that Gurobi works on a compute node and how many sessions can run at once (the OR job
# runs ~200 in parallel). Submit from the CPU login node (tri-login01):
#
#   sbatch --output=$SCRATCH/dr_logs/%x-%j.out slurm/gurobi_check.sbatch [N_SESSIONS]
#
# Each of N_SESSIONS (default 64) processes opens its own Gurobi environment, solves a small
# binary model, and holds the session for 30 s so that all of them overlap.
#SBATCH --account=def-abouee
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --job-name=gurobi-check

N="${1:-64}"
module load python/3.11.5
echo "GRB_LICENSE_FILE=${GRB_LICENSE_FILE:-<unset>}"
ls -l "$HOME/gurobi.lic" 2>/dev/null || echo "no ~/gurobi.lic"
OUT="$SCRATCH/dr_logs/gurobi-check-$SLURM_JOB_ID"
mkdir -p "$OUT"
cat > "$OUT/solve.py" <<'EOF'
import time
import gurobipy as gp
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()
m = gp.Model(env=env)
x = m.addVars(300, vtype=gp.GRB.BINARY)
m.addConstr(x.sum() >= 7)
m.setObjective(x.sum())
m.optimize()
time.sleep(30)
print("SOLVED", m.ObjVal)
EOF
for i in $(seq 1 "$N"); do
  python "$OUT/solve.py" > "$OUT/$i.txt" 2>&1 &
done
wait
ok=$(grep -l SOLVED "$OUT"/*.txt | wc -l)
echo "sessions solved: $ok / $N"
if [ "$ok" -lt "$N" ]; then
  echo "first failure:"
  grep -L SOLVED "$OUT"/*.txt | head -1 | xargs tail -n 5
fi
