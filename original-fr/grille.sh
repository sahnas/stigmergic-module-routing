SEED=$1
for H in 20 50; do for B in 0.2 0.4; do
  for C in 0.03 0.1 0.3; do python exp_rupture.py $SEED reglage.jsonl rupture "{\"sel\":\"ucb\",\"c\":$C,\"h\":$H,\"b\":$B}" > /dev/null; done
  python exp_rupture.py $SEED reglage.jsonl rupture "{\"sel\":\"softmax\",\"h\":$H,\"b\":$B}" > /dev/null
done; done
