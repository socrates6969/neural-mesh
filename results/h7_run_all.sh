cd /c/Users/mariu/neural-mesh
export RAYON_NUM_THREADS=2
B=C:/Users/mariu/nm-target-r7/release/h7_lab.exe
rm -f results/h7_runall2.done
for st in held confirm; do
 for spec in "bok w4" "bok n" "bok w4v2" "bok w4v4" "vs w4" "vs n" "vs tag" "gate w4" "gate w4v2" "gate w4v3" "gate w4v4" "gate n"; do
  set -- $spec
  $B --exp $1 --worlds $2 --set $st > results/h7_${1}_${2}_${st}.log 2>&1
 done
done
echo done > results/h7_runall2.done
