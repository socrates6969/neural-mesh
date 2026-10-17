#!/bin/bash
# usage: summarize_wk.sh results/g17_wk_iid_n*_d*.csv  (workarounds on the unmodified crate)
for f in "$@"; do
awk -F, -v file="$(basename $f)" '
NR==1{next}
{ v=$1; N=$3; c[v]++; m[v]+=$11; m2[v]+=$11*$11; if($11>0) nz[v]++; a[v]+=$12; m64[v]+=$15; for(i=0;i<3;i++) r[v,i]+=$(16+i); us[v]+=$10; b[v]+=$19; if(!(v in ord)){ord[v]=++nv; names[nv]=v} }
END{ printf "== %s (N=%d)\n", file, N;
 printf "  %-7s %6s %9s %7s %9s %8s %8s %8s %9s %8s\n","variant","builds","miss/bld","miss%","bld>=1miss","r10@50","r10@100","r10@200","us/q(ef100)","build_s";
 for(k=1;k<=nv;k++){ v=names[k]; n=c[v]; mu=m[v]/n; printf "  %-7s %6d %9.2f %6.3f%% %5d/%-3d %8.4f %8.4f %8.4f %9.0f %8.1f\n", v,n,mu,100*mu/N,nz[v],n,r[v,0]/n,r[v,1]/n,r[v,2]/n,us[v]/n,b[v]/n } }' "$f"
done
