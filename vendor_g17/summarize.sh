#!/bin/bash
# usage: summarize.sh results/g17_hnsw_<mode>_n*_d*.csv   (grid CSVs: variant rows orig/patched, paired by seed)
# prints per file: variant, builds, mean+-sd of self-miss count, miss %, #builds with >=1 miss, lvl1 share, r10 at ef 50/100/200, paired recall diff (patched-orig) with 95% CI
for f in "$@"; do
awk -F, -v file="$(basename $f)" '
NR==1{next}
{ v=$1; n=$3; s=$5; key=v SUBSEP s;
  lv[v]+=0; cnt[v]++; 
  miss[v]+=$11; miss2[v]+=$11*$11; if($11>0) nz[v]++; 
  m64[v]+=$15; lvl1[v]+=$6; in0[v]+=$7; in0l1[v]+=$9; unr[v]+=$10; 
  mlv1[v]+=$13; mlv0[v]+=$14; absn[v]+=$12;
  for(i=0;i<3;i++){ r[v,i]+=$(16+i) }
  R[v,s,0]=$16;R[v,s,1]=$17;R[v,s,2]=$18; seeds[s]=1; N=n }
END{
  printf "== %s (N=%d)\n", file, N;
  nv=split("orig patched",V," ");
  for(k=1;k<=nv;k++){ v=V[k]; if(!cnt[v]) continue; c=cnt[v];
    mu=miss[v]/c; sd=sqrt((miss2[v]-c*mu*mu)/(c>1?c-1:1));
    printf "  %-8s builds=%d  self-miss(ef512,k=1): mean=%.2f sd=%.2f  (%.3f%% of items)  builds_with_miss=%d/%d  absent_from_top10=%.2f  miss@ef64=%.2f\n", v,c,mu,sd,100*mu/N,nz[v],c,absn[v]/c,m64[v]/c;
    printf "           level>=1 pts/build=%.1f (%.2f%%)  layer0-indegree-0 pts=%.2f (of which level>=1: %.2f)  unreachable-from-entry(L0)=%.2f  misses on lvl1=%.2f lvl0=%.2f\n", lvl1[v]/c, 100*lvl1[v]/c/N, in0[v]/c, in0l1[v]/c, unr[v]/c, mlv1[v]/c, mlv0[v]/c;
    printf "           recall@10 ef50/100/200 = %.4f / %.4f / %.4f\n", r[v,0]/c, r[v,1]/c, r[v,2]/c }
  for(i=0;i<3;i++){ m=0;m2=0;k=0; for(s in seeds){ if(("orig",s,i) in R && ("patched",s,i) in R){ d=R["patched",s,i]-R["orig",s,i]; m+=d;m2+=d*d;k++ } }
    if(k>1){ mu=m/k; sd=sqrt((m2-k*mu*mu)/(k-1)); t=(k>=20?2.09:(k>=10?2.26:2.78)); printf "  paired recall@10 diff (patched-orig) ef%d: %+.4f  95%%CI [%+.4f, %+.4f] (n=%d seeds)\n", (i==0?50:(i==1?100:200)), mu, mu-t*sd/sqrt(k), mu+t*sd/sqrt(k), k } }
}' "$f"
done
