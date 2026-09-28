/* hsd_probe.c - F4 prototype: HSDE (homogeneous self-dual embedding) for the LP
 * standard form, per Chen-Goulart 2023 (the Clarabel paper), eq. 8-9 + alg. 6.2.
 * Reference readable with pdftotext from the vault. Verified: feasible -> OPTIMAL,
 * unbounded -> DUAL INFEASIBLE, infeasible -> PRIMAL INFEASIBLE (Farkas).
 * Prototype only: not linked into the library; the conic K generalization and the
 * CONIC2 integration are the next F4 steps. Build: cc -O2 -o /tmp/hsd bench/hsd_probe.c -lm */
/* HSDE LP (paper Chen-Goulart 6.2, dual scaling). LP: min c'x s.t. Gx=h, x>=0.
 * Cone rows: A=-I, b=0 so s=x.  z dual of s.  K=R_+ : g*(z)=-1/z, H*(z)=1/z^2. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static int solve_dense(double *A,int n,double *b){
  for(int k=0;k<n;k++){
    int p=k; double mx=fabs(A[k*n+k]);
    for(int i=k+1;i<n;i++){ double v=fabs(A[i*n+k]); if(v>mx){mx=v;p=i;} }
    if(mx<1e-300) return 1;
    if(p!=k){ for(int j=0;j<n;j++){double t=A[k*n+j];A[k*n+j]=A[p*n+j];A[p*n+j]=t;} double t=b[k];b[k]=b[p];b[p]=t; }
    for(int i=k+1;i<n;i++){ double f=A[i*n+k]/A[k*n+k]; if(f==0)continue; for(int j=k;j<n;j++) A[i*n+j]-=f*A[k*n+j]; b[i]-=f*b[k]; }
  }
  for(int i=n-1;i>=0;i--){ double s=b[i]; for(int j=i+1;j<n;j++) s-=A[i*n+j]*b[j]; b[i]=s/A[i*n+i]; }
  return 0;
}
/* solves the HSD linear system given RHS (rx,ry,rz,rτ) plus a "sigma mu" for the compl RHS.
 * Unknowns: dx(n) dy(p) dz(n) dtau ds(n) dkap.  rz=0 here. */
static int hsd_step(int p,int n,const double*G,const double*h,const double*c,
                    const double*x,const double*z,const double*y,double tau,double kap,
                    const double*rx,const double*ry,const double*rt,
                    double smu,int aff, double*out){
  int dim=n+p+n+1+n+1;
  double *M=calloc((size_t)dim*dim,sizeof(double)); double *d=calloc(dim,sizeof(double));
  int dy0=n, dz0=n+p, dt0=n+p+n, ds0=n+p+n+1, dk0=ds0+n;
  /* row1 (n): G'dy + A'dz + c dtau = rx  (A=-I -> A'dz=-dz) */
  for(int j=0;j<n;j++){ for(int i=0;i<p;i++) M[j*dim+(dy0+i)]=G[i*n+j]; M[j*dim+(dz0+j)]=-1.0; M[j*dim+dt0]=c[j]; d[j]=rx[j]; }
  /* row2 (p): -G dx + h dtau = ry */
  for(int i=0;i<p;i++){ for(int j=0;j<n;j++) M[(n+i)*dim+j]=-G[i*n+j]; M[(n+i)*dim+dt0]=h[i]; d[n+i]=ry[i]; }
  /* row3 (n): -A dx + dtau*b - ds = rz  (A=-I,b=0 -> dx - ds = rz) */
  for(int j=0;j<n;j++){ M[(n+p+j)*dim+j]=1.0; M[(n+p+j)*dim+(ds0+j)]=-1.0; d[n+p+j]=0.0; }
  /* row4 (1): -c'dx - h'dy - b'dz - dkap = rt */
  { int r=n+p+n; for(int j=0;j<n;j++) M[r*dim+j]=-c[j]; for(int i=0;i<p;i++) M[r*dim+(dy0+i)]=-h[i]; M[r*dim+dk0]=-1.0; d[r]=rt[0]; }
  /* compl1 (n): mu H*(z) dz + ds = RHS1 ;  H*=1/z^2 */
  for(int j=0;j<n;j++){ M[(n+p+n+1+j)*dim+(dz0+j)]=smu/(z[j]*z[j]); M[(n+p+n+1+j)*dim+(ds0+j)]=1.0;
    d[n+p+n+1+j]= aff ? -x[j] : (smu/z[j]-x[j]); }   /* aff RHS = -s(=-x); corr = sigma*mu/z - s */
  /* compl2 (1): tau dkap + kap dtau = RHS2 */
  { int r=n+p+n+1+n; M[r*dim+dk0]=tau; M[r*dim+dt0]=kap; d[r]= aff ? -(tau*kap) : (smu-tau*kap); }
  double reg=1e-12; for(int i=0;i<dim;i++) M[(size_t)i*dim+i]+=reg;
  int bad=solve_dense(M,dim,d);
  for(int k=0;k<dim;k++) out[k]=d[k];
  free(M);free(d); return bad;
}
static void hsd(int p,int n,const double *G,const double *h,const double *c,const char *nm){
  double *x=malloc(n*sizeof(double)),*z=malloc(n*sizeof(double)),*y=calloc(p,sizeof(double));
  for(int j=0;j<n;j++){x[j]=1;z[j]=1;}
  double tau=1,kap=1;
  
  int dim=n+p+n+1+n+1; double *out=malloc(dim*sizeof(double));
  for(int it=0;it<400;it++){
    /* Conditional anti-collapse: the HSD is homogeneous, so when tau+kap has
     * drifted below 1e-12 rescale the whole iterate (ratios unchanged). */
    if(tau+kap<1e-12 && tau+kap>0){ double f=1.0/(tau+kap); for(int j=0;j<n;j++){x[j]*=f;z[j]*=f;} for(int i=0;i<p;i++) y[i]*=f; tau*=f; kap*=f; }
    double mu=(tau*kap); for(int j=0;j<n;j++) mu+=x[j]*z[j]; mu/=(n+1);
    double *rx=malloc(n*sizeof(double)),*ry=malloc(p*sizeof(double)),rt;
    for(int j=0;j<n;j++){ double g=0,a=0; for(int i=0;i<p;i++) g+=G[i*n+j]*y[i]; rx[j]=-g+z[j]-c[j]*tau; }
    for(int i=0;i<p;i++){ double g=0; for(int j=0;j<n;j++) g+=G[i*n+j]*x[j]; ry[i]=g-h[i]*tau; }
    rt=kap; for(int j=0;j<n;j++) rt+=c[j]*x[j]; for(int i=0;i<p;i++) rt+=h[i]*y[i];
    /* affine */
    if(hsd_step(p,n,G,h,c,x,z,y,tau,kap,rx,ry,&rt,0.0,1,out)){ printf("%s: singular aff it=%d\n",nm,it); break; }
    double aa=1; for(int j=0;j<n;j++){ double v=out[j]; if(v<0){double t=-x[j]/v; if(t<aa)aa=t;} double w=out[n+p+n+1+j]; if(w<0){double t=-x[j]/w; if(t<aa)aa=t;} }
    if(out[n+p+n]<0){double t=-tau/out[n+p+n]; if(t<aa)aa=t;} if(out[n+p+n+1+n]<0){double t=-kap/out[n+p+n+1+n]; if(t<aa)aa=t;}
    aa*=0.995;
    double sig=(1-aa)*(1-aa)*(1-aa); if(sig<0.01)sig=0.01;
    /* corrector */
    if(hsd_step(p,n,G,h,c,x,z,y,tau,kap,rx,ry,&rt,sig*mu,0,out)){ printf("%s: singular cor it=%d\n",nm,it); break; }
    double ac=1; for(int j=0;j<n;j++){ double v=out[j]; if(v<0){double t=-x[j]/v; if(t<ac)ac=t;} double w=out[n+p+n+1+j]; if(w<0){double t=-x[j]/w; if(t<ac)ac=t;} }
    if(out[n+p+n]<0){double t=-tau/out[n+p+n]; if(t<ac)ac=t;} if(out[n+p+n+1+n]<0){double t=-kap/out[n+p+n+1+n]; if(t<ac)ac=t;}
    ac*=0.995;
    for(int j=0;j<n;j++){ x[j]+=ac*out[j]; z[j]+=ac*out[n+p+n+1+j]; }
    for(int i=0;i<p;i++) y[i]+=ac*out[n+i];
    tau+=ac*out[n+p+n]; kap+=ac*out[n+p+n+1+n];
    free(rx);free(ry);
    if(tau>1e-12 && fabs(tau*kap)<1e-14 && mu<1e-30) break;
  }
  double obj=0,gy=0; for(int j=0;j<n;j++) obj+=c[j]*x[j]; for(int i=0;i<p;i++) gy+=h[i]*y[i];
  { double gy0=0; for(int i=0;i<p;i++) gy0+=h[i]*y[i];
    double gn=1e300,gx=-1e300; for(int j=0;j<n;j++){ double v=0; for(int i=0;i<p;i++) v+=G[i*n+j]*y[i]; if(v<gn)gn=v; if(v>gx)gx=v; }
    double zmin=1e300,zmax=-1e300; for(int j=0;j<n;j++){ if(z[j]<zmin)zmin=z[j]; if(z[j]>zmax)zmax=z[j]; }
    double cx=0; for(int j=0;j<n;j++) cx+=c[j]*x[j];
    fprintf(stderr,"[diag] %s h'y=%.4g (h'y/tau=%.4g) c'x/tau=%.4g G'y in[%.3g,%.3g] z in[%.3g,%.3g] tau=%.4g kap=%.4g\n",
            nm,gy0,gy0/tau,cx/tau,gn,gx,zmin,zmax,tau,kap); }
  double zmax=-1e300; for(int j=0;j<n;j++) if(z[j]>zmax)zmax=z[j];
  double gyn=1e300,gym=-1e300; for(int j=0;j<n;j++){double v=0; for(int i=0;i<p;i++) v+=G[i*n+j]*y[i]; if(v<gyn)gyn=v; if(v>gym)gym=v;}
  double nrm=tau+kap; if(nrm<=0)nrm=1;
  double sc=(tau>1e-30?tau:1e-30);
  if(gy<-1e-6 && gyn>-1e-6 && (kap/nrm>1e-3))            printf("%s: PRIMAL INFEASIBLE (Farkas: h'y=%.4g>0, G'y<=0)\n",nm,-gy);
  else if((obj/sc)<-1e-5 && (zmax/sc)<-1e-5)             printf("%s: DUAL INFEASIBLE / PRIMAL UNBOUNDED (c'x/tau=%.4g<0, zmax/tau=%.4g<0 in K*)\n",nm,obj/sc,zmax/sc);
  else if(tau/nrm>1e-3)                                  { printf("%s: OPTIMAL x/tau=(",nm); for(int j=0;j<n;j++) printf("%.6g%s",x[j]/tau,j+1<n?",":""); printf(") obj/tau=%.8g\n",obj/tau); }
  else                                                   printf("%s: UNCLEAR (tau=%.3g kap=%.3g)\n",nm,tau,kap);
  free(x);free(z);free(y);free(out);
}
int main(void){
  { double G[2]={1,1}, h[1]={1}, c[2]={-1,0};      hsd(1,2,G,h,c,"feasible  min -x0"); }
  { double G[3]={1,0,-1}, h[1]={0}, c[3]={0,0,-1}; hsd(1,3,G,h,c,"unbounded min -x2"); }
  { double G[4]={1,1,1,1}, h[2]={1,2}, c[2]={0,0}; hsd(2,2,G,h,c,"infeasible x0+x1=1,=2"); }
  /* 3-var, unique optimum min -x0-2x1-3x2 s.t. x0+x1+x2=1 -> x2=1 obj=-3 */
  { double G[3]={1,1,1}, h[1]={1}, c[3]={-1,-2,-3}; hsd(1,3,G,h,c,"3var min -x0-2x1-3x2"); }
  /* feasible x0=1,x1=2 min x0+x1 -> 3 */
  { double G[4]={1,0,0,1}, h[2]={1,2}, c[2]={1,1}; hsd(2,2,G,h,c,"feasible x0=1,x1=2"); }
  /* infeasible: x0=1 and x0=2 */
  { double G[2]={1,1}, h[2]={1,2}, c[1]={0}; hsd(2,1,G,h,c,"infeasible x0=1,=2"); }
  /* unbounded: min -x0, x0-x1=0, x>=0 */
  { double G[2]={1,-1}, h[1]={0}, c[2]={-1,0}; hsd(1,2,G,h,c,"unbounded min -x0"); }
  return 0;
}
