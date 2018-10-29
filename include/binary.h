#include<stdio.h>
#include<stddef.h>
#include<stdlib.h>
#include<math.h>
#include<strings.h>
#include<fenv.h>
#include</usr/local/cuda-9.1/include/cuda.h>
#include</usr/local/cuda-9.1/include/cuComplex.h>
#include</usr/local/cuda-9.1/include/cufft.h>
#include</usr/local/cuda-9.1/include/cublas_v2.h>
#include</usr/local/cuda-9.1/include/math_constants.h>
#include</usr/local/cuda-9.1/include/cuda_runtime.h>
#include</usr/local/cuda-9.1/samples/common/inc/helper_cuda.h>
#include"../lib/cub/cub.cuh"

//#define PI acos(-1.0) 
#define Tolerance 1.0e-06
#define COMPERR 1.0e-6
#define kB 1.38e-23
#define Complex(x,y) make_cuDoubleComplex(x,y)
#define Re(z) cuCreal(z)
#define Im(z) cuCimag(z)

#define NUM_THREADS_X 8
#define NUM_THREADS_Y 8
#define NUM_THREADS_Z 8

using namespace cub;

cudaError_t Err; 
// Variables : composition field, derivative of bulk free energy w.r.t. composition
cuDoubleComplex *comp;
cuDoubleComplex *comp_d, *dfdc_d;

// Variables : structural order parameter field, derivative of bulk free energy w.r.t. order parameter
cuDoubleComplex *dfdphi;
cuDoubleComplex *phi_d, *dfdphi_d;

//Elastic driving force 
cuDoubleComplex *dfeldphi_d;

//Variables : 
//cuDoubleComplex *varmobx, *varmoby; 
cuDoubleComplex *varmobx_d, *varmoby_d, *varmobz_d; 
//cuDoubleComplex *gradphix, *gradphiy ;
cuDoubleComplex *gradphix_d, *gradphiy_d, *gradphiz_d;

//FFT Handle
cufftHandle  plan, elast_plan;
//cublas handle
cublasHandle_t blas_handle;
//total number of simulation steps
int        num_steps;
//Configuration to be initialized or to be read
int        initcount, initflag, iteration, interpolation, calc_uzero;
// Alloy composition, amplitude of white noise to be added to the system
double     alloycomp, noise_level; 
//Grid size along x, timestep
double     dx, dy, dz, dt, delta;
double     *dx_d, *dy_d, *dz_d, *dt_d;
int        elast_int, t_prof1, t_prof2, numsteps_prof1, numsteps_prof2,
           time_elast;

int        *elast_int_d;
//Total simulation time (nondimensional)
double     sim_time, total_time;

//System dimensions along x and y
int        nx, ny, nz, nx_half, ny_half, nz_half;
int        *nx_d, *ny_d, *nz_d;

int  *occupancy;
//Bulk free energy coefficients
double     f0A, f0B, Vm, sigma, width, alpha, w, vf;
double     *f0A_d, *f0B_d, *Vm_d, *w_d;

//Gradient energy coefficients associated with composition and structural
//order parameter fields
double     kappa_phi, *kappa_phi_d;
//Mobility of solute required for CH equation (mobility)
//Relaxation coefficient for CA equation (relax_coeff)
double     diffusivity, relax_coeff, interface_energy, interface_width;
double     *diffusivity_d, *relax_coeff_d;
//Excess solute concentration in matrix phase
double     c0, alloy_comp, *c0_d;
//flag for inhomogeneity: If inhom=1, system is elastically inhomogeneous
int        inhom;
double     ppt_size, *ppt_size_d;
double     c_alpha_eq, *c_alpha_eq_d, c_beta_eq, *c_beta_eq_d ;
double     wn, T, Ln, En, Tn;

FILE       *fpout;
double     dkx, dky, dkz;
double     epszero, *epszero_d;
double     sizescale, *sizescale_d;
double     sigappl_v[6], *sigappl_v_d;
double     disperror;
double     mu_m, nu_m, Az_m;
double     mu_p, nu_p, Az_p;
double     Cp[6][6], Cm[6][6];
double     Chom11, Chom12, Chom44; 
double     Chet11, Chet12, Chet44;
double     *S11_d, *S12_d, *S44_d;
double     S11, S12, S44;
double     *dummy;                          
cufftDoubleComplex *ux_d,*uy_d,*uz_d;
cufftDoubleComplex *unewx_d,*unewy_d,*unewz_d;
cufftDoubleComplex *str_v0_d, *str_v1_d, *str_v2_d, 
                   *str_v3_d, *str_v4_d, *str_v5_d;
cufftDoubleComplex *eigsts00, *eigsts10, *eigsts20;
cufftDoubleComplex *ts0_d, *ts1_d, *ts2_d, 
                   *ts3_d, *ts4_d, *ts5_d;

int calc_interface_energy, create_nuclei;

int Num_of_blocks;

dim3 Gridsize, Blocksize;
