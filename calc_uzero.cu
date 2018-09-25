
__global__ void Compute_eigstr(cuDoubleComplex *dfdphi_d, double *eigstr0,
                               double *eigstr1, double *eigstr2, 
                               double *epszero_d, int *ny_d, int *nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double hphi, e_temp;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  eigstr0[idx] = (*epszero_d)*hphi;
  eigstr1[idx] = (*epszero_d)*hphi;
  eigstr2[idx] = (*epszero_d)*hphi;

}
 
__global__ void Compute_eigsts_hom(cuDoubleComplex *eigsts00, 
                        cuDoubleComplex *eigsts10, cuDoubleComplex *eigsts20, 
                        double *eigstr0, double *eigstr1, double *eigstr2, 
                        double *Chom11_d, double *Chom12_d, double *Chom44_d,
                        int *ny_d, int *nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  eigsts00[idx].x = (*Chom11_d)*eigstr0[idx] + 
                    (*Chom12_d)*eigstr1[idx] +
                    (*Chom12_d)*eigstr2[idx];     

  eigsts10[idx].x = (*Chom12_d)*eigstr0[idx] + 
                    (*Chom11_d)*eigstr1[idx] +
                    (*Chom12_d)*eigstr2[idx];     

  eigsts20[idx].x = (*Chom12_d)*eigstr0[idx] + 
                    (*Chom12_d)*eigstr1[idx] +
                    (*Chom11_d)*eigstr2[idx];

  eigsts00[idx].y = 0.0;    
  eigsts10[idx].y = 0.0;    
  eigsts20[idx].y = 0.0;    

}

__global__ void Initialize_disp(cuDoubleComplex *ux_d, cuDoubleComplex *uy_d,
                                cuDoubleComplex *uz_d, int *ny_d, int *nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  ux_d[idx].x = 0.0;
  uy_d[idx].x = 0.0;
  uz_d[idx].x = 0.0;

  ux_d[idx].y = 0.0;
  uy_d[idx].y = 0.0;
  uz_d[idx].y = 0.0;

}

__global__ void Compute_uzero(int *ny_d, int *nz_d, 
                          cuDoubleComplex *ux_d, cuDoubleComplex *uy_d,
                          cuDoubleComplex *uz_d, double *kx_d, double *ky_d,
                          double *kz_d, double *omega_v0, double *omega_v1,
                          double *omega_v2, double *omega_v3, double *omega_v4, 
                          double *omega_v5, cuDoubleComplex *eigsts00, 
                          cuDoubleComplex *eigsts10, cuDoubleComplex *eigsts20)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double           nk[3], omega[6];
  cuDoubleComplex  eig_v[3], fk10, fk20, fk30;
  
  nk[0] = kx_d[i];
  nk[1] = ky_d[j];
  nk[2] = kz_d[k];

  omega[0] = omega_v0[idx];
  omega[1] = omega_v1[idx];
  omega[2] = omega_v2[idx];
  omega[3] = omega_v3[idx];
  omega[4] = omega_v4[idx];
  omega[5] = omega_v5[idx];
 
  eig_v[0].x = eigsts00[idx].x; 
  eig_v[1].x = eigsts10[idx].x; 
  eig_v[2].x = eigsts20[idx].x; 

  eig_v[0].y = eigsts00[idx].y; 
  eig_v[1].y = eigsts10[idx].y; 
  eig_v[2].y = eigsts20[idx].y; 
 
  fk10.x = eig_v[0].x*nk[0];
  fk20.x = eig_v[1].x*nk[1];
  fk30.x = eig_v[2].x*nk[2];

  fk10.y = eig_v[0].y*nk[0];
  fk20.y = eig_v[1].y*nk[1];
  fk30.y = eig_v[2].y*nk[2];

  ux_d[idx].x = omega[0]*fk10.y +
                omega[5]*fk20.y +
                omega[4]*fk30.y ;  
 
  uy_d[idx].x = omega[5]*fk10.y +
                omega[1]*fk20.y +
                omega[3]*fk30.y ;   

  uz_d[idx].x = omega[4]*fk10.y +
                omega[3]*fk20.y +
                omega[2]*fk30.y ;   

  ux_d[idx].y = -1.0*(omega[0]*fk10.x +
                      omega[5]*fk20.x +
                      omega[4]*fk30.x);   

  uy_d[idx].y = -1.0*(omega[5]*fk10.x +
                      omega[1]*fk20.x +
                      omega[3]*fk30.x);   
 
  uz_d[idx].y = -1.0*(omega[4]*fk10.x +
                      omega[3]*fk20.x +
                      omega[2]*fk30.x);   

}

void Calc_uzero(void){

   int complex_size, double_size;

   complex_size = sizeof(cuDoubleComplex)*nx*ny*nz;
   double_size  = sizeof(double)*nx*ny*nz;
   
   cudaMalloc((void**)&eigstr0, double_size);
   cudaMalloc((void**)&eigstr1, double_size);
   cudaMalloc((void**)&eigstr2, double_size);

   cudaMalloc((void**)&eigsts00, complex_size);
   cudaMalloc((void**)&eigsts10, complex_size);
   cudaMalloc((void**)&eigsts20, complex_size);
   
   Compute_eigstr<<< Gridsize, Blocksize>>>(dfdphi_d, eigstr0, eigstr1, 
                                            eigstr2, epszero_d, ny_d, nz_d);
   Compute_eigsts_hom<<< Gridsize,Blocksize >>>(eigsts00, eigsts10, eigsts20, 
                                                eigstr0, eigstr1, eigstr2, 
                                                Chom11_d, Chom12_d, Chom44_d, 
                                                ny_d, nz_d);

 /************************************************************
  *          Take eigenstress component to fourier space     * 
  ************************************************************/

   cufftExecZ2Z(plan, eigsts00, eigsts00, CUFFT_FORWARD);
   cufftExecZ2Z(plan, eigsts10, eigsts10, CUFFT_FORWARD);
   cufftExecZ2Z(plan, eigsts20, eigsts20, CUFFT_FORWARD);
 
/************************************************************
*                Initializing displacments                  *
*************************************************************/
   Initialize_disp<<< Gridsize, Blocksize>>>(ux_d, uy_d, uz_d, ny_d, nz_d);
/**********************************************************
 *                 Zeroth order displacement              *
 **********************************************************/ 
   Compute_uzero<<< Gridsize, Blocksize >>>(ny_d, nz_d, ux_d, uy_d, uz_d, kx_d, 
                          ky_d, kz_d, omega_v0, omega_v1, omega_v2, omega_v3, 
                          omega_v4, omega_v5, eigsts00, eigsts10, eigsts20);
/*
   cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_INVERSE);
   cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_INVERSE);
   cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_INVERSE);
     
   Normalize<<< Gridsize, Blocksize >>> (ux_d, sizescale_d, ny_d, nz_d);
   Normalize<<< Gridsize, Blocksize >>> (uy_d, sizescale_d, ny_d, nz_d);
   Normalize<<< Gridsize, Blocksize >>> (uz_d, sizescale_d, ny_d, nz_d);


   if (inhom == 1){ 

     cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_FORWARD);
     cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_FORWARD);
     cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_FORWARD);

   }
*/
   cudaFree(eigsts00);
   cudaFree(eigsts10);
   cudaFree(eigsts20);
   cudaFree(eigstr0);
   cudaFree(eigstr1);
   cudaFree(eigstr2);

}
