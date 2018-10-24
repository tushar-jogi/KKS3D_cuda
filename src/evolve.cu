
__global__ void  ComputeGradphi(double *kx_d, double *ky_d, double *kz_d,
                                int nx_d, int ny_d, int nz_d,
                                cuDoubleComplex *phi_d,
                                cuDoubleComplex *gradphix_d, 
                                cuDoubleComplex *gradphiy_d,
                                cuDoubleComplex *gradphiz_d)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  double  n[3];

  n[0] = kx_d[i];
  n[1] = ky_d[j];
  n[2] = kz_d[k];

  gradphix_d[idx].x = -1.0*n[0]*phi_d[idx].y;  
  gradphix_d[idx].y = n[0]*phi_d[idx].x;  
  gradphiy_d[idx].x = -1.0*n[1]*phi_d[idx].y;  
  gradphiy_d[idx].y = n[1]*phi_d[idx].x;  
  gradphiz_d[idx].x = -1.0*n[2]*phi_d[idx].y;  
  gradphiz_d[idx].y = n[2]*phi_d[idx].x;  

  __syncthreads();    

}


__global__ void ComputeDrivForce(cuDoubleComplex *comp_d, 
                                 cuDoubleComplex *dfdphi_d,
                                 cuDoubleComplex *gradphix_d, 
                                 cuDoubleComplex *gradphiy_d,
                                 cuDoubleComplex *gradphiz_d, 
                                 double f0AVminv_d, double f0BVminv_d, 
                                 double c_beta_eq_d, double c_alpha_eq_d, 
                                 double diffusivity_d, double w_d, int ny_d,
                                 int nz_d)
{

  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  double  interp_phi, interp_prime, g_prime;
  double  ctemp, etemp;
  double  f_alpha, f_beta, mubar;
  double  A_by_B, B_by_A;
  double  calpha, cbeta;

  A_by_B = (f0AVminv_d)/(f0BVminv_d);
  B_by_A = (f0BVminv_d)/(f0AVminv_d);
   
  ctemp  = comp_d[idx].x;
  etemp  = dfdphi_d[idx].x;
   
  interp_phi   = etemp * etemp * etemp * 
                (6.0 * etemp * etemp - 15.0 * etemp + 10.0);
  interp_prime = 30.0 * etemp * etemp * pow((1.0 - etemp), 2.0);  
  g_prime      = 2.0 * etemp * (1.0 - etemp) * (1.0 - 2.0 * etemp);

  calpha       = (ctemp - interp_phi * 
                 ((c_beta_eq_d) - (c_alpha_eq_d) * A_by_B))/
                 (interp_phi*A_by_B + (1.0-interp_phi));

  cbeta        = (ctemp + (1.0 - interp_phi) * 
                 (B_by_A * (c_beta_eq_d) - (c_alpha_eq_d)))/
                 (interp_phi + B_by_A*(1.0 - interp_phi));

  comp_d[idx].x  = calpha*(1.0 - interp_phi) + 
                   cbeta*interp_phi; 
  comp_d[idx].y  = 0.0;

  f_alpha      = (f0AVminv_d)*(calpha - (c_alpha_eq_d))* 
                 (calpha - (c_alpha_eq_d));
  f_beta       = (f0BVminv_d)*(cbeta - (c_beta_eq_d))* 
                 (cbeta - (c_beta_eq_d));
   
  gradphix_d[idx].x = (diffusivity_d)*interp_prime* 
                     (calpha - cbeta) * gradphix_d[idx].x;
  gradphix_d[idx].y = (diffusivity_d)*interp_prime* 
                     (calpha - cbeta) * gradphix_d[idx].y;
  gradphiy_d[idx].x = (diffusivity_d)*interp_prime* 
                     (calpha - cbeta) * gradphiy_d[idx].x;
  gradphiy_d[idx].y = (diffusivity_d)*interp_prime* 
                     (calpha - cbeta) * gradphiy_d[idx].y;
  gradphiz_d[idx].x = (diffusivity_d)*interp_prime* 
                     (calpha - cbeta) * gradphiz_d[idx].x;
  gradphiz_d[idx].y = (diffusivity_d)*interp_prime* 
                     (calpha - cbeta) * gradphiz_d[idx].y;
   
  mubar = 2.0 * (f0BVminv_d) * (cbeta - (c_beta_eq_d));

  dfdphi_d[idx].x = interp_prime*(f_beta - f_alpha + 
                    (calpha - cbeta)*mubar) + 
                    (w_d)*g_prime; 
  dfdphi_d[idx].y = 0.0;

  __syncthreads();

}
/*
__global__ void  ComputeDfdc(cuDoubleComplex *dfdc_d, 
                             cuDoubleComplex *varmobx_d, 
                             cuDoubleComplex *varmoby_d, 
                             cuDoubleComplex *varmobz_d,
                             int nx_d, int ny_d, int nz_d, 
                             double *kx_d, double *ky_d, 
                             double *kz_d)
{

  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));
  double  n[3];

  n[0] = kx_d[i];
  n[1] = ky_d[j];
  n[2] = kz_d[k];

  dfdc_d[idx].x = -1.0*(n[0]*varmobx_d[idx].y + 
                        n[1]*varmoby_d[idx].y +
                        n[2]*varmobz_d[idx].y); 
 
  dfdc_d[idx].y = (n[0]*varmobx_d[idx].x + 
                   n[1]*varmoby_d[idx].x +
                   n[2]*varmobz_d[idx].x);
  
  __syncthreads();    

}
*/
__global__ void Update_comp_phi (cuDoubleComplex *comp_d, 
                                 cuDoubleComplex *phi_d, 
                                 cuDoubleComplex *dfdphi_d,
                                 cuDoubleComplex *dfeldphi_d, 
                                 cuDoubleComplex *gradphix_d,
                                 cuDoubleComplex *gradphiy_d, 
                                 cuDoubleComplex *gradphiz_d, double *kx_d, 
                                 double *ky_d, double *kz_d, double dt_d,
                                 double diffusivity_d, double kappa_phi_d, 
                                 double relax_coeff_d, int elast_int_d, 
                                 int nx_d, int ny_d, int nz_d)
{
  
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  double           kpow2, lhs, lhse;
  cuDoubleComplex  rhs, rhse; 
 
  kpow2 = kx_d[i]*kx_d[i] + ky_d[j]*ky_d[j] + kz_d[k]*kz_d[k];

  lhs = 1.0 + (diffusivity_d)*kpow2*(dt_d);   

  rhs.x = comp_d[idx].x + (dt_d)*(-1.0*(kx_d[i]*gradphix_d[idx].y +
                                        ky_d[j]*gradphiy_d[idx].y +
                                        kz_d[k]*gradphiz_d[idx].y));
  rhs.y = comp_d[idx].y + (dt_d)*(kx_d[i]*gradphix_d[idx].x +
                                  ky_d[j]*gradphiy_d[idx].x + 
                                  kz_d[k]*gradphiz_d[idx].x);

  //rhs.x = comp_d[idx].x + (dt_d)*dfdc_d[idx].x;
  //rhs.y = comp_d[idx].y + (dt_d)*dfdc_d[idx].y;
  comp_d[idx].x = rhs.x/lhs; 
  comp_d[idx].y = rhs.y/lhs;
  
  lhse = 1.0 + 2.0*(relax_coeff_d)*(kappa_phi_d)*kpow2*(dt_d);

  if ((elast_int_d) == 1 ){

     rhse.x  = phi_d[idx].x - (relax_coeff_d)*(dt_d)*
               (dfdphi_d[idx].x + dfeldphi_d[idx].x);
     rhse.y  = phi_d[idx].y - (relax_coeff_d)*(dt_d)*
               (dfdphi_d[idx].y + dfeldphi_d[idx].y);

  }
  else{

     rhse.x  = phi_d[idx].x - (relax_coeff_d)*(dt_d)*
              (dfdphi_d[idx].x);
     rhse.y  = phi_d[idx].y - (relax_coeff_d)*(dt_d)*
              (dfdphi_d[idx].y);

  }

  phi_d[idx].x = rhse.x/lhse;
  phi_d[idx].y = rhse.y/lhse;

  dfdphi_d[idx].x = phi_d[idx].x;
  dfdphi_d[idx].y = phi_d[idx].y;
 

}

__global__ void Normalize_elast(cufftComplex *x, double sizescale_d, int ny_d,
                          int nz_d)
{

  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));
 
  x[idx].x = (float)(x[idx].x * (sizescale_d));
  x[idx].y = (float)(x[idx].y * (sizescale_d));

}
__global__ void Normalize(cuDoubleComplex *x, double sizescale_d, int ny_d,
                          int nz_d)
{

  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));
 
  x[idx].x = x[idx].x * (sizescale_d);
  x[idx].y = x[idx].y * (sizescale_d);

}

__global__ void SaveReal(double *temp, cuDoubleComplex *x, int ny_d,
                         int nz_d)
{
  
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  temp[idx] = x[idx].x; 

}
__global__ void Find_err_matrix(double *temp, cuDoubleComplex *comp_d, 
                                int ny_d, int nz_d)
{
  
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int k = threadIdx.z + blockIdx.z*blockDim.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  temp[idx] = fabs(comp_d[idx].x - temp[idx]);
  
}    
void Evolve(void)
{
  void Output_Conf (int steps);
  void Calc_uzero(void);
  void InhomElast(void);
  void HomElast(void);
  int       loop_condition, count;
  double    *kx, *ky, *kz;
  double    *tempreal_d;
  double    maxerror, *maxerr_d;
  double    f0AVminv, f0BVminv;
  void      *t_storage = NULL;
  size_t    t_storage_bytes = 0;
  size_t    complex_size, double_size;
  cufftDoubleComplex    comp_at_corner;

  cub::DeviceReduce::Max(t_storage, t_storage_bytes, tempreal_d, maxerr_d,
                         nx*ny*nz);

  complex_size = nx*ny*nz*sizeof(cuDoubleComplex);
  double_size  = nx*ny*nz*sizeof(double);

  f0AVminv = f0A * (1.0/Vm) ;
  f0BVminv = f0B * (1.0/Vm) ;

  checkCudaErrors(cudaMalloc((void**)&tempreal_d, double_size));
  checkCudaErrors(cudaMalloc((void**)&maxerr_d, sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&S11_d, sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&S12_d, sizeof(double)));
  checkCudaErrors(cudaMalloc((void**)&S44_d, sizeof(double)));
  checkCudaErrors(cudaMalloc(&t_storage, t_storage_bytes));

  checkCudaErrors(cudaMemcpy(comp, comp_d, complex_size,
        cudaMemcpyDeviceToHost));

  SaveReal<<< Gridsize, Blocksize >>>(tempreal_d, comp_d, ny, nz);

 //Fourier vectors: defined in host and copied to device memory 
  kx = (double*) malloc(nx*sizeof(double));
  ky = (double*) malloc(ny*sizeof(double));
  kz = (double*) malloc(nz*sizeof(double));

  for (int i = 0 ; i < nx ; i++ ) {
    if (i < nx/2) 
      kx[i] = (double) i * dkx;
    else 
      kx[i] = (double)(i-nx) * dkx;
  } 

  for (int j = 0; j < ny; j++){
    if (j < ny/2)
      ky[j] = (double)j * dky;
    else
      ky[j] = (double)(j-ny) * dky;
  }

  for (int k = 0; k < nz; k++){
    if (k < nz/2)
      kz[k] = (double)k * dkz;
    else
      kz[k] = (double)(k-nz) * dkz;
  }

  checkCudaErrors(cudaMemcpy(kx_d, kx, nx*sizeof(double),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ky_d, ky, ny*sizeof(double),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(kz_d, kz, nz*sizeof(double),
        cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc((void**)&gradphix_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&gradphiy_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&gradphiz_d, complex_size));

  free(kx);
  free(ky);
  free(kz);

  if (inhom != 1){

    S11 = ((Chom11)+(Chom12))/((Chom11)*(Chom11) + (Chom11)*(Chom12) -
          2.0*(Chom12)*(Chom12));
    S12 = (-1.0*(Chom12))/((Chom11)*(Chom11) + (Chom11)*(Chom12) -
          2.0*(Chom12)*(Chom12));
    S44 = 1.0/(Chom44);

    checkCudaErrors(cudaMemcpy(S11_d, &S11, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(S12_d, &S12, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(S44_d, &S44, sizeof(double),
          cudaMemcpyHostToDevice));

  }

  if (cufftExecZ2Z(plan, phi_d, phi_d, CUFFT_FORWARD)!= CUFFT_SUCCESS)
    printf("fft failed");

  cudaGetLastError();

  printf("%s\n",cudaGetErrorString(Err));

  iteration = 1;
  loop_condition = 1;

  //Time loop
  for (count = initcount; count <= num_steps; count++) {
  
    if (((count % t_prof1)==0 && count <= numsteps_prof1) ||
        ((count % t_prof2)==0 && count  > numsteps_prof1) ||  
         (count == num_steps) || 
         (loop_condition == 0)) {

      printf ("total_time=%le\n", sim_time);
      printf ("writing configuration to file!\n");

      checkCudaErrors(cudaMemcpy(comp, comp_d, complex_size,
            cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(dfdphi, dfdphi_d, complex_size,
            cudaMemcpyDeviceToHost));

      Output_Conf(count);

    }
 
    if (count > num_steps || loop_condition == 0)
      break;
   
    printf("Iteration No: %d\n",iteration);

      //Finding elastic driving force in real space
    if (elast_int == 1 && count >= time_elast ){

      if (inhom == 1){
        if (count == initcount)
        Calc_uzero();
        InhomElast();
      }
      else
        HomElast();
    }


    ComputeGradphi<<< Gridsize, Blocksize >>>(kx_d, ky_d, kz_d, 
                                       nx, ny, nz, 
                                       phi_d, gradphix_d, gradphiy_d, 
                                       gradphiz_d);

    cufftExecZ2Z(plan, gradphix_d, gradphix_d, CUFFT_INVERSE);
    cufftExecZ2Z(plan, gradphiy_d, gradphiy_d, CUFFT_INVERSE);
    cufftExecZ2Z(plan, gradphiz_d, gradphiz_d, CUFFT_INVERSE);
    
    Normalize<<<Gridsize,Blocksize >>>(gradphix_d, sizescale, ny, nz);
    Normalize<<<Gridsize,Blocksize >>>(gradphiy_d, sizescale, ny, nz);
    Normalize<<<Gridsize,Blocksize >>>(gradphiz_d, sizescale, ny, nz);
    
    ComputeDrivForce<<<Gridsize, Blocksize >>>(comp_d, dfdphi_d, 
                                  gradphix_d, gradphiy_d, gradphiz_d, 
                                  f0AVminv, f0BVminv, c_beta_eq, 
                                  c_alpha_eq, diffusivity, 
                                  w, ny, nz);

    if (cufftExecZ2Z(plan,gradphix_d, gradphix_d,CUFFT_FORWARD) != CUFFT_SUCCESS)
       printf("fft failed\n");
    if (cufftExecZ2Z(plan,gradphiy_d, gradphiy_d,CUFFT_FORWARD) != CUFFT_SUCCESS)
       printf("fft failed\n");
    if (cufftExecZ2Z(plan,gradphiz_d, gradphiz_d,CUFFT_FORWARD) != CUFFT_SUCCESS)
       printf("fft failed\n");
    
    cufftExecZ2Z(plan, comp_d,     comp_d, CUFFT_FORWARD);
    cufftExecZ2Z(plan, dfdphi_d, dfdphi_d, CUFFT_FORWARD);
    
    if (elast_int == 1 && count > time_elast)
      cufftExecZ2Z(plan, dfeldphi_d, dfeldphi_d, CUFFT_FORWARD);
    
    Update_comp_phi<<< Gridsize, Blocksize >>>(comp_d, phi_d, dfdphi_d,
                                 dfeldphi_d, gradphix_d, gradphiy_d, 
                                 gradphiz_d, kx_d, ky_d, kz_d, dt,
                                 diffusivity, kappa_phi, relax_coeff,
                                 elast_int, nx, ny, nz);

    cufftExecZ2Z(plan, comp_d,     comp_d, CUFFT_INVERSE);
    cufftExecZ2Z(plan, dfdphi_d, dfdphi_d, CUFFT_INVERSE);

    Normalize<<< Gridsize, Blocksize >>>(comp_d,   sizescale, ny, nz);
    Normalize<<< Gridsize, Blocksize >>>(dfdphi_d, sizescale, ny, nz);

    //save difference between new and old composition in tempreal_d
    Find_err_matrix<<< Gridsize, Blocksize >>>(tempreal_d, comp_d, ny, nz);

    cudaMemcpy(&comp_at_corner,comp_d,sizeof(cufftDoubleComplex),
                cudaMemcpyDeviceToHost);

    //if (fabs(Re(comp_at_corner) - c0) >= 1.0e-01)
    //{
    //	printf("Growth condition has vanished!!!!\n");
    //    exit(0);
    //}

    cub::DeviceReduce::Max(t_storage, t_storage_bytes, tempreal_d, maxerr_d, 
                           nx*ny*nz);       
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(&maxerror, maxerr_d, sizeof(double),
          cudaMemcpyDeviceToHost));

    if(maxerror <= Tolerance){
      printf("Microstructure converged\n");
      loop_condition = 0;      
    }
    sim_time = sim_time + dt;

    SaveReal <<<Gridsize,Blocksize>>> (tempreal_d, comp_d, ny, nz);

    iteration = iteration + 1;
  }//time loop ends

  cudaFree(tempreal_d);
  cudaFree(maxerr_d);
  cudaFree(t_storage);
  cudaFree(S11_d);
  cudaFree(S12_d);
  cudaFree(S44_d);
  cudaFree(gradphix_d);
  cudaFree(gradphiy_d);
  cudaFree(gradphiz_d);
}

#include "out_conf.cu"
#include "calc_uzero.cu"
#include "inhomelast.cu"
#include "homelast.cu"
