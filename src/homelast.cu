/**************************************************************************
   
   Description: Elasticity solver for homogeneous modulus. 
                Major variable is ux_d, uy_d and uz_d.
                As a result to dilatational misfit approximation,
                elastic driving force requires normal components of
                strains. 
		
   Date: 24/09/2018 

************************************************************************/
#define TOLERENCE 1.0e-06

__global__ void Compute_eigstr_hom(cufftDoubleComplex *eigstr0, 
                                   cufftDoubleComplex *eigstr1, 
                                   cufftDoubleComplex *eigstr2, 
                                   cufftDoubleComplex *dfdphi_d, 
                                   double epszero_d, 
                                   int ny_d, int nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  double eig11, eig22, eig33, hphi, e_temp;
  
  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  eig11 = epszero_d * hphi; 
  eig22 = epszero_d * hphi; 
  eig33 = epszero_d * hphi; 

  eigstr0[idx].x = eig11; 

  eigstr1[idx].x = eig22; 

  eigstr2[idx].x = eig33; 

  eigstr0[idx].y = 0.0;    
  eigstr1[idx].y = 0.0;    
  eigstr2[idx].y = 0.0;    

}

__global__ void Compute_Sij_hom(double *Cavg11, double *Cavg12, double *Cavg44,
                            double *S11_d, double *S12_d, double *S44_d)
{
  *S11_d = ((*Cavg11) + (*Cavg12))/((*Cavg11)*(*Cavg11) + (*Cavg11)*(*Cavg12)-
           2.0*(*Cavg12)*(*Cavg12));
  *S12_d = (-1.0*(*Cavg12))/((*Cavg11)*(*Cavg11) + (*Cavg11)*(*Cavg12) -
           2.0*(*Cavg12)*(*Cavg12));
  *S44_d = 1.0/(*Cavg44);

}

/*__global__ void Compute_uzero(int ny_d, int nz_d, 
                          cufftDoubleComplex *ux_d, cufftDoubleComplex *uy_d,
                          cufftDoubleComplex *uz_d, double *kx_d, double *ky_d,
                          double *kz_d, cufftDoubleComplex *eigsts00, 
                          cufftDoubleComplex *eigsts10, 
                          cufftDoubleComplex *eigsts20,
                          double Chom11_d, double Chom12_d, double Chom44_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  double               adjomega[6], det_omega, invomega_v[6];
  double               nk[3];
  double               omega[6];
  cufftDoubleComplex   eig_v[3], fk10, fk20, fk30;
  
  nk[0] = (double)kx_d[i];
  nk[1] = (double)ky_d[j];
  nk[2] = (double)kz_d[k];

  invomega_v[0] = (Chom11_d)*nk[0]*nk[0] + (Chom44_d)*nk[1]*nk[1] +
                  (Chom44_d)*nk[2]*nk[2];
  invomega_v[1] = (Chom44_d)*nk[0]*nk[0] + (Chom11_d)*nk[1]*nk[1] +
                  (Chom44_d)*nk[2]*nk[2];
  invomega_v[2] = (Chom44_d)*nk[0]*nk[0] + (Chom44_d)*nk[1]*nk[1] +
                  (Chom11_d)*nk[2]*nk[2];
  invomega_v[3] = ((Chom12_d) + (Chom44_d))*nk[1]*nk[2];
  invomega_v[4] = ((Chom12_d) + (Chom44_d))*nk[0]*nk[2];
  invomega_v[5] = ((Chom12_d) + (Chom44_d))*nk[0]*nk[1];

  det_omega = invomega_v[0]*(invomega_v[1]*invomega_v[2] -
                             invomega_v[3]*invomega_v[3])-
              invomega_v[5]*(invomega_v[5]*invomega_v[2] -
                             invomega_v[4]*invomega_v[3])+
              invomega_v[4]*(invomega_v[5]*invomega_v[3] -
                             invomega_v[4]*invomega_v[1]);

  adjomega[0] = (invomega_v[1]*invomega_v[2]-
                 invomega_v[3]*invomega_v[3]);
  adjomega[1] = (invomega_v[0]*invomega_v[2]-
                 invomega_v[4]*invomega_v[4]);
  adjomega[2] = (invomega_v[0]*invomega_v[1]-
                 invomega_v[5]*invomega_v[5]);
  adjomega[3] =-(invomega_v[0]*invomega_v[3]-
                 invomega_v[4]*invomega_v[5]);
  adjomega[4] = (invomega_v[5]*invomega_v[3]-
                 invomega_v[4]*invomega_v[1]);
  adjomega[5] =-(invomega_v[5]*invomega_v[2]-
                 invomega_v[4]*invomega_v[3]);

  if (fabs(det_omega) > 1.0e-06){
     omega[0] = (1.0/det_omega)*adjomega[0];
     omega[1] = (1.0/det_omega)*adjomega[1];
     omega[2] = (1.0/det_omega)*adjomega[2];
     omega[3] = (1.0/det_omega)*adjomega[3];
     omega[4] = (1.0/det_omega)*adjomega[4];
     omega[5] = (1.0/det_omega)*adjomega[5];
  }

  else{
     omega[0] = 0.0;
     omega[1] = 0.0;
     omega[2] = 0.0;
     omega[3] = 0.0;
     omega[4] = 0.0;
     omega[5] = 0.0;
  }

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

}*/

__global__ void Compute_perstr(int ny, int nz, cuDoubleComplex *ux_d, cuDoubleComplex *uy_d, 
                               cuDoubleComplex *uz_d, double *kx_d, double *ky_d, double *kz_d)
{
	
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (nz)*(j + i*(ny));

  ux_d[idx].x = -1.0*kx_d[i]*ux_d[idx].y;
  uy_d[idx].x = -1.0*ky_d[j]*uy_d[idx].y;
  uz_d[idx].x = -1.0*kz_d[k]*uz_d[idx].y;

  ux_d[idx].y = kx_d[i]*ux_d[idx].x;
  uy_d[idx].y = ky_d[j]*uy_d[idx].x;
  uz_d[idx].y = kz_d[k]*uz_d[idx].x;
   
  
}

__global__ void Find_volumeAvg_eigstr(double *x, double sizescale)
{
   x[0] = (x[0]*sizescale);
   x[1] = (x[1]*sizescale);
   x[2] = (x[2]*sizescale);
}

__global__ void Compute_dfeldphi_hom(cuDoubleComplex *ux_d,
                                     cuDoubleComplex *uy_d,
                                     cuDoubleComplex *uz_d,
                                     cuDoubleComplex *dfdphi_d,
                                     cuDoubleComplex *dfeldphi_d, 
                   double Chom11, double Chom12, double Chom44, 
                   double *hom_strain_v, double epszero_d,
                   int ny_d, int nz_d)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (nz_d)*(j + i*(ny_d));

   double hphi, hphi_p, e_temp, str_v[6], estr[3], hstr[6];
 
   e_temp = dfdphi_d[idx].x;
   hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp -15.0*e_temp +10.0);
   hphi_p = (30.0*e_temp*e_temp*(1.0-e_temp)*(1.0-e_temp));           
   

   str_v[0] = ux_d[idx].x;
   str_v[1] = uy_d[idx].x;
   str_v[2] = uz_d[idx].x;

   estr[0]  = (epszero_d)*hphi;
   estr[1]  = (epszero_d)*hphi;
   estr[2]  = (epszero_d)*hphi;
 
   hstr[0]  = hom_strain_v[0];
   hstr[1]  = hom_strain_v[1];
   hstr[2]  = hom_strain_v[2];

   dfeldphi_d[idx].x = 
                Chom11*
                (hstr[0]+str_v[0]-estr[0])*
                (epszero_d)*hphi_p +
                Chom11*
                (hstr[1]+str_v[1]-estr[1])*
                (epszero_d)*hphi_p +
                Chom11*
                (hstr[2]+str_v[2]-estr[2])*
                (epszero_d)*hphi_p +
                Chom12*
                (hstr[1]+str_v[1]-estr[1])*
                (epszero_d)*hphi_p +
                Chom12*
                (hstr[1]+str_v[1]-estr[1])*
                (epszero_d)*hphi_p +
                Chom12*
                (hstr[0]+str_v[0]-estr[0])*
                (epszero_d)*hphi_p +
                Chom12*
                (hstr[0]+str_v[0]-estr[0])*
                (epszero_d)*hphi_p +
                Chom12*
                (hstr[2]+str_v[2]-estr[2])*
                (epszero_d)*hphi_p + 
                Chom12*
                (hstr[2]+str_v[2]-estr[2])*
                (epszero_d)*hphi_p;

     dfeldphi_d[idx].y = 0.0; 
} 

__global__ void Find_hom_strain(double *hom_strain_v, double *sigappl_v_d, double *S11_d, 
                                double *S12_d, double *S44_d)
{
	hom_strain_v[0] += (*S11_d)*(sigappl_v_d[0])+ (*S12_d)*(sigappl_v_d[1] + sigappl_v_d[2]);
	hom_strain_v[1] += (*S11_d)*(sigappl_v_d[1])+ (*S12_d)*(sigappl_v_d[0] + sigappl_v_d[2]);
	hom_strain_v[2] += (*S11_d)*(sigappl_v_d[2])+ (*S12_d)*(sigappl_v_d[0] + sigappl_v_d[1]);
} 

void HomElast(void){

   int              complex_size;
   void             *t_storage = NULL;
   size_t           t_storage_bytes = 0;
   double           *hom_strain_v;
   cublasStatus_t    stat;
   checkCudaErrors(cudaMalloc((void**)&hom_strain_v, 3*sizeof(double)));
   checkCudaErrors(cudaMalloc(&t_storage, t_storage_bytes));

   /*cub::DeviceReduce::Sum(t_storage, t_storage_bytes, ux_d,
                          hom_strain_v[0], nx*ny*nz);*/

   complex_size = sizeof(cufftDoubleComplex)*nx*ny*nz;
 
   //save eigen strain ux_d, uy_d and uz_d 
   Compute_eigstr_hom<<<Gridsize,Blocksize>>>(ux_d, uy_d, uz_d, 
                                              dfdphi_d, epszero, ny, nz);
   
   stat = cublasCreate(&blas_handle); 
   if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      exit(EXIT_FAILURE);
   }

   stat = cublasDzasum(blas_handle, nx*ny*nz, ux_d, 1, &hom_strain_v[0]);
   stat = cublasDzasum(blas_handle, nx*ny*nz, uy_d, 1, &hom_strain_v[1]);
   stat = cublasDzasum(blas_handle, nx*ny*nz, uz_d, 1, &hom_strain_v[2]);

   if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS sum failed\n");
      exit(EXIT_FAILURE);
   }
   //Sum of eigen strain 
   /*cub::DeviceReduce::Sum(t_storage, t_storage_bytes, ux_d,
                          hom_strain_v[0], nx*ny*nz);
   cub::DeviceReduce::Sum(t_storage, t_storage_bytes, uy_d,
                          hom_strain_v[1], nx*ny*nz);
   cub::DeviceReduce::Sum(t_storage, t_storage_bytes, uz_d,
                          hom_strain_v[2], nx*ny*nz);
   */ 
   Find_volumeAvg_eigstr<<<1,1>>>(hom_strain_v, sizescale);

   Find_hom_strain<<<1,1>>>(hom_strain_v, sigappl_v_d, S11_d, S12_d, S44_d); 

   //Save eigen stress in ux_d, uy_d and uz_d 
   Compute_eigsts_hom<<<Gridsize,Blocksize >>>(ux_d, uy_d, uz_d, 
                                               dfdphi_d, 
                                               Chom11, Chom12, Chom44, 
                                               epszero, ny, nz);


  /************************************************************
   *          Take eigenstress component to fourier space     * 
   ************************************************************/
   cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_FORWARD);
   cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_FORWARD);
   cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_FORWARD);
 
  /**********************************************************
   *                 Zeroth order displacement              *
   **********************************************************/ 
   Compute_uzero<<< Gridsize, Blocksize >>>(ny, nz, 
                          ux_d, uy_d, uz_d, kx_d, 
                          ky_d, kz_d, ux_d, uy_d, uz_d,
                          Chom11, Chom12, Chom44);

   
   Compute_perstr<<<Gridsize, Blocksize>>>(ny, nz, ux_d, uy_d, uz_d, kx_d, ky_d, kz_d);

   cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_INVERSE);
   cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_INVERSE);
   cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_INVERSE);

   Normalize<<<Gridsize,Blocksize >>>(ux_d, sizescale, ny, nz); 
   Normalize<<<Gridsize,Blocksize >>>(uy_d, sizescale, ny, nz); 
   Normalize<<<Gridsize,Blocksize >>>(uz_d, sizescale, ny, nz); 

   Compute_dfeldphi_hom<<< Gridsize, Blocksize>>>
                      (ux_d, uy_d, uz_d, dfdphi_d, dfeldphi_d, Chom11, Chom12,
                       Chom44, hom_strain_v, epszero, ny, nz);

   cublasDestroy(blas_handle);

}
