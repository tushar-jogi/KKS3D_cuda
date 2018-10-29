__global__ void Compute_eigsts_hom(cufftDoubleComplex *eigsts00, 
                                   cufftDoubleComplex *eigsts10, 
                                   cufftDoubleComplex *eigsts20, 
                                   cufftDoubleComplex *dfdphi_d, 
                                   double Chom11_d, double Chom12_d, 
                                   double Chom44_d, double epszero_d, 
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

  eigsts00[idx].x = (Chom11_d)*eig11 + 
                    (Chom12_d)*eig22 +
                    (Chom12_d)*eig33 ;     

  eigsts10[idx].x = (Chom12_d)*eig11 + 
                    (Chom11_d)*eig22 +
                    (Chom12_d)*eig33 ;     

  eigsts20[idx].x = (Chom12_d)*eig11 + 
                    (Chom12_d)*eig22 +
                    (Chom11_d)*eig33 ;

  eigsts00[idx].y = 0.0;    
  eigsts10[idx].y = 0.0;    
  eigsts20[idx].y = 0.0;    

}

__global__ void Initialize_disp(cufftDoubleComplex *ux_d, 
                                cufftDoubleComplex *uy_d,
                                cufftDoubleComplex *uz_d, int ny_d, int nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (nz_d)*(j + i*(ny_d));

  ux_d[idx].x = 0.0;
  uy_d[idx].x = 0.0;
  uz_d[idx].x = 0.0;

  ux_d[idx].y = 0.0;
  uy_d[idx].y = 0.0;
  uz_d[idx].y = 0.0;

}

__global__ void Compute_uzero(int nx_d, int ny_d, int nz_d, 
                          cufftDoubleComplex *ux_d, cufftDoubleComplex *uy_d,
                          cufftDoubleComplex *uz_d, double dkx, double dky,
                          double dkz, cufftDoubleComplex *eigsts00, 
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
  
  if (i < nx_d/2) 
     nk[0] = (double) i * dkx;
  else 
     nk[0] = (double)(i-nx_d) * dkx;

  if (j < ny_d/2) 
     nk[1] = (double) j * dky;
  else 
     nk[1] = (double)(j-ny_d) * dky;

  if (k < nz_d/2) 
     nk[2] = (double) j * dkz;
  else 
     nk[2] = (double)(j-nz_d) * dkz;

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

}

void Calc_uzero(void){


  
   //Eigen stress saved in ux_d. uy_d and uz_d 
   Compute_eigsts_hom<<< Gridsize,Blocksize >>>(ux_d, uy_d, uz_d, 
                                                dfdphi_d, 
                                                Chom11, Chom12, Chom44, 
                                                epszero, ny, nz);

 /************************************************************
  *          Take eigenstress component to fourier space     * 
  ************************************************************/

   cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_FORWARD);
   cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_FORWARD);
   cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_FORWARD);
 
/************************************************************
*                Initializing displacments                  *
*************************************************************/
   //Initialize_disp<<< Gridsize, Blocksize>>>(ux_d, uy_d, uz_d, ny, nz);
/**********************************************************
 *                 Zeroth order displacement              *
 **********************************************************/ 
   Compute_uzero<<< Gridsize, Blocksize >>>(nx, ny, nz, ux_d, uy_d, uz_d, dkx, 
                          dky, dkz, ux_d, uy_d, uz_d,
                          Chom11, Chom12, Chom44);

}
