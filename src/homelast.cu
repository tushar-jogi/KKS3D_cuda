/**************************************************************************

   last update: Memory is reduced by computing Ctotal, Cinhom and
                eigenstrain on the fly.
   Date: 24/09/2018 

************************************************************************/
#define TOLERENCE 1.0e-06

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
__global__ void Compute_Ctotal(cuDoubleComplex *dfdphi_d, int ny_d, 
                              int nz_d, double *Ctotal, double Chom11_d, 
                              double Chet11_d)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (nz_d)*(j + i*(ny_d));

   double hphi, e_temp;

   e_temp = Re(dfdphi_d[idx]);
   hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp  + 10.0);

   Ctotal[idx]  =  (Chom11_d) + (Chet11_d*(2.0*hphi - 1.0));
   
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

__global__ void Compute_uzero(int ny_d, int nz_d, 
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

}
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
__global__ void Average(double *x, double sizescale)
{
   *x = (double)(*x*(sizescale));
}

void HomElast(void){

  int complex_size;

  checkCudaErrors(cudaMalloc((void**)&hom_strain_v, 3*sizeof(double)));
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, ux_d,
                         hom_strain_v[0], nx*ny*nz);
  complex_size = sizeof(cufftDoubleComplex)*nx*ny*nz;
   
  Compute_eigsts_hom<<< Gridsize,Blocksize >>>(ux_d, uy_d, uz_d, 
                                               dfdphi_d, 
                                               Chom11, Chom12, Chom44, 
                                               epszero, ny, nz);

  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, ux_d,
                         hom_strain_v[0], nx*ny*nz);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, uy_d,
                         hom_strain_v[1], nx*ny*nz);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, uz_d,
                         hom_strain_v[2], nx*ny*nz);
    
  Average<<<1,1>>>(hom_strain_v[0], sizescale);
  Average<<<1,1>>>(hom_strain_v[1], sizescale);
  Average<<<1,1>>>(hom_strain_v[2], sizescale);
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
                      (ux_d, uy_d, uz_d, dfdphi_d, dfeldphi_d, Chom11_d, Chom12_d,
                       Chom44_d, hom_strain_v, epszero_d,
                       ny_d, nz_d);

}

__global__ void Compute_dfeldphi_hom(cuDoubleComplex *ux_d,
                                     cuDoubleComplex *uy_d,
                                     cuDoubleComplex *uy_d,
                                     cuDoubleComplex *dfdphi_d,
                                     cuDoubleComplex *dfeldphi_d, 
                                     double Chom11_d, double Chom12_d, double Chom44_d, 
                                     double *hom_strain_v, double epszero_d,
                                     int ny_d, int nz_d)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (nz_d)*(j + i*(ny_d));

   double hphi, hphi_p, e_temp, str_v[6], estr[3], hstr[6];
 
   e_temp = dfdphi_d[idx].x;
   hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);
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
                (epszero_d)*hphi_p
               ;

     dfeldphi_d[idx].y = 0.0; 
} 


  Compute_homstr<<<1,1>>>(hom_strain_v, S11_d, S12_d, S44_d, sigappl_v_d,
                          avgeigsts0, avgeigsts1, avgeigsts2, 
                          avgpersts0, avgpersts1, avgpersts2, 
                          avgpersts3, avgpersts4, avgpersts5);

  Compute_dfeldphi<<<Gridsize, Blocksize>>>(str_v0_d, str_v1_d, str_v2_d,
                                            str_v3_d, str_v4_d, str_v5_d, 
                                            dfdphi_d, dfeldphi_d, Chom11,
                                            Chom12, Chom44, Chet11,
                                            Chet12, Chet44, hom_strain_v, 
                                            epszero, ny, nz);
