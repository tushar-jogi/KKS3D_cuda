/**************************************************************************

   last update: Memory is reduced by computing Ctotal, Cinhom and
                eigenstrain on the fly.
   Date: 24/09/2018 

************************************************************************/
#define TOLERENCE 1.0e-06

__global__ void Compute_Ctotal(cuDoubleComplex *dfdphi_d, int *ny_d, 
                              int *nz_d, double *Ctotal, double *Chom11_d, 
                              double *Chet11_d)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (*nz_d)*(j + i*(*ny_d));

   double hphi, e_temp;

   e_temp = Re(dfdphi_d[idx]);
   hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp  + 10.0);

   Ctotal[idx]  =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
   
}


__global__ void Compute_Sij(double *Cavg11, double *Cavg12, double *Cavg44,
                            double *S11_d, double *S12_d, double *S44_d)
{
  *S11_d = ((*Cavg11) + (*Cavg12))/((*Cavg11)*(*Cavg11) + (*Cavg11)*(*Cavg12)-
           2.0*(*Cavg12)*(*Cavg12));
  *S12_d = (-1.0*(*Cavg12))/((*Cavg11)*(*Cavg11) + (*Cavg11)*(*Cavg12) -
           2.0*(*Cavg12)*(*Cavg12));
  *S44_d = 1.0/(*Cavg44);

}

__global__ void Compute_Cinhom(double *Cinhom11, double *Cinhom12, 
                               double *Cinhom44, double *Ctotal11, 
                               double *Ctotal12, double *Ctotal44,
                               double *Chom11_d, double *Chom12_d,
                               double *Chom44_d, int *ny_d, int *nz_d)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (*nz_d)*(j + i*(*ny_d));
    
 
   Cinhom11[idx] = Ctotal11[idx] - (*Chom11_d); 
   Cinhom12[idx] = Ctotal12[idx] - (*Chom12_d); 
   Cinhom44[idx] = Ctotal44[idx] - (*Chom44_d); 
}

/*__global__ void Compute_eigstr(cuDoubleComplex *dfdphi_d, double *eigstr0,
                               double *eigstr1, double *eigstr2, 
                               double *epszero_d)
{
   int idx = (threadIdx.x + threadIdx.y * blockDim.x) +
            (blockIdx.x*blockDim.x*blockDim.y);

   double hphi, e_temp;

   e_temp = dfdphi_d[idx].x;

   hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

   eigstr0[idx] = (*epszero_d)*hphi;
   eigstr1[idx] = (*epszero_d)*hphi;
   eigstr2[idx] = (*epszero_d)*hphi;

}*/

__global__ void Compute_perstr(int *ny_d, int *nz_d, double *kx_d, 
                               double *ky_d, double *kz_d, 
                         cuDoubleComplex *unewx_d, cuDoubleComplex *unewy_d,
                         cuDoubleComplex *unewz_d, cuDoubleComplex *str_v0_d,
                         cuDoubleComplex *str_v1_d, cuDoubleComplex *str_v2_d,
                         cuDoubleComplex *str_v3_d, cuDoubleComplex *str_v4_d,
                         cuDoubleComplex *str_v5_d)
{
  	

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double  nk[3];

  nk[0] = kx_d[i];
  nk[1] = ky_d[j];
  nk[2] = kz_d[k];

  str_v0_d[idx].x = -1.0*unewx_d[idx].y*nk[0];
  str_v1_d[idx].x = -1.0*unewy_d[idx].y*nk[1];
  str_v2_d[idx].x = -1.0*unewz_d[idx].y*nk[2];
  str_v3_d[idx].x = -1.0*(unewy_d[idx].y*nk[2] + unewz_d[idx].y*nk[1]);
  str_v4_d[idx].x = -1.0*(unewx_d[idx].y*nk[2] + unewz_d[idx].y*nk[0]);
  str_v5_d[idx].x = -1.0*(unewx_d[idx].y*nk[1] + unewy_d[idx].y*nk[0]);

  str_v0_d[idx].y =  unewx_d[idx].x*nk[0];
  str_v1_d[idx].y =  unewy_d[idx].x*nk[1];
  str_v2_d[idx].y =  unewz_d[idx].x*nk[2];
  str_v3_d[idx].y =  unewy_d[idx].x*nk[2] + unewz_d[idx].x*nk[1];
  str_v4_d[idx].y =  unewx_d[idx].x*nk[2] + unewz_d[idx].x*nk[0];
  str_v5_d[idx].y =  unewx_d[idx].x*nk[1] + unewy_d[idx].x*nk[0];
   
}

__global__ void Compute_eigsts0(double *Chom11_d, double *Chom12_d,
                                double *Chet11_d, double *Chet12_d, 
                                cuDoubleComplex *dfdphi_d, double *eigsts, 
                                double *epszero_d, int *ny_d, int *nz_d)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double hphi, e_temp, eig[3], Ct11, Ct12;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  eig[0] = (*epszero_d)*hphi;
  eig[1] = (*epszero_d)*hphi;
  eig[2] = (*epszero_d)*hphi;


  Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
  Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

  eigsts[idx] = Ct11*eig[0] + 
                Ct12*eig[1] +
                Ct12*eig[2];     
  __syncthreads();     
}

__global__ void Compute_eigsts1(double *Chom11_d, double *Chom12_d,
                                double *Chet11_d, double *Chet12_d, 
                                cuDoubleComplex *dfdphi_d, double *eigsts, 
                                double *epszero_d, int *ny_d, int *nz_d)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double hphi, e_temp, eig[3], Ct11, Ct12;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  eig[0] = (*epszero_d)*hphi;
  eig[1] = (*epszero_d)*hphi;
  eig[2] = (*epszero_d)*hphi;


  Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
  Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

  eigsts[idx] = Ct12*eig[0] + 
                Ct11*eig[1] +
                Ct12*eig[2];     
  __syncthreads();     
}
__global__ void Compute_eigsts2(double *Chom11_d, double *Chom12_d,
                                double *Chet11_d, double *Chet12_d, 
                                cuDoubleComplex *dfdphi_d, double *eigsts, 
                                double *epszero_d, int *ny_d, int *nz_d)
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double hphi, e_temp, eig[3], Ct11, Ct12;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  eig[0] = (*epszero_d)*hphi;
  eig[1] = (*epszero_d)*hphi;
  eig[2] = (*epszero_d)*hphi;


  Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
  Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

  eigsts[idx] = Ct12*eig[0] + 
                Ct12*eig[1] +
                Ct11*eig[2];     
  __syncthreads();     
}

__global__ void Compute_persts0(double *Chom11_d, double *Chom12_d, 
                     double *Chet11_d, double *Chet12_d, cuDoubleComplex *dfdphi_d, 
                     cuDoubleComplex *str_v0_d, cuDoubleComplex *str_v1_d, 
                     cuDoubleComplex *str_v2_d, double *persts, int *ny_d, int *nz_d)
{
   	 
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double e_temp, hphi; 
  double str_v[3], Ct11, Ct12;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  str_v[0] = str_v0_d[idx].x;
  str_v[1] = str_v1_d[idx].x;
  str_v[2] = str_v2_d[idx].x;

  Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
  Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

  persts[idx] = Ct11*str_v[0] + 
                Ct12*str_v[1] +
                Ct12*str_v[2];

  __syncthreads();     
}
__global__ void Compute_persts1(double *Chom11_d, double *Chom12_d, 
                     double *Chet11_d, double *Chet12_d, cuDoubleComplex *dfdphi_d, 
                     cuDoubleComplex *str_v0_d, cuDoubleComplex *str_v1_d, 
                     cuDoubleComplex *str_v2_d, double *persts, int *ny_d, int *nz_d)
{
   	 
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double e_temp, hphi; 
  double str_v[3], Ct11, Ct12;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  str_v[0] = str_v0_d[idx].x;
  str_v[1] = str_v1_d[idx].x;
  str_v[2] = str_v2_d[idx].x;

  Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
  Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

  persts[idx] = Ct12*str_v[0] + 
                Ct11*str_v[1] +
                Ct12*str_v[2];     
  __syncthreads();     
}
__global__ void Compute_persts2(double *Chom11_d, double *Chom12_d, 
                     double *Chet11_d, double *Chet12_d, cuDoubleComplex *dfdphi_d, 
                     cuDoubleComplex *str_v0_d, cuDoubleComplex *str_v1_d, 
                     cuDoubleComplex *str_v2_d, double *persts, int *ny_d, int *nz_d)
{
   	 
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double e_temp, hphi; 
  double str_v[3], Ct11, Ct12;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  str_v[0] = str_v0_d[idx].x;
  str_v[1] = str_v1_d[idx].x;
  str_v[2] = str_v2_d[idx].x;

  Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
  Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

  persts[idx] = Ct12*str_v[0] + 
                Ct12*str_v[1] +
                Ct11*str_v[2];     
  __syncthreads();     
}
__global__ void Compute_persts3(double *Chom44_d, double *Chet44_d, 
                     cuDoubleComplex *dfdphi_d, cuDoubleComplex *str_v3_d, 
                     double *persts, int *ny_d, int *nz_d)
{
   	 
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double e_temp, hphi; 
  double str_v, Ct44;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  str_v = str_v3_d[idx].x;

  Ct44 =  (*Chom44_d) + (*Chet44_d*(2.0*hphi - 1.0));

  persts[idx] = Ct44*str_v;

  __syncthreads(); 
}
__global__ void Compute_persts4(double *Chom44_d, double *Chet44_d, 
                     cuDoubleComplex *dfdphi_d, cuDoubleComplex *str_v4_d, 
                     double *persts, int *ny_d, int *nz_d)
{
   	 
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double e_temp, hphi; 
  double str_v, Ct44;

  e_temp = dfdphi_d[idx].x;
  str_v  = str_v4_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  Ct44 =  (*Chom44_d) + (*Chet44_d*(2.0*hphi - 1.0));

  persts[idx] = Ct44*str_v ; 
  __syncthreads(); 
}
__global__ void Compute_persts5(double *Chom44_d, double *Chet44_d, 
                     cuDoubleComplex *dfdphi_d, cuDoubleComplex *str_v5_d, 
                     double *persts, int *ny_d, int *nz_d)
{
   	 
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double e_temp, hphi; 
  double str_v, Ct44;

  e_temp = dfdphi_d[idx].x;

  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  str_v = str_v5_d[idx].x;

  Ct44 =  (*Chom44_d) + (*Chet44_d*(2.0*hphi - 1.0));

  persts[idx] = Ct44*str_v ; 
  __syncthreads(); 
}

__global__ void Compute_homstr(double *hom_strain_v, double *S11_d, 
                               double *S12_d, double *S44_d, 
                               double *sigappl_v_d, double *avgeigsts0,
                               double *avgeigsts1, double *avgeigsts2,
                               double *avgpersts0, double *avgpersts1,
                               double *avgpersts2, double *avgpersts3,
                               double *avgpersts4, double *avgpersts5)
{
   hom_strain_v[0] = (*S11_d)*(sigappl_v_d[0] + *avgeigsts0 - *avgpersts0) + 
                     (*S12_d)*(sigappl_v_d[1] + *avgeigsts1 - *avgpersts1) +
                     (*S12_d)*(sigappl_v_d[2] + *avgeigsts2 - *avgpersts2);

   hom_strain_v[1] = (*S12_d)*(sigappl_v_d[0] + *avgeigsts0 - *avgpersts0) + 
                     (*S11_d)*(sigappl_v_d[1] + *avgeigsts1 - *avgpersts1) +
                     (*S12_d)*(sigappl_v_d[2] + *avgeigsts2 - *avgpersts2);

   hom_strain_v[2] = (*S12_d)*(sigappl_v_d[0] + *avgeigsts0 - *avgpersts0) + 
                     (*S12_d)*(sigappl_v_d[1] + *avgeigsts1 - *avgpersts1) +
                     (*S11_d)*(sigappl_v_d[2] + *avgeigsts2 - *avgpersts2);

   hom_strain_v[3] = (*S44_d)*(sigappl_v_d[3] - *avgpersts3); 

   hom_strain_v[4] = (*S44_d)*(sigappl_v_d[4] - *avgpersts4); 

   hom_strain_v[5] = (*S44_d)*(sigappl_v_d[5] - *avgpersts5); 
}

__global__ void Compute_ts(double *Chom11_d, double *Chom12_d, 
                           double *Chom44_d, double *Chet11_d, 
                           double *Chet12_d, double *Chet44_d,
                           cuDoubleComplex *dfdphi_d, 
                           cuDoubleComplex *ts0_d, cuDoubleComplex *ts1_d,
                           cuDoubleComplex *ts2_d, cuDoubleComplex *ts3_d,
                           cuDoubleComplex *ts4_d, cuDoubleComplex *ts5_d,
                       cuDoubleComplex *str_v0_d, cuDoubleComplex *str_v1_d,
                       cuDoubleComplex *str_v2_d, cuDoubleComplex *str_v3_d,
                       cuDoubleComplex *str_v4_d, cuDoubleComplex *str_v5_d,
                       double *hom_strain_v ,double *epszero_d, int *ny_d, int *nz_d)
{
   	
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));
  double e_temp, Ct11, Ct12, Ct44, Cin11, Cin12, Cin44, hphi;
  double temp_v[6], eig[3];

  e_temp = dfdphi_d[idx].x; 
  hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);

  Ct11 = (*Chom11_d) + (*Chet11_d)*(2.0*hphi - 1.0);  
  Ct12 = (*Chom12_d) + (*Chet12_d)*(2.0*hphi - 1.0);  
  Ct44 = (*Chom44_d) + (*Chet44_d)*(2.0*hphi - 1.0);  

  Cin11 = Ct11 - (*Chom11_d); 
  Cin12 = Ct12 - (*Chom12_d); 
  Cin44 = Ct44 - (*Chom44_d); 

  eig[0] = (*epszero_d)*hphi;
  eig[1] = (*epszero_d)*hphi;
  eig[2] = (*epszero_d)*hphi;

  temp_v[0] = str_v0_d[idx].x;
  temp_v[1] = str_v1_d[idx].x;
  temp_v[2] = str_v2_d[idx].x;
  temp_v[3] = str_v3_d[idx].x;
  temp_v[4] = str_v4_d[idx].x;
  temp_v[5] = str_v5_d[idx].x;

  ts0_d[idx].x = Ct11*(eig[0]-(hom_strain_v[0]))-
                 Cin11*temp_v[0]+
                 Ct12*(eig[1]-(hom_strain_v[1]))-
                 Cin12*temp_v[1]+
                 Ct12*(eig[2]-(hom_strain_v[2]))-
                 Cin12*temp_v[2];

  ts1_d[idx].x = Ct12*(eig[0]-(hom_strain_v[0]))-
                 Cin12*temp_v[0]+
                 Ct11*(eig[1]-(hom_strain_v[1]))-
                 Cin11*temp_v[1]+
                 Ct12*(eig[2]-(hom_strain_v[2]))-
                 Cin12*temp_v[2];

  ts2_d[idx].x = Ct12*(eig[0]-(hom_strain_v[0]))-
                 Cin12*temp_v[0]+
                 Ct12*(eig[1]-(hom_strain_v[1]))-
                 Cin12*temp_v[1]+
                 Ct11*(eig[2]-(hom_strain_v[2]))-
                 Cin11*temp_v[2];

  ts3_d[idx].x = Ct44*(-1.0*(hom_strain_v[3]))-
                 Cin44*temp_v[3];

  ts4_d[idx].x = Ct44*(-1.0*(hom_strain_v[4]))-
                 Cin44*temp_v[4];

  ts5_d[idx].x = Ct44*(-1.0*(hom_strain_v[5]))-
                 Cin44*temp_v[5];
            
  ts0_d[idx].y = 0.0;
  ts1_d[idx].y = 0.0;
  ts2_d[idx].y = 0.0;
  ts3_d[idx].y = 0.0;
  ts4_d[idx].y = 0.0;
  ts5_d[idx].y = 0.0;
}

__global__ void Update_disp(int *ny_d, int *nz_d, 
                            double *kx_d, double *ky_d, double *kz_d, 
                            double *omega_v0, double *omega_v1,
                            double *omega_v2, double *omega_v3, 
                            double *omega_v4, double *omega_v5, 
                            cuDoubleComplex *ts0_d, cuDoubleComplex *ts1_d,
                            cuDoubleComplex *ts2_d, cuDoubleComplex *ts3_d, 
                            cuDoubleComplex *ts4_d, cuDoubleComplex *ts5_d,
                            cuDoubleComplex *unewx_d, 
                            cuDoubleComplex *unewy_d, 
                            cuDoubleComplex *unewz_d)
{
   	
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  double          nk[3], omega[6];
  cuDoubleComplex stmp_v[6], fk10, fk20, fk30;

  nk[0] = kx_d[i];
  nk[1] = ky_d[j];
  nk[2] = kz_d[k];

  omega[0] = omega_v0[idx];
  omega[1] = omega_v1[idx];
  omega[2] = omega_v2[idx];
  omega[3] = omega_v3[idx];
  omega[4] = omega_v4[idx];
  omega[5] = omega_v5[idx];

  stmp_v[0].x = ts0_d[idx].x;    
  stmp_v[1].x = ts1_d[idx].x;    
  stmp_v[2].x = ts2_d[idx].x;    
  stmp_v[3].x = ts3_d[idx].x;    
  stmp_v[4].x = ts4_d[idx].x;    
  stmp_v[5].x = ts5_d[idx].x;

  stmp_v[0].y = ts0_d[idx].y;    
  stmp_v[1].y = ts1_d[idx].y;    
  stmp_v[2].y = ts2_d[idx].y;    
  stmp_v[3].y = ts3_d[idx].y;    
  stmp_v[4].y = ts4_d[idx].y;    
  stmp_v[5].y = ts5_d[idx].y;


  fk10.x = stmp_v[0].x * nk[0] + stmp_v[5].x * nk[1] + 
           stmp_v[4].x * nk[2];
  fk20.x = stmp_v[5].x * nk[0] + stmp_v[1].x * nk[1] + 
           stmp_v[3].x * nk[2];
  fk30.x = stmp_v[4].x * nk[0] + stmp_v[3].x * nk[1] + 
           stmp_v[2].x * nk[2];    
           
  fk10.y = stmp_v[0].y * nk[0] + stmp_v[5].y * nk[1] + 
           stmp_v[4].y * nk[2];
  fk20.y = stmp_v[5].y * nk[0] + stmp_v[1].y * nk[1] + 
           stmp_v[3].y * nk[2];
  fk30.y = stmp_v[4].y * nk[0] + stmp_v[3].y * nk[1] + 
           stmp_v[2].y * nk[2];

  unewx_d[idx].x = (omega[0] * fk10.y + 
                    omega[5] * fk20.y +
                    omega[4] * fk30.y);

  unewy_d[idx].x = (omega[5] * fk10.y + 
                    omega[1] * fk20.y +
                    omega[3] * fk30.y);
 
  unewz_d[idx].x = (omega[4] * fk10.y + 
                    omega[3] * fk20.y +
                    omega[2] * fk30.y);

  unewx_d[idx].y = -1.0*(omega[0] * fk10.x + 
                         omega[5] * fk20.x +
                         omega[4] * fk30.x);

  unewy_d[idx].y = -1.0*(omega[5] * fk10.x + 
                         omega[1] * fk20.x +
                         omega[3] * fk30.x);

  unewz_d[idx].y = -1.0*(omega[4] * fk10.x + 
                         omega[3] * fk20.x +
                         omega[2] * fk30.x);
}

__global__ void Compute_sq_diff_disp(cuDoubleComplex *unewx_d, 
                                  cuDoubleComplex *unewy_d, 
                                  cuDoubleComplex *unewz_d, 
                                  cuDoubleComplex *ux_d, 
                                  cuDoubleComplex *uy_d,
                                  cuDoubleComplex *uz_d,
                                  double *sq_diff_disp,
                                  int *ny_d, int *nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));

  sq_diff_disp[idx] = pow((unewx_d[idx].x - ux_d[idx].x),2.0) +
                      pow((unewy_d[idx].x - uy_d[idx].x),2.0) +
                      pow((unewz_d[idx].x - uz_d[idx].x),2.0);

} 

__global__ void Copy_new_sol(cuDoubleComplex *unewx_d,cuDoubleComplex *unewy_d, 
                             cuDoubleComplex *unewz_d,cuDoubleComplex *ux_d, 
                             cuDoubleComplex *uy_d, cuDoubleComplex *uz_d,
                             int *ny_d, int *nz_d)
{

  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k = threadIdx.z + blockDim.z*blockIdx.z;

  int idx = k + (*nz_d)*(j + i*(*ny_d));
   
  ux_d[idx].x = unewx_d[idx].x;
  uy_d[idx].x = unewy_d[idx].x;
  uz_d[idx].x = unewz_d[idx].x;

  ux_d[idx].y = unewx_d[idx].y;
  uy_d[idx].y = unewy_d[idx].y;
  uz_d[idx].y = unewz_d[idx].y;
}

__global__ void Compute_dfeldphi(cuDoubleComplex *str_v0_d, 
                         cuDoubleComplex *str_v1_d, cuDoubleComplex *str_v2_d,
                         cuDoubleComplex *str_v3_d, cuDoubleComplex *str_v4_d,
                         cuDoubleComplex *str_v5_d, cuDoubleComplex *dfdphi_d,
                         cuDoubleComplex *dfeldphi_d, double *Chom11_d,
                         double *Chom12_d, double *Chom44_d, double *Chet11_d,
                         double *Chet12_d, double *Chet44_d,  
                         double *hom_strain_v, double *epszero_d,
                         int *ny_d, int *nz_d)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (*nz_d)*(j + i*(*ny_d));

   double hphi, hphi_p, e_temp, str_v[6], estr[3], hstr[6];
   double Ct11, Ct12;
 
   e_temp = dfdphi_d[idx].x;
   hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp + 10.0);
   hphi_p = (30.0*e_temp*e_temp*(1.0-e_temp)*(1.0-e_temp));           
   
   Ct11 =  (*Chom11_d) + (*Chet11_d*(2.0*hphi - 1.0));
   Ct12 =  (*Chom12_d) + (*Chet12_d*(2.0*hphi - 1.0));

   str_v[0] = str_v0_d[idx].x;
   str_v[1] = str_v1_d[idx].x;
   str_v[2] = str_v2_d[idx].x;
   str_v[3] = str_v3_d[idx].x;
   str_v[4] = str_v4_d[idx].x;
   str_v[5] = str_v5_d[idx].x;

   estr[0] = (*epszero_d)*hphi;
   estr[1] = (*epszero_d)*hphi;
   estr[2] = (*epszero_d)*hphi;
 
   hstr[0] = hom_strain_v[0];
   hstr[1] = hom_strain_v[1];
   hstr[2] = hom_strain_v[2];
   hstr[3] = hom_strain_v[3];
   hstr[4] = hom_strain_v[4];
   hstr[5] = hom_strain_v[5];

   dfeldphi_d[idx].x = 0.5*
               ((*Chet11_d)*2.0*hphi_p*
                   (hstr[0]+str_v[0]-estr[0])
                  *(hstr[0]+str_v[0]-estr[0])
               +(*Chet11_d)*2.0*hphi_p*
                   (hstr[1]+str_v[1]-estr[1])
                  *(hstr[1]+str_v[1]-estr[1])
               +(*Chet11_d)*2.0*hphi_p*
                   (hstr[2]+str_v[2]-estr[2])
                  *(hstr[2]+str_v[2]-estr[2])
               +2.0*(*Chet12_d)*2.0*hphi_p*
                   (hstr[0]+str_v[0]-estr[0])
                  *(hstr[1]+str_v[1]-estr[1])
               +2.0*(*Chet12_d)*2.0*hphi_p*
                   (hstr[0]+str_v[0]-estr[0])
                  *(hstr[2]+str_v[2]-estr[2])
               +2.0*(*Chet12_d)*2.0*hphi_p*
                   (hstr[1]+str_v[1]-estr[1])
                  *(hstr[2]+str_v[2]-estr[2])
               +(*Chet44_d)*2.0*hphi_p*
                   (hstr[3]+str_v[3])
                  *(hstr[3]+str_v[3])
               +(*Chet44_d)*2.0*hphi_p*
                   (hstr[4]+str_v[4])
                  *(hstr[4]+str_v[4])
               +(*Chet44_d)*2.0*hphi_p*
                   (hstr[5]+str_v[5])
                  *(hstr[5]+str_v[5])) -
               (
                Ct11*
                (hstr[0]+str_v[0]-estr[0])*
                (*epszero_d)*hphi_p +
                Ct11*
                (hstr[1]+str_v[1]-estr[1])*
                (*epszero_d)*hphi_p +
                Ct11*
                (hstr[2]+str_v[2]-estr[2])*
                (*epszero_d)*hphi_p +
                Ct12*
                (hstr[1]+str_v[1]-estr[1])*
                (*epszero_d)*hphi_p +
                Ct12*
                (hstr[1]+str_v[1]-estr[1])*
                (*epszero_d)*hphi_p +
                Ct12*
                (hstr[0]+str_v[0]-estr[0])*
                (*epszero_d)*hphi_p +
                Ct12*
                (hstr[0]+str_v[0]-estr[0])*
                (*epszero_d)*hphi_p +
                Ct12*
                (hstr[2]+str_v[2]-estr[2])*
                (*epszero_d)*hphi_p + 
                Ct12*
                (hstr[2]+str_v[2]-estr[2])*
                (*epszero_d)*hphi_p
               );

     dfeldphi_d[idx].y = 0.0; 
} 
__global__ void Average(double *x, double *sizescale)
{
   *x = *x*(*sizescale);
}
__global__ void FindTotalstr(cuDoubleComplex *str_v0_d, double *eigstr0,
                             double *hom_strain_v, double *totalstr_d,
                             int *ny_d, int *nz_d)
{
   
   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + (*nz_d)*(j + i*(*ny_d));

   totalstr_d[idx] = hom_strain_v[2] + str_v0_d[idx].x - eigstr0[idx];
}

void InhomElast (void){

  int              converge=1,iter=1, FALSE = 0;
  int              complex_size,double_size;
  double           *Cavg11,*Cavg12,*Cavg44;
  double           *avgeigsts0, *avgeigsts1, *avgeigsts2;
  double           *avgpersts0, *avgpersts1, *avgpersts2;
  double           *avgpersts3, *avgpersts4, *avgpersts5;
  double           *hom_strain_v;
  double           *disperr_d, *sq_diff_disp;
  //double           *totalstr_d, *totalstr;
  void             *t_storage = NULL;
  size_t           t_storage_bytes = 0;

  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Ctotal,
                         Cavg11, nx*ny*nz);

  complex_size = nx*ny*nz*sizeof(cuDoubleComplex);
  double_size  = nx*ny*nz*sizeof(double);

  //totalstr = (double*)malloc(double_size);

  checkCudaErrors(cudaMalloc((void**)&str_v0_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&str_v1_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&str_v2_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&str_v3_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&str_v4_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&str_v5_d, complex_size));
  //checkCudaErrors(cudaMalloc((void**)&totalstr_d, complex_size));

  checkCudaErrors(cudaMalloc((void**)&unewx_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&unewy_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&unewz_d, complex_size));

  checkCudaErrors(cudaMalloc((void**)&ts0_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&ts1_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&ts2_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&ts3_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&ts4_d, complex_size));
  checkCudaErrors(cudaMalloc((void**)&ts5_d, complex_size));

  //checkCudaErrors(cudaMalloc((void**)&Cinhom11, double_size)); 
  //checkCudaErrors(cudaMalloc((void**)&Cinhom12, double_size)); 
  //checkCudaErrors(cudaMalloc((void**)&Cinhom44, double_size)); 

  checkCudaErrors(cudaMalloc((void**)&Ctotal, double_size)); 
//  checkCudaErrors(cudaMalloc((void**)&Ctotal12, double_size)); 
//  checkCudaErrors(cudaMalloc((void**)&Ctotal44, double_size)); 

  checkCudaErrors(cudaMalloc((void**)&Cavg11, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&Cavg12, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&Cavg44, sizeof(double))); 

  checkCudaErrors(cudaMalloc((void**)&avgeigsts0, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgeigsts1, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgeigsts2, sizeof(double))); 

  checkCudaErrors(cudaMalloc((void**)&avgpersts0, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgpersts1, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgpersts2, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgpersts3, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgpersts4, sizeof(double))); 
  checkCudaErrors(cudaMalloc((void**)&avgpersts5, sizeof(double))); 

  //checkCudaErrors(cudaMalloc((void**)&eigstr0, double_size));
  //checkCudaErrors(cudaMalloc((void**)&eigstr1, double_size));
  //checkCudaErrors(cudaMalloc((void**)&eigstr2, double_size));

  checkCudaErrors(cudaMalloc((void**)&eigsts, double_size));
  //checkCudaErrors(cudaMalloc((void**)&eigsts1, double_size));
  //checkCudaErrors(cudaMalloc((void**)&eigsts2, double_size));

  checkCudaErrors(cudaMalloc((void**)&persts, double_size));
  //checkCudaErrors(cudaMalloc((void**)&persts1, double_size));
  //checkCudaErrors(cudaMalloc((void**)&persts2, double_size));
  //checkCudaErrors(cudaMalloc((void**)&persts3, double_size));
  //checkCudaErrors(cudaMalloc((void**)&persts4, double_size));
  //checkCudaErrors(cudaMalloc((void**)&persts5, double_size));
  checkCudaErrors(cudaMalloc((void**)&sq_diff_disp, double_size));
  checkCudaErrors(cudaMalloc((void**)&disperr_d, sizeof(double)));

  checkCudaErrors(cudaMalloc((void**)&hom_strain_v, 6*sizeof(double)));
  checkCudaErrors(cudaMalloc(&t_storage, t_storage_bytes));

 /*----------------------------------------------------------------------
  *     Defining total elastic constants and average elastic constants 
  *     in Voight's form
  *----------------------------------------------------------------------*/

  Compute_Ctotal<<< Gridsize, Blocksize>>>(dfdphi_d, ny_d, nz_d, Ctotal, Chom11_d,
                                           Chet11_d); 
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Ctotal,
                         Cavg11, nx*ny*nz);
  Compute_Ctotal<<< Gridsize, Blocksize>>>(dfdphi_d, ny_d, nz_d, Ctotal, Chom12_d,
                                           Chet12_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Ctotal,
                         Cavg12, nx*ny*nz);
  Compute_Ctotal<<< Gridsize, Blocksize>>>(dfdphi_d, ny_d, nz_d, Ctotal, Chom44_d,
                                           Chet44_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Ctotal,
                         Cavg44, nx*ny*nz);
  //Compute_Ctotal<<< Gridsize, Blocksize >>>(dfdphi_d, ny_d, nz_d, 
  //                                          Ctotal, Ctotal12, Ctotal44, 
  //                                          Chom11_d, Chom12_d, Chom44_d, 
  //                                          Chet11_d, Chet12_d, Chet44_d);

 /*-----------------------------------------------------------------------
  *    Defining average elastic constants tensor in Voight's form
  *---------------------------------------------------------------------*/
  ///////////////////////////////////////////////////////////////////////
  /*cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Ctotal12, 
                 	 Cavg12, nx*ny*nz);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, Ctotal44, 
			 Cavg44, nx*ny*nz);
  */
  ///////////////////////////////////////////////////////////////////////
  Average<<<1,1>>>(Cavg11, sizescale_d); 
  Average<<<1,1>>>(Cavg12, sizescale_d); 
  Average<<<1,1>>>(Cavg44, sizescale_d); 

 /*----------------------------------------------------------------------
  *                        Compliance tensor calculations 
  *---------------------------------------------------------------------*/
  Compute_Sij<<<1,1>>>(Cavg11, Cavg12, Cavg44, S11_d, S12_d, S44_d);

 /*Inhomogenous part of total stiffness tensor in Voight's form*/ 

  /*Compute_Cinhom<<<Gridsize, Blocksize>>>(Cinhom11, Cinhom12, Cinhom44, 
                                          Ctotal11, Ctotal12, Ctotal44,
                                          Chom11_d, Chom12_d, Chom44_d, 
                                          ny_d, nz_d);
  //Finding eigen strain
  *Compute_eigstr<<<Gridsize, Blocksize>>>(dfdphi_d, eigstr0, eigstr1, 
                                          eigstr2, epszero_d, ny_d, nz_d);
  */
  //Finding eigen stress
  Compute_eigsts0<<<Gridsize, Blocksize>>>(Chom11_d, Chom12_d, Chet11_d, Chet12_d, 
                                           dfdphi_d, eigsts, epszero_d,ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, eigsts, 
			 avgeigsts0, nx*ny*nz);
  Compute_eigsts1<<<Gridsize, Blocksize>>>(Chom11_d, Chom12_d, Chet11_d, Chet12_d, 
                                           dfdphi_d, eigsts, epszero_d,ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, eigsts, 
			 avgeigsts1, nx*ny*nz);
  Compute_eigsts2<<<Gridsize, Blocksize>>>(Chom11_d, Chom12_d, Chet11_d, Chet12_d, 
                                           dfdphi_d, eigsts, epszero_d,ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, eigsts, 
			 avgeigsts2, nx*ny*nz);
  //Finding average eigen stress
  //cub::DeviceReduce::Sum(t_storage, t_storage_bytes, eigsts1, 
  //			 avgeigsts1, nx*ny*nz);
  //cub::DeviceReduce::Sum(t_storage, t_storage_bytes, eigsts2, 
  //			 avgeigsts2, nx*ny*nz);

  Average<<<1,1>>>(avgeigsts0, sizescale_d); 
  Average<<<1,1>>>(avgeigsts1, sizescale_d); 
  Average<<<1,1>>>(avgeigsts2, sizescale_d); 

  cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_FORWARD);
  cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_FORWARD);
  cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_FORWARD);

  Compute_perstr<<< Gridsize, Blocksize >>>(ny_d, nz_d, kx_d, ky_d, kz_d, 
                                            ux_d, uy_d, uz_d, 
                                            str_v0_d, str_v1_d, str_v2_d,
                                            str_v3_d, str_v4_d, str_v5_d);

  cufftExecZ2Z(plan, str_v0_d, str_v0_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v1_d, str_v1_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v2_d, str_v2_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v3_d, str_v3_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v4_d, str_v4_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v5_d, str_v5_d, CUFFT_INVERSE);

  cufftExecZ2Z(plan, ux_d, ux_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, uy_d, uy_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, uz_d, uz_d, CUFFT_INVERSE);

  Normalize<<< Gridsize, Blocksize >>>(str_v0_d, sizescale_d, ny_d, nz_d); 
  Normalize<<< Gridsize, Blocksize >>>(str_v1_d, sizescale_d, ny_d, nz_d);  
  Normalize<<< Gridsize, Blocksize >>>(str_v2_d, sizescale_d, ny_d, nz_d);  
  Normalize<<< Gridsize, Blocksize >>>(str_v3_d, sizescale_d, ny_d, nz_d);  
  Normalize<<< Gridsize, Blocksize >>>(str_v4_d, sizescale_d, ny_d, nz_d);  
  Normalize<<< Gridsize, Blocksize >>>(str_v5_d, sizescale_d, ny_d, nz_d);

  Normalize<<< Gridsize, Blocksize >>>(ux_d, sizescale_d, ny_d, nz_d);
  Normalize<<< Gridsize, Blocksize >>>(uy_d, sizescale_d, ny_d, nz_d); 
  Normalize<<< Gridsize, Blocksize >>>(uz_d, sizescale_d, ny_d, nz_d); 


 /*-----------------------------------------------------------------------
  *                        Refinement of displacement
  *---------------------------------------------------------------------*/
  while (converge != FALSE){

      //Finidng periodic stresses
      Compute_persts0<<< Gridsize, Blocksize >>>(Chom11_d, Chom12_d, 
                                               Chet11_d, Chet12_d, dfdphi_d, 
                                               str_v0_d, str_v1_d, str_v2_d, 
                                               persts, ny_d, nz_d);
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts0, nx*ny*nz);
      Compute_persts1<<< Gridsize, Blocksize >>>(Chom11_d, Chom12_d, 
                                               Chet11_d, Chet12_d, dfdphi_d, 
                                               str_v0_d, str_v1_d, str_v2_d, 
                                               persts, ny_d, nz_d);
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts1, nx*ny*nz);
      Compute_persts2<<< Gridsize, Blocksize >>>(Chom11_d, Chom12_d, 
                                               Chet11_d, Chet12_d, dfdphi_d, 
                                               str_v0_d, str_v1_d, str_v2_d, 
                                               persts, ny_d, nz_d);
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts2, nx*ny*nz);
      Compute_persts3<<< Gridsize, Blocksize >>>(Chom44_d, Chet44_d, dfdphi_d, 
                     str_v3_d, persts, ny_d, nz_d);
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts3, nx*ny*nz);
      Compute_persts4<<< Gridsize, Blocksize >>>(Chom44_d, Chet44_d, dfdphi_d, 
                     str_v4_d, persts, ny_d, nz_d);
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts4, nx*ny*nz);
      Compute_persts5<<< Gridsize, Blocksize >>>(Chom44_d, Chet44_d, dfdphi_d, 
                     str_v5_d, persts, ny_d, nz_d);
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts5, nx*ny*nz);

      Average<<<1,1>>>(avgpersts0, sizescale_d);
      Average<<<1,1>>>(avgpersts1, sizescale_d);
      Average<<<1,1>>>(avgpersts2, sizescale_d);
      Average<<<1,1>>>(avgpersts3, sizescale_d);
      Average<<<1,1>>>(avgpersts4, sizescale_d);
      Average<<<1,1>>>(avgpersts5, sizescale_d);

      //Finding homogeneous strain
      Compute_homstr<<<1,1>>>(hom_strain_v, S11_d, S12_d, S44_d, sigappl_v_d,
                              avgeigsts0, avgeigsts1, avgeigsts2, 
                              avgpersts0, avgpersts1, avgpersts2, 
                              avgpersts3, avgpersts4, avgpersts5);

      //Finding ts
      Compute_ts<<<Gridsize, Blocksize>>>(Chom11_d, Chom12_d, Chom44_d, 
                                          Chet11_d, Chet12_d, Chet44_d, dfdphi_d, 
                                          ts0_d, ts1_d, ts2_d, ts3_d,
                                          ts4_d, ts5_d, str_v0_d, str_v1_d,
                                          str_v2_d, str_v3_d, str_v4_d, str_v5_d,
                                          hom_strain_v , epszero_d, ny_d, nz_d);

      cufftExecZ2Z(plan, ts0_d, ts0_d, CUFFT_FORWARD);
      cufftExecZ2Z(plan, ts1_d, ts1_d, CUFFT_FORWARD);
      cufftExecZ2Z(plan, ts2_d, ts2_d, CUFFT_FORWARD);
      cufftExecZ2Z(plan, ts3_d, ts3_d, CUFFT_FORWARD);
      cufftExecZ2Z(plan, ts4_d, ts4_d, CUFFT_FORWARD);
      cufftExecZ2Z(plan, ts5_d, ts5_d, CUFFT_FORWARD);
   
      //Update displacements
      Update_disp<<< Gridsize,Blocksize >>>(ny_d, nz_d, kx_d, ky_d, kz_d, 
              omega_v0, omega_v1, omega_v2, omega_v3, omega_v4, omega_v5,
              ts0_d, ts1_d, ts2_d, ts3_d, ts4_d, ts5_d, 
              unewx_d, unewy_d, unewz_d);

      // Finding periodic strains
      Compute_perstr<<< Gridsize, Blocksize >>>(ny_d, nz_d, kx_d, ky_d, kz_d, 
              unewx_d, unewy_d, unewz_d, str_v0_d, str_v1_d, str_v2_d,
              str_v3_d, str_v4_d, str_v5_d);

      cufftExecZ2Z(plan, unewx_d, unewx_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, unewy_d, unewy_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, unewz_d, unewz_d, CUFFT_INVERSE);

      cufftExecZ2Z(plan, str_v0_d, str_v0_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, str_v1_d, str_v1_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, str_v2_d, str_v2_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, str_v3_d, str_v3_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, str_v4_d, str_v4_d, CUFFT_INVERSE);
      cufftExecZ2Z(plan, str_v5_d, str_v5_d, CUFFT_INVERSE);
      
      Normalize<<< Gridsize, Blocksize >>>(unewx_d, sizescale_d, ny_d, nz_d);
      Normalize<<< Gridsize, Blocksize >>>(unewy_d, sizescale_d, ny_d, nz_d);
      Normalize<<< Gridsize, Blocksize >>>(unewz_d, sizescale_d, ny_d, nz_d);

      Normalize<<< Gridsize, Blocksize >>>(str_v0_d, sizescale_d, ny_d, nz_d); 
      Normalize<<< Gridsize, Blocksize >>>(str_v1_d, sizescale_d, ny_d, nz_d);  
      Normalize<<< Gridsize, Blocksize >>>(str_v2_d, sizescale_d, ny_d, nz_d);  
      Normalize<<< Gridsize, Blocksize >>>(str_v3_d, sizescale_d, ny_d, nz_d);  
      Normalize<<< Gridsize, Blocksize >>>(str_v4_d, sizescale_d, ny_d, nz_d);  
      Normalize<<< Gridsize, Blocksize >>>(str_v5_d, sizescale_d, ny_d, nz_d);

      //Find change in new and previous solution and save it to
      //unewx_d, unewy_d and unewz_d
      Compute_sq_diff_disp<<<Gridsize,Blocksize>>>(unewx_d, unewy_d, unewz_d, 
                                                   ux_d, uy_d, uz_d, 
                                                   sq_diff_disp, ny_d, 
                                                   nz_d);
  
      cub::DeviceReduce::Sum(t_storage, t_storage_bytes, sq_diff_disp, 
			 disperr_d, nx*ny*nz);
/*      double err_x, err_y, err_z;

      stat = cublasDznrm2(blas_handle, nx*ny*nz, unewx_d, 1, &err_x); 
      stat = cublasDznrm2(blas_handle, nx*ny*nz, unewy_d, 1, &err_y); 
      stat = cublasDznrm2(blas_handle, nx*ny*nz, unewz_d, 1, &err_z);

      if (stat != CUBLAS_STATUS_SUCCESS){
         
         printf("Error in finding norm");
 
      }

      disperror = sqrt(err_x*err_x + err_y*err_y + err_z*err_z); 
*/
      cudaMemcpy(&disperror, disperr_d, sizeof(double), cudaMemcpyDeviceToHost);
      disperror = sqrt(disperror); 
      //printf ("\niter=%d error = %le", iter, disperror);

      if (disperror < TOLERENCE){
        //printf("\nConvergence achieved at %d\n", iter);
        converge = 0;
      }
      
      iter = iter + 1;

      Copy_new_sol<<< Gridsize, Blocksize>>> (unewx_d, unewy_d, unewz_d,
                                              ux_d, uy_d, uz_d,
                                              ny_d, nz_d);
  }

/*
  cufftExecZ2Z(plan, str_v0_d, str_v0_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v1_d, str_v1_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v2_d, str_v2_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v3_d, str_v3_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v4_d, str_v4_d, CUFFT_INVERSE);
  cufftExecZ2Z(plan, str_v5_d, str_v5_d, CUFFT_INVERSE);
   
  
  Normalize<<< Gridsize, Blocksize >>>(str_v0_d, sizescale_d, ny_d, nz_d);      
  Normalize<<< Gridsize, Blocksize >>>(str_v1_d, sizescale_d, ny_d, nz_d);      
  Normalize<<< Gridsize, Blocksize >>>(str_v2_d, sizescale_d, ny_d, nz_d);      
  Normalize<<< Gridsize, Blocksize >>>(str_v3_d, sizescale_d, ny_d, nz_d);      
  Normalize<<< Gridsize, Blocksize >>>(str_v4_d, sizescale_d, ny_d, nz_d);      
  Normalize<<< Gridsize, Blocksize >>>(str_v5_d, sizescale_d, ny_d, nz_d);      
*/
  Compute_persts0<<< Gridsize, Blocksize >>>(Chom11_d, Chom12_d, 
                                           Chet11_d, Chet12_d, dfdphi_d, 
                                           str_v0_d, str_v1_d, str_v2_d, 
                                           persts, ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts0, nx*ny*nz);
  Compute_persts1<<< Gridsize, Blocksize >>>(Chom11_d, Chom12_d, 
                                           Chet11_d, Chet12_d, dfdphi_d, 
                                           str_v0_d, str_v1_d, str_v2_d, 
                                           persts, ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts1, nx*ny*nz);
  Compute_persts2<<< Gridsize, Blocksize >>>(Chom11_d, Chom12_d, 
                                           Chet11_d, Chet12_d, dfdphi_d, 
                                           str_v0_d, str_v1_d, str_v2_d, 
                                           persts, ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts2, nx*ny*nz);
  Compute_persts3<<< Gridsize, Blocksize>>>(Chom44_d, Chet44_d, dfdphi_d, 
                   str_v3_d, persts, ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts3, nx*ny*nz);
  Compute_persts4<<< Gridsize, Blocksize >>>(Chom44_d, Chet44_d, dfdphi_d, 
                   str_v4_d, persts, ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts4, nx*ny*nz);
  Compute_persts5<<< Gridsize, Blocksize >>>(Chom44_d, Chet44_d, dfdphi_d, 
                   str_v5_d, persts, ny_d, nz_d);
  cub::DeviceReduce::Sum(t_storage, t_storage_bytes, persts, 
			 avgpersts5, nx*ny*nz);

  Compute_homstr<<<1,1>>>(hom_strain_v, S11_d, S12_d, S44_d, sigappl_v_d,
                          avgeigsts0, avgeigsts1, avgeigsts2, 
                          avgpersts0, avgpersts1, avgpersts2, 
                          avgpersts3, avgpersts4, avgpersts5);

  Compute_dfeldphi<<<Gridsize, Blocksize>>>(str_v0_d, str_v1_d, str_v2_d,
                                            str_v3_d, str_v4_d, str_v5_d, 
                                            dfdphi_d, dfeldphi_d, Chom11_d,
                                            Chom12_d, Chom44_d, Chet11_d,
                                            Chet12_d, Chet44_d, hom_strain_v, 
                                            epszero_d, ny_d, nz_d);

  Compute_dfeldphi<<< Gridsize, Blocksize >>>(str_v0_d, str_v1_d, str_v2_d,
                         str_v3_d, str_v4_d, str_v5_d, dfdphi_d,dfeldphi_d, 
                         Chom11_d, Chom12_d, Chom44_d, Chet11_d, Chet12_d, 
                         Chet44_d, hom_strain_v, epszero_d, ny_d, nz_d);

//  FindTotalstr<<< Gridsize, Blocksize>>> (str_v2_d, eigstr2, hom_strain_v, totalstr_d, ny_d, nz_d);

//  cudaMemcpy(totalstr, totalstr_d, double_size, cudaMemcpyDeviceToHost);

  /*fp = fopen("strfield.txt","w");
  for (int k =0; k<nz; k++){
	
    fprintf(fp,"%d\t%le\n",k,totalstr[k+nz*(ny/2+ny*nx/2)]);
    
  }
     
  fclose(fp);*/

  cudaFree(str_v0_d);
  cudaFree(str_v1_d);
  cudaFree(str_v2_d);
  cudaFree(str_v3_d);
  cudaFree(str_v4_d);
  cudaFree(str_v5_d);
  cudaFree(unewx_d);
  cudaFree(unewy_d);
  cudaFree(unewz_d);
  cudaFree(ts0_d);
  cudaFree(ts1_d);
  cudaFree(ts2_d);
  cudaFree(ts3_d);
  cudaFree(ts4_d);
  cudaFree(ts5_d);
  cudaFree(eigsts);
  //cudaFree(eigsts1);
  //cudaFree(eigsts2);
  //cudaFree(eigstr0);
  //cudaFree(eigstr1);
  //cudaFree(eigstr2);
  cudaFree(avgeigsts0);
  cudaFree(avgeigsts1);
  cudaFree(avgeigsts2);
  cudaFree(persts);
  //cudaFree(persts1);
  //cudaFree(persts2);
  //cudaFree(persts3);
  //cudaFree(persts4);
  //cudaFree(persts5);
  cudaFree(avgpersts0);
  cudaFree(avgpersts1);
  cudaFree(avgpersts2);
  cudaFree(avgpersts3);
  cudaFree(avgpersts4);
  cudaFree(avgpersts5);
  //cudaFree(Cinhom11);
  //cudaFree(Cinhom12);
  //cudaFree(Cinhom44);
  cudaFree(Ctotal);
  //cudaFree(Ctotal12);
  //cudaFree(Ctotal44);
  cudaFree(hom_strain_v);
  cudaFree(Cavg11);
  cudaFree(Cavg12);
  cudaFree(Cavg44);
  cudaFree(sq_diff_disp);
  cudaFree(disperr_d);
  cudaFree(t_storage);
  //cudaFree(totalstr_d);
  //free(totalstr);
}
