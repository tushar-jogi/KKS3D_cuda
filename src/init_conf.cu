__global__ void CreateNuclei(int nx, int ny, int nz, double centx, double centy,
                             double centz, double ppt_size_d, 
                             cuDoubleComplex *comp_d, cuDoubleComplex *phi_d, 
                             cuDoubleComplex *dfdphi_d, double c_beta_eq_d, 
                             double c0_d)
{

   int i = threadIdx.x + blockIdx.x*blockDim.x;
   int j = threadIdx.y + blockIdx.y*blockDim.y;
   int k = threadIdx.z + blockIdx.z*blockDim.z;

   int idx = k+nz*(j + i*(ny));

   if ((((double)i - centx)*((double)i - centx) + 
        ((double)j - centy)*((double)j - centy) + 
        ((double)k - centz)*((double)k - centz))  
        <= ((ppt_size_d)*(ppt_size_d)))
   {  
        comp_d[idx].x   = (c_beta_eq_d);
        comp_d[idx].y   = 0.0;
        phi_d[idx].x    = 1.0;
        phi_d[idx].y    = 0.0;
        dfdphi_d[idx].x = phi_d[idx].x;
        dfdphi_d[idx].y = phi_d[idx].y;
   }
   else
   {
        comp_d[idx].x   = (c0_d);
        comp_d[idx].y   = 0.0;
        phi_d[idx].x    = 0.0;
        phi_d[idx].y    = 0.0;
        dfdphi_d[idx].x = phi_d[idx].x;
        dfdphi_d[idx].y = phi_d[idx].y;
   }

   __syncthreads();
}

__global__ void CreateRandomNuclei(int nx, int ny, int nz, 
                             double centx, double centy, double centz,
                             double rad, cuDoubleComplex *comp_d, 
                             cuDoubleComplex *phi_d, cuDoubleComplex *dfdphi_d,
                             double c_beta_eq, double c_alpha_eq, double c0, 
                             int *occupancy)
{

   int i = threadIdx.x + blockIdx.x*blockDim.x;
   int j = threadIdx.y + blockIdx.y*blockDim.y;
   int k = threadIdx.z + blockIdx.z*blockDim.z;
  
   int idx = k + nz*(j + i*ny);   

   if ((pow((double)(i - centx), 2.0) + 
        pow((double)(j - centy), 2.0) +
        pow((double)(k - centz), 2.0)) <= 
       (pow(2.0*rad,2.0))){

       if ((pow((double)(i - centx), 2.0) + 
            pow((double)(j - centy), 2.0) +
            pow((double)(k - centz), 2.0)) <= 
            pow(rad,2.0)) {

          comp_d[idx].x   = c_beta_eq;
          comp_d[idx].y   = 0.0;
          phi_d[idx].x    = 1.0;
          phi_d[idx].y    = 0.0;
          dfdphi_d[idx].x = phi_d[idx].x;
          dfdphi_d[idx].y = phi_d[idx].y;

       }

       else{

          double length;
          length = sqrt( pow((double)i - centx, 2.0) +
                         pow((double)j - centy, 2.0) +
                         pow((double)k - centz, 2.0));

          double ratio; 
          ratio = (length - rad)/rad;

          comp_d[idx].x = ratio*c0 + (1.0-ratio)*c_alpha_eq;
          comp_d[idx].y = 0.0;
                                                        
       }

       occupancy[idx] = 1;

   }
   __syncthreads();
}  


__global__ void FindCriticalNucleus (double f0A, double Vm, double c0, 
                     double c_alpha_eq, double c_beta_eq, 
                     double mu_m, double mu_p, double nu_m, double nu_p, 
                     double epszero, double *ppt_size_d, 
                     double interface_energy, int elast_int)
{

    double  chem_driving_force, elast_driving_force, crit_nucleus;

    chem_driving_force  = -1.0*((f0A/Vm)*(c0-c_alpha_eq)*(c0-c_alpha_eq)
                          +2.0*(c_beta_eq-c0)*(f0A/Vm)*(c0-c_alpha_eq));
    elast_driving_force = (2.0*(0.5*(mu_p+mu_m))*(1.0 + 0.5*(nu_m + nu_p))*
                          epszero*epszero)/(1.0-0.5*(nu_p+nu_m));

    if (elast_int == 1)
      crit_nucleus = -2.0*interface_energy/
                    (chem_driving_force + elast_driving_force);
    else
      crit_nucleus = -2.0*interface_energy/chem_driving_force;

    *ppt_size_d = 1.1*crit_nucleus;

}

__global__ void Initialize_comp_phi(int nx, int ny, int nz, 
                          cuDoubleComplex *comp_d, cuDoubleComplex *phi_d, 
                          cuDoubleComplex *dfdphi_d,
                          double c0)
{

   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   int idx = k + nz*(j + i*ny);

   comp_d[idx].x   = c0;
   comp_d[idx].y   = 0.0;

   phi_d[idx].x    = 0.0;
   phi_d[idx].y    = 0.0;

   dfdphi_d[idx].x = 0.0;
   dfdphi_d[idx].y = 0.0;

}

__global__ void CheckOverlap(int nx, int ny, int nz, 
                             double centx, double centy, double centz,
                             double rad, int *occupancy, int *overlap)
{
   	
   int i = threadIdx.x + blockDim.x*blockIdx.x;
   int j = threadIdx.y + blockDim.y*blockIdx.y;
   int k = threadIdx.z + blockDim.z*blockIdx.z;

   if ((((double)i - centx)*((double)i - centx) + 
        ((double)j - centy)*((double)j - centy) +
        ((double)k - centz)*((double)k - centz))  
        <= ((2.0*rad)*(2.0*rad)))
   {

	if (occupancy[k + nz*(j+i*ny)] == 1)
	   *overlap = 1;
   }

}

void Read_Restart()
{
  FILE *fpread;
  char fr[100];

  sprintf(fr, "conf.%07d", initcount);
  fpread = fopen(fr, "r");
  fread(comp, sizeof(cuDoubleComplex), nx * ny * nz, fpread);
  fread(dfdphi,  sizeof(cuDoubleComplex), nx * ny * nz, fpread);
  fclose(fpread);

}
void Init_Conf(void)
{

   double C11m, C12m, C44m;
   double C11p, C12p, C44p;
   long   SEED = -987349927;
   int    Blocks_X, Blocks_Y, Blocks_Z;
   int    Thread_X, Thread_Y, Thread_Z;

   void   Read_Restart();
   double ran(long *idum);

   //Finding Grid size
   Num_of_blocks = (nx*ny*nz)/(NUM_THREADS_X*NUM_THREADS_Y*NUM_THREADS_Z);

   Blocks_X = (int)ceil(pow(Num_of_blocks, (1.0/3.0)));
   Blocks_Y = Blocks_X;
   Blocks_Z = Blocks_X;

   Thread_X = NUM_THREADS_X;
   Thread_Y = NUM_THREADS_Y;
   Thread_Z = NUM_THREADS_Z;

   Gridsize = dim3(Blocks_X, Blocks_Y, Blocks_Z);
   Blocksize = dim3(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z);

   printf("Grid  Dimensions: (%d, %d, %d)\n", Blocks_X, Blocks_Y, Blocks_Z);
   printf("Block Dimensions: (%d, %d, %d)\n", Thread_X, Thread_Y, Thread_Z);

   if (initflag == 1) {

     //Generating Single particle at center of the box
     if (create_nuclei == 0){

        checkCudaErrors(cudaMemcpy(ppt_size_d, &ppt_size,
              sizeof(double), cudaMemcpyHostToDevice));

        printf("Nondimensional precipitate radius = %lf\n",ppt_size);

        double centx, centy, centz;

        centx = nx/2;
        centy = ny/2;
        centz = nz/2;

        CreateNuclei<<<Gridsize,Blocksize>>>(nx, ny, nz, centx, centy, centz, 
                                             ppt_size, comp_d, phi_d, dfdphi_d,
                                             c_beta_eq, c0);
     }
     
     //Generating random particle distribution for given volume fraction
     else if (create_nuclei == 1)
     {

        int *overlap;
        int overlap_h;

        checkCudaErrors(cudaMalloc((void**)&occupancy, nx*ny*nz*sizeof(int)));
        checkCudaErrors(cudaMemset(occupancy, 0, nx*ny*nz*sizeof(int)));
        checkCudaErrors(cudaMalloc((void**)&overlap, sizeof(int)));
        checkCudaErrors(cudaMemset(overlap, 0, sizeof(int)));

        FindCriticalNucleus<<<1, 1>>>(f0A, Vm, c0, c_alpha_eq,
                                      c_beta_eq, mu_m, mu_p, nu_m, nu_p, 
                                      epszero, ppt_size_d, interface_energy, 
                                      elast_int);

        checkCudaErrors(cudaMemcpy(&ppt_size, ppt_size_d,
             sizeof(double), cudaMemcpyDeviceToHost));

        printf("Precipitate radius = %lf\n",ppt_size);

        ppt_size /= dx;

        //printf("Nondimensional precipitate radius = %lf\n",ppt_size);

        checkCudaErrors(cudaMemcpy(ppt_size_d, &ppt_size,
             sizeof(double), cudaMemcpyHostToDevice));

	Initialize_comp_phi<<<Gridsize,Blocksize>>>(nx, ny, nz, comp_d, phi_d, 
                           dfdphi_d, c0);

        double vol = (double) nx*ny*nz;
        double vol_of_particle = (4.0/3.0)*CUDART_PI*ppt_size*ppt_size*
                                  ppt_size;

        int Nofpart = ceil(vol*vf/vol_of_particle);
        printf("Number of particles = %d\n", Nofpart);
        
        int part = 1;

        while (part <= Nofpart){


          double centx = (double) nx*ran(&SEED);         
          double centy = (double) ny*ran(&SEED);
          double centz = (double) nz*ran(&SEED);

          double mdev = 0.05*ppt_size;
          double rad  = ppt_size + (2.0*ran(&SEED) - 1.0)*mdev;

          if ( ((centx - rad) < 0.0) || ((centx + rad) > (nx-1)) ||
               ((centy - rad) < 0.0) || ((centy + rad) > (ny-1)) ||
               ((centz - rad) < 0.0) || ((centz + rad) > (nz-1)) )
               continue; 

          CheckOverlap<<<Gridsize,Blocksize>>>(nx, ny, nz, centx, centy, centz,
                                               rad, occupancy, overlap);

          checkCudaErrors (cudaMemcpy(&overlap_h, overlap, sizeof(int), 
                           cudaMemcpyDeviceToHost));          
 
          if (overlap_h == 0){

            CreateRandomNuclei<<<Gridsize,Blocksize>>>(nx, ny, nz, 
                        centx, centy, centz, rad, comp_d, phi_d, dfdphi_d, 
                        c_beta_eq, c_alpha_eq, c0, occupancy);

          }

          else
          {
            checkCudaErrors(cudaMemset(overlap, 0, sizeof(int)));
            continue;
          }

          part++;
        } 
        cudaFree(overlap);
        cudaFree(occupancy);
     }
     //Generate precipitate of critical nucleus size at center of box         
     else if (create_nuclei == 2){


        double centx, centy, centz;

        centx = nx/2;
        centy = ny/2;
        centz = nz/2;

        FindCriticalNucleus<<<1, 1>>>(f0A, Vm, c0, c_alpha_eq,
                                      c_beta_eq, mu_m, mu_p, nu_m, nu_p, 
                                      epszero, ppt_size_d, interface_energy, 
                                      elast_int);

        checkCudaErrors(cudaMemcpy(&ppt_size, ppt_size_d,
              sizeof(double), cudaMemcpyDeviceToHost));

        printf("Nondimensional precipitate radius = %lf\n",ppt_size/dx);
        CreateNuclei<<<Gridsize,Blocksize>>>(nx, ny, nz, centx, centy, centz, 
                                             ppt_size, comp_d, phi_d, dfdphi_d,
                                             c_beta_eq, c0);
     }
                
   }
   if (initflag == 0){
     Read_Restart();
   
      checkCudaErrors(cudaMemcpy(comp_d, comp,
             nx*ny*nz*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(phi_d, dfdphi,
             nx*ny*nz*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(dfdphi_d, dfdphi,
             nx*ny*nz*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice));

 
   }

   checkCudaErrors(cudaMemcpy(comp, comp_d,
         nx*ny*nz*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost));
   checkCudaErrors(cudaMemcpy(dfdphi, dfdphi_d,
         nx*ny*nz*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost));

   for (int m = 0; m < 6; m++) {
     for (int n = 0; n < 6; n++) {
       Cp[m][n] = 0.0;
       Cm[m][n] = 0.0;
     }
   }

   if (mu_m == mu_p && nu_m == nu_p && Az_m == Az_p)
       inhom = 0;
   else
       inhom = 1;

   C11m = mu_m*(2.0*(2.0 + Az_m)/(1.0 + Az_m) -
 	       (1.0 - 4.0*nu_m)/(1.0-2.0*nu_m));
   C12m = mu_m*(2.0*(Az_m/(1.0 + Az_m)) -
               (1.0 - 4.0*nu_m)/(1.0 - 2.0*nu_m));
   C44m = mu_m*(2.0*Az_m/(1.0 + Az_m));

   C11p = mu_p*(2.0*(2.0 + Az_p)/(1.0 + Az_p) -
 	      (1.0 - 4.0*nu_p)/(1.0 - 2.0 * nu_p));
   C12p = mu_p*(2.0*(Az_p/(1.0 + Az_p)) -
	      (1.0 - 4.0*nu_p)/(1.0 - 2.0 * nu_p));
   C44p = mu_p*(2.0*Az_p/(1.0 + Az_p));

   printf("c11m=%le\t c12m=%le\t c44m=%le\n", C11m, C12m, C44m);
   printf("c11p=%le\t c12p=%le\t c44p=%le\n", C11p, C12p, C44p);

   Chom11 = 0.5*(C11p + C11m);    
   Chom12 = 0.5*(C12p + C12m);    
   Chom44 = 0.5*(C44p + C44m);    

   Chet11 = 0.5*(C11p - C11m);    
   Chet12 = 0.5*(C12p - C12m);    
   Chet44 = 0.5*(C44p - C44m);

   dkx = 2.0 * CUDART_PI / ((double) nx * dx);
   dky = 2.0 * CUDART_PI / ((double) ny * dy);
   dkz = 2.0 * CUDART_PI / ((double) nz * dz);

   sizescale = 1.0/(double)(nx*ny*nz);

}


#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran(long *idum)
{
	int j;
	long k;
	static long iy=0;
	static long iv[NTAB];
	double temp;

	if (*idum <= 0 || !iy) {
	   if (-(*idum) < 1) *idum=1;
	   else *idum = -(*idum);
           for (j=NTAB+7;j>=0;j--) {
               k=(*idum)/IQ;
              *idum=IA*(*idum-k*IQ)-IR*k;
              if (*idum < 0) *idum += IM;
              if (j < NTAB) iv[j] = *idum;
           }
           iy=iv[0];
        }
        k=(*idum)/IQ;
       *idum=IA*(*idum-k*IQ)-IR*k;
        if (*idum < 0) *idum += IM;
        j=iy/NDIV;
        iy=iv[j];
        iv[j] = *idum;
        if ((temp=AM*iy) > RNMX) return RNMX;
        else return temp;
}
#undef IA
#undef IM
#undef AM
#undef IQ
#undef IR
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX
