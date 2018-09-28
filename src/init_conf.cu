__global__ void CreateNuclei(int *nx_d, int *ny_d, int *nz_d, 
                             double *ppt_size_d, cuDoubleComplex *comp_d, 
                             cuDoubleComplex *phi_d, cuDoubleComplex *dfdphi_d,
                             double *c_beta_eq_d, double *c0_d)
{

   int i = threadIdx.x + blockIdx.x*blockDim.x;
   int j = threadIdx.y + blockIdx.y*blockDim.y;
   int k = threadIdx.z + blockIdx.z*blockDim.z;

   int idx = k + (*nz_d)*(j + i*(*ny_d));

   if ((i - (*nx_d/2))*(i - (*nx_d/2)) + 
       (j - (*ny_d/2))*(j - (*ny_d/2)) + 
       (k - (*nz_d/2))*(k - (*nz_d/2)) <=
       ((*ppt_size_d)*(*ppt_size_d)) )
     {  
        comp_d[idx].x   = (*c_beta_eq_d);
        comp_d[idx].y   = 0.0;
        phi_d[idx].x    = 1.0;
        phi_d[idx].y    = 0.0;
        dfdphi_d[idx].x = phi_d[idx].x;
        dfdphi_d[idx].y = phi_d[idx].y;
     }
   else
     {
        comp_d[idx].x   = (*c0_d);
        comp_d[idx].y   = 0.0;
        phi_d[idx].x    = 0.0;
        phi_d[idx].y    = 0.0;
        dfdphi_d[idx].x = phi_d[idx].x;
        dfdphi_d[idx].y = phi_d[idx].y;
     }

     __syncthreads();
}

void Init_Conf(void)
{
  double C11m, C12m, C44m;
  double C11p, C12p, C44p;
  int    Blocks_X, Blocks_Y, Blocks_Z;
  int    Threads_X, Threads_Y, Threads_Z;
  double chem_driving_force, elast_driving_force, crit_nucleus;
  //Finding Grid size

  Num_of_blocks = (nx*ny*nz)/(NUM_THREADS_X*NUM_THREADS_Y*NUM_THREADS_Z);

  Blocks_X = (int)ceil(pow(Num_of_blocks, (1.0/3.0)));
  Blocks_Y = Blocks_X;
  Blocks_Z = Blocks_X;
  Gridsize = dim3(Blocks_X, Blocks_Y, Blocks_Z);
  Blocksize = dim3(NUM_THREADS_X, NUM_THREADS_Y, NUM_THREADS_Z);

  Threads_X = NUM_THREADS_X;
  Threads_Y = NUM_THREADS_Y;
  Threads_Z = NUM_THREADS_Z;

  printf("Grid Dimensions: (%d, %d, %d)\n",Blocks_X, Blocks_Y, Blocks_Z);
  printf("Block Dimensions: (%d, %d, %d)\n",Threads_X, Threads_Y, Threads_Z);

  if (initflag == 1) {

    if (create_nuclei == 0){

      //Finidng the critical nucleus size
/*
      chem_driving_force = - 1.0*((f0A/Vm)*(c0-c_alpha_eq)*(c0-c_alpha_eq)
                           + 2.0*(c_beta_eq-c0)*(f0A/Vm) * (c0-c_alpha_eq));
      elast_driving_force = (2.0*0.5*(mu_m+mu_p)*(1.0+nu_p))*epszero*epszero/
                            (1.0-nu_p);
      crit_nucleus = -2.0*interface_energy/
                     (chem_driving_force + elast_driving_force);

      printf("Critical nucleus radius = %le\n", crit_nucleus); 
      ppt_size = 1.1*crit_nucleus;

/////////////////////////////////////////////////////////////////////////////   
//                                                                         //
//      //Here the nucleus size decides the dx.                            //
//      //The interface width should be smaller than ppt_size.             //
//      //I assumed that 24 grid points will resolve the nucleus, out of   //
//      //which 16 grid points will be defined for interface.              //   //                                                                         //
/////////////////////////////////////////////////////////////////////////////
*/
      ppt_size /= dx;

      if (ppt_size < 24.0){

        printf("Not enough points to resolve nucleus.\n");
        printf("Decrease dx or increase f0A.\n");
        exit(0);

      }
       
          
      printf("dx=%lf, interface width =%lf\n", dx, Ln);
      printf("Nondimensional precipitate radius = %lf\n",ppt_size);
      checkCudaErrors(cudaMemcpy(ppt_size_d, &ppt_size,
            sizeof(double),cudaMemcpyHostToDevice));

      CreateNuclei<<<Gridsize,Blocksize>>>(nx_d, ny_d, nz_d, ppt_size_d,
                                           comp_d, phi_d, dfdphi_d,
                                           c_beta_eq_d, c0_d);

    }

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

  //Homogeneous C and heterogeneous C
  Chom11  = 0.5*(C11p + C11m);    
  Chom12  = 0.5*(C12p + C12m);    
  Chom44  = 0.5*(C44p + C44m);    

  checkCudaErrors(cudaMemcpy(Chom11_d, &Chom11, sizeof(double),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(Chom12_d, &Chom12, sizeof(double),
        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(Chom44_d, &Chom44, sizeof(double),
      cudaMemcpyHostToDevice));

  Chet11  = 0.5*(C11p - C11m);    
  Chet12  = 0.5*(C12p - C12m);    
  Chet44  = 0.5*(C44p - C44m);

  checkCudaErrors(cudaMemcpy(Chet11_d, &Chet11, sizeof(double),
      cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(Chet12_d, &Chet12, sizeof(double),
      cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(Chet44_d, &Chet44, sizeof(double),
      cudaMemcpyHostToDevice));

  //basis fourier vector
  dkx = 2.0 * CUDART_PI / ((double) nx * dx);
  dky = 2.0 * CUDART_PI / ((double) ny * dy);
  dkz = 2.0 * CUDART_PI / ((double) nz * dz);

  sizescale = 1.0/(double)(nx*ny*nz);
  checkCudaErrors(cudaMemcpy(sizescale_d, &sizescale, sizeof(double),
      cudaMemcpyHostToDevice));

 }
