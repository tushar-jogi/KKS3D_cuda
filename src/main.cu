#include"../include/binary.h"

int main (void)
{   
    //Function declarations
    void Get_Input_Parameters (char *fnin, char *fnout);
    void Init_Conf ();
    void Evolve ();

    size_t  complex_size, double_size, float_size, complex_size_elast;
    char finput[15] = "bin1ary";
    char fnin[30], fnout[30];
    FILE *fp;

    feenableexcept(FE_DIVBYZERO|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW);

    if (!(fp = fopen (finput, "r"))) {
      printf ("File:%s could not be opened\n", finput);
      exit (EXIT_FAILURE);
    }
    if(fscanf (fp, "%s", fnin)==1){
      printf("Input Parameters Filename:%s\n",fnin);
    }
    if(fscanf (fp, "%s", fnout)==1){
      printf("Output Parameters Filename:%s\n",fnout);
    }
    if (!(fpout = fopen (fnout, "w"))) {
      printf ("File:%s could not be opened\n", fnout);
      exit (EXIT_FAILURE);
    }

    fclose (fp);
    fclose (fpout);

    //Reading simulation parameters
    Get_Input_Parameters (fnin, fnout);
    
    checkCudaErrors( cudaMalloc((void**)&nx_d, sizeof(int)));
    checkCudaErrors( cudaMalloc((void**)&ny_d, sizeof(int)));
    checkCudaErrors( cudaMalloc((void**)&nz_d, sizeof(int)));
    checkCudaErrors( cudaMalloc((void**)&elast_int_d, sizeof(int)));
    checkCudaErrors( cudaMalloc((void**)&dt_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&c0_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&w_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&kappa_phi_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&diffusivity_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&relax_coeff_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&ppt_size_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&c_beta_eq_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&c_alpha_eq_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&epszero_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&sizescale_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&Chom11_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&Chom12_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&Chom44_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&Chet11_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&Chet12_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&Chet44_d, sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&sigappl_v_d, 6*sizeof(float)));

    checkCudaErrors(cudaMemcpy(nx_d, &nx, sizeof(int),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ny_d, &ny, sizeof(int),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(nz_d, &nz, sizeof(int),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(elast_int_d, &elast_int, sizeof(int),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dt_d, &dt, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c0_d, &c0, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(diffusivity_d, &diffusivity, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(relax_coeff_d, &relax_coeff, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ppt_size_d, &ppt_size, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_beta_eq_d, &c_beta_eq, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(c_alpha_eq_d, &c_alpha_eq, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sigappl_v_d, sigappl_v, 6*sizeof(float),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(epszero_d, &epszero, sizeof(float),
          cudaMemcpyHostToDevice));

    //sizes of complex and double variables
    complex_size = nx*ny*nz*sizeof(cuDoubleComplex);
    complex_size_elast = nx*ny*nz*sizeof(cufftComplex);
    double_size  = nx*ny*nz*sizeof(double);
    float_size   = nx*ny*nz*sizeof(float);


    //Allocation of global variables
    comp     = (cuDoubleComplex*) malloc (complex_size);
    dfdphi   = (cuDoubleComplex*) malloc (complex_size); 

    checkCudaErrors(cudaMalloc((void**)&comp_d,complex_size));
    checkCudaErrors(cudaMalloc((void**)&phi_d,complex_size));
    checkCudaErrors(cudaMalloc((void**)&dfdphi_d,complex_size));
    checkCudaErrors(cudaMalloc((void**)&dfdc_d,complex_size));

    //Allocation of elasticity varaibles
    if (elast_int == 1){

      checkCudaErrors(cudaMalloc((void**)&ux_d, complex_size_elast));
      checkCudaErrors(cudaMalloc((void**)&uy_d, complex_size_elast));
      checkCudaErrors(cudaMalloc((void**)&uz_d, complex_size_elast));
      checkCudaErrors(cudaMalloc((void**)&dfeldphi_d,complex_size));
      checkCudaErrors(cudaMalloc((void**)&omega_v0, float_size));
      checkCudaErrors(cudaMalloc((void**)&omega_v1, float_size));
      checkCudaErrors(cudaMalloc((void**)&omega_v2, float_size));
      checkCudaErrors(cudaMalloc((void**)&omega_v3, float_size));
      checkCudaErrors(cudaMalloc((void**)&omega_v4, float_size));
      checkCudaErrors(cudaMalloc((void**)&omega_v5, float_size));

    }

    //Generating initial profile
    Init_Conf();

    checkCudaErrors(cudaMalloc((void**)&kx_d, nx*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&ky_d, ny*sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&kz_d, nz*sizeof(double)));

    kappa_phi = 3.0/(2.0*alpha) *(interface_energy*Ln);
    printf("Kappa phi = %lf\n", kappa_phi);
    
    w = 6.0 * alpha * interface_energy / Ln;
    printf("Barrier potential = %lf\n", w);
     
    checkCudaErrors(cudaMemcpy(w_d,  &w,  sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(kappa_phi_d,  &kappa_phi,  sizeof(double),
          cudaMemcpyHostToDevice));

    sim_time = 0.0;

    calc_uzero = 1; 

    //Declaring fft plan
    cufftPlan3d(&plan, nx, ny, nz, CUFFT_Z2Z); 
    cufftPlan3d(&elast_plan, nx, ny, nz, CUFFT_C2C); 

    cudaDeviceSynchronize();
    //call to evolve
    Evolve ();

    free (comp);
    free (dfdphi);
    cudaFree (comp_d);
    cudaFree (phi_d);
    cudaFree (dfdc_d);
    cudaFree (kx_d);
    cudaFree (ky_d);
    cudaFree (kz_d);
    cudaFree (dfdphi_d);

    cudaFree(nx_d );
    cudaFree(ny_d );
    cudaFree(nz_d );
    cudaFree(dt_d );
    cudaFree(c0_d );
    cudaFree(w_d );
    cudaFree(kappa_phi_d );
    cudaFree(ppt_size_d );
    cudaFree(c_beta_eq_d );
    cudaFree(c_alpha_eq_d );
    cudaFree(epszero_d );
    cudaFree(Chom11_d );
    cudaFree(Chom12_d );
    cudaFree(Chom44_d );
    cudaFree(Chet11_d );
    cudaFree(Chet12_d );
    cudaFree(Chet44_d );
    cudaFree(sigappl_v_d );

    if (elast_int == 1){

      cudaFree (ux_d);
      cudaFree (uy_d);
      cudaFree (uz_d);
      cudaFree (dfeldphi_d);
      cudaFree (omega_v0);
      cudaFree (omega_v1);
      cudaFree (omega_v2);
      cudaFree (omega_v3);
      cudaFree (omega_v4);
      cudaFree (omega_v5);

    }

    return 0;
}

#include "get_input.cu"
#include "init_conf.cu"
#include "evolve.cu"
