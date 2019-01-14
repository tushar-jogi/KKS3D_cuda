#include"../include/binary.h"

int main (int argc, char*argv[])
{   
    //Function declarations
    void Get_Input_Parameters (char *fnin, char *fnout);
    void Init_Conf ();
    void Evolve ();
    void Usage(void);
     

    size_t  complex_size;
    char finput[15] = "bin1ary";
    char fnin[30]="InputParams", fnout[30]="OutParams";
    FILE *fp; 
    if (argc > 1){
    	for (int i=0; i<argc; i++){

        	if (strcmp(argv[i],"-i")==0 && argc == 3){ 
           		strcpy(fnin, argv[i+1]);
           		printf("Reading Input Parameters from %s\n", fnin);
                        break;
        	}
        	else if (strcmp(argv[i],"-i")==0 && argc==2){
           		printf("Input file not provided\n");
           		Usage(); 
                        exit (EXIT_FAILURE);
        	}
        	else if (strcmp(argv[i],"--help")==0){
           		Usage(); 
                        exit (EXIT_FAILURE);
        	}
        
    	}
    }
    else{
      Usage();
    

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
    }
    //Reading simulation parameters
    Get_Input_Parameters (fnin, fnout);
    
    checkCudaErrors( cudaMalloc((void**)&ppt_size_d, sizeof(double)));
    checkCudaErrors( cudaMalloc((void**)&sigappl_v_d, 6*sizeof(float)));

    checkCudaErrors(cudaMemcpy(ppt_size_d, &ppt_size, sizeof(double),
          cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(sigappl_v_d, sigappl_v, 6*sizeof(float),
          cudaMemcpyHostToDevice));

    //sizes of complex and double variables
    complex_size = nx*ny*nz*sizeof(cuDoubleComplex);

    //Allocation of global variables
    comp     = (cuDoubleComplex*) malloc (complex_size);
    dfdphi   = (cuDoubleComplex*) malloc (complex_size); 

    checkCudaErrors(cudaMalloc((void**)&comp_d, complex_size));
    checkCudaErrors(cudaMalloc((void**)&phi_d, complex_size));
    checkCudaErrors(cudaMalloc((void**)&dfdphi_d, complex_size));

  if (elast_int == 1){
     checkCudaErrors(cudaMalloc((void**)&ux_d, complex_size));
     checkCudaErrors(cudaMalloc((void**)&uy_d, complex_size));
     checkCudaErrors(cudaMalloc((void**)&uz_d, complex_size));
     checkCudaErrors(cudaMalloc((void**)&dfeldphi_d, complex_size));
  }
    //Generating initial profile
    Init_Conf();

    kappa_phi = 3.0/(2.0*alpha) *(interface_energy*Ln);
    printf("Kappa phi = %lf\n", kappa_phi);
    
    w = 6.0 * alpha * interface_energy / Ln;
    printf("Barrier potential = %lf\n", w);

    sim_time = 0.0;

    calc_uzero = 1; 

    //Declaring fft plan
    cufftPlan3d(&plan, nx, ny, nz, CUFFT_Z2Z); 
    cufftPlan3d(&elast_plan, nx, ny, nz, CUFFT_Z2Z); 

    cudaDeviceSynchronize();
    //call to evolve
    Evolve ();

    free (comp);
    free (dfdphi);
    cudaFree (comp_d);
    cudaFree (phi_d);
    cudaFree (dfdphi_d);

    cudaFree(ppt_size_d);
    cudaFree(sigappl_v_d);

    if (elast_int == 1){

      cudaFree (ux_d);
      cudaFree (uy_d);
      cudaFree (uz_d);
      cudaFree (dfeldphi_d);

    }

    return 0;
}

void Usage(void){
     printf("This is the help for KKS3D\n"
            "Options are:\n"
            "    --help: display what you are reading\n"
            "    -i filename: opens the filename for input parameters\n");

}

#include "get_input.cu"
#include "init_conf.cu"
#include "evolve.cu"
