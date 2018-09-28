void Evolve(void)
{
  void Output_Conf (int steps);
  void Calc_uzero(void);
  void InhomElast(void);
  void HomElast(void);

  int              loop_condition, count;
  int              complex_size, double_size;
  long             SEED = -346739021;
  double           kpow2, noiselevel = 1.0e-02;
  double           *tempreal ;
  double           lhs, lhse;
  double           err, maxerror;
  double           f0AVminv, f0BVminv;
  double           cbeta, calpha;
  double           adjomega[6], invomega_v[6], det_omega;
  double           n[3];
  double           rphi_new ;
  double           rphi ;
  double           A_by_B, B_by_A;
  cuDoubleComplex  rhs, rhse;

  complex_size = nx*ny*sizeof(cuDoubleComplex);
  double_size  = nx*ny*sizeof(double);

  f0AVminv = f0A * (1.0/Vm) ;
  f0BVminv = f0B * (1.0/Vm) ;
 
  A_by_B = f0AVminv/f0BVminv;
  B_by_A = f0BVminv/f0AVminv;

  kappa_phi = 3.0/(2.0*alpha) *(interface_energy*Ln);
  printf("Kappa_phi = %le \n", kappa_phi);

  w = 6.0 * alpha * interface_energy / Ln;
  printf("Barrier height = %le (J/m^3)\n",w);

  tempreal = (double*) malloc(double_size);

  for (int l = 0; l < nx*ny; l++) {
    tempreal[l]  = Re(comp[l]);
  }

  for (int i = 0 ; i < nx ; i++ ) {
    if (i < nx_half) 
      kx[i] = (double) i * dkx;
    else 
      kx[i] = (double)( i - nx ) * dkx;
  } 

  for (int j = 0; j < ny; j++){
    if (j < ny_half)
      ky[j] = (double)j * dky;
    else
      ky[j] = (double)(j-ny) * dky;
  }

  if (elast_int == 1){

    for (int ii = 0; ii < 6; ii++){
      invomega_v[ii] = 0.0;   
    }

    for (int i = 0; i < nx; i++){
      for (int j = 0; j < ny; j++){
          
        int l = (j+i*ny); 

        n[0] = kx[i];
        n[1] = ky[j];
        n[2] = 0.0;

        invomega_v[0] = Chom11*n[0]*n[0] + Chom44*n[1]*n[1] +
                        Chom44*n[2]*n[2];
        invomega_v[1] = Chom44*n[0]*n[0] + Chom11*n[1]*n[1] +
                        Chom44*n[2]*n[2];
        invomega_v[2] = Chom44*n[0]*n[0] + Chom44*n[1]*n[1] +
                        Chom11*n[2]*n[2];
        invomega_v[3] = (Chom12 + Chom44)*n[1]*n[2];
        invomega_v[4] = (Chom12 + Chom44)*n[0]*n[2];
        invomega_v[5] = (Chom12 + Chom44)*n[0]*n[1];
         
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
          omega_v[0][l] = (1.0/det_omega)*adjomega[0];
          omega_v[1][l] = (1.0/det_omega)*adjomega[1];
          omega_v[2][l] = (1.0/det_omega)*adjomega[2];
          omega_v[3][l] = (1.0/det_omega)*adjomega[3];
          omega_v[4][l] = (1.0/det_omega)*adjomega[4];
          omega_v[5][l] = (1.0/det_omega)*adjomega[5];
        } 
        else{
          omega_v[0][l] = 0.0;
          omega_v[1][l] = 0.0;
          omega_v[2][l] = 0.0;
          omega_v[3][l] = 0.0;
          omega_v[4][l] = 0.0;
          omega_v[5][l] = 0.0;
        }
      }  
    }
  }

  if (inhom != 1)
  {

    S_v[0][0] = S_v[1][1] = S_v[2][2] = (Chom11+Chom12)/
                                        (Chom11*Chom11 + 
                                         Chom11*Chom12 -
                                     2.0*Chom12*Chom12);
    S_v[0][1] = S_v[0][2] = S_v[1][2] = (-1.0*Chom12)/
                                        (Chom11*Chom11 +
                                         Chom11*Chom12 -
                                     2.0*Chom12*Chom12);
    S_v[3][3] = S_v[4][4] = S_v[5][5] = 1.0/Chom44;
    S_v[1][0] = S_v[2][0] = S_v[2][1] = S_v[0][1];

    S_v[0][3] = S_v[0][4] = S_v[0][5] = 0.0;
    S_v[1][3] = S_v[1][4] = S_v[1][5] = 0.0;
    S_v[2][3] = S_v[2][4] = S_v[2][5] = 0.0;
    S_v[3][0] = S_v[3][1] = S_v[3][2] = S_v[3][4] = S_v[3][5] = 0.0;
    S_v[4][0] = S_v[4][1] = S_v[4][2] = S_v[4][3] = S_v[4][5] = 0.0;
    S_v[5][0] = S_v[5][1] = S_v[5][2] = S_v[5][3] = S_v[5][4] = 0.0;

  } 

  Err=cudaMalloc((void**)&phi_d, complex_size);

  checkCudaErrors(cudaMemcpy(phi_d, phi, complex_size, 
                  cudaMemcpyHostToDevice));
  if(cufftExecZ2Z(plan, (cuDoubleComplex*)phi_d, (cuDoubleComplex*)phi_d,
                  CUFFT_FORWARD)!= CUFFT_SUCCESS)
  printf("fft failed");

  checkCudaErrors(cudaMemcpy(phi, phi_d, complex_size, 
                  cudaMemcpyDeviceToHost));
  cudaFree(phi_d);

  cudaMalloc((void**)&comp_d,complex_size);
  cudaMalloc((void**)&dfdphi_d,complex_size);

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
      Output_Conf(count);
    }
 
    if (count > num_steps || loop_condition == 0)
      break;
   
    printf("Iteration No: %d\n",iteration);

      //Finding elastic driving force in real space
    if (elast_int == 1 && count >= time_elast ){

      if (count == initcount)
        Calc_uzero();
      if (inhom == 1)
        InhomElast();
      else
        HomElast();
    }

    //Finding gradphi
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {

        int l = (j+i*ny);

        double K[2];

        if (i != nx_half)
         K[0] = kx[i];
        else 
         K[0] = 0.0;

        if (j != ny_half)
         K[1] = ky[j];
        else
         K[1] = 0.0;

        gradphix[l].x = -1.0*K[0]*phi[l].y; 
        gradphix[l].y =  K[0]*phi[l].x;
        gradphiy[l].x = -1.0*K[1]*phi[l].y;
        gradphiy[l].y =  K[1]*phi[l].x;

      }
    }
 
    Err=cudaMalloc((void**)&gradphix_d, complex_size);
    Err=cudaMalloc((void**)&gradphiy_d, complex_size);

    cudaMemcpy(gradphix_d, gradphix, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gradphiy_d, gradphiy, complex_size, cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan, (cuDoubleComplex*)gradphix_d,
             (cuDoubleComplex*)gradphix_d, CUFFT_INVERSE);
    cufftExecZ2Z(plan, (cuDoubleComplex*)gradphiy_d,
             (cuDoubleComplex*)gradphiy_d, CUFFT_INVERSE);

    cudaMemcpy(gradphix, gradphix_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradphiy, gradphiy_d, complex_size, cudaMemcpyDeviceToHost);

    cudaFree(gradphix_d);
    cudaFree(gradphiy_d);

    for (int i = 0; i < nx; i++ ){
      for (int j = 0; j < ny; j++ ){

        int l = (j+i*ny);

        gradphix[l].x *= sizescale;
        gradphiy[l].x *= sizescale; 
        gradphix[l].y *= sizescale;
        gradphiy[l].y *= sizescale; 

      }
    }
  
    //Here dfdc and dfdphi are in real space. Note that dfdphi 
    //stores phi in real space.

    //Finidng dfdphi 
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
           
        int l = (j+i*ny);

        double  interp_phi, interp_prime, g_prime;
        double  ctemp, etemp;
        double  f_alpha, f_beta, mubar;

   
        ctemp  = comp[l].x;
        etemp  = dfdphi[l].x;
   
        interp_phi   = etemp * etemp * etemp * 
                      (6.0 * etemp * etemp - 15.0 * etemp + 10.0);
        interp_prime = 30.0*etemp*etemp*pow((1.0 - etemp), 2.0);  
        g_prime      = 2.0*etemp*(1.0 - etemp)*(1.0 - 2.0 * etemp);

        calpha       = (ctemp - interp_phi * 
                       (c_beta_eq - c_alpha_eq * A_by_B))/
                       (interp_phi*A_by_B + (1.0-interp_phi));

        cbeta        = (ctemp + (1.0 - interp_phi) * 
                       (B_by_A * c_beta_eq - c_alpha_eq))/
                       (interp_phi + B_by_A*(1.0 - interp_phi));

        comp[l].x    = calpha*(1.0 - interp_phi) + 
                       cbeta*interp_phi; 
        comp[l].y    = 0.0;

        f_alpha      = f0AVminv*(calpha - c_alpha_eq)* 
                       (calpha - c_alpha_eq);
        f_beta       = f0BVminv*(cbeta - c_beta_eq)* 
                       (cbeta - c_beta_eq);
   
        varmobx[l].x = diffusivity*interp_prime* 
                       (calpha - cbeta) * gradphix[l].x;
        varmobx[l].y = diffusivity*interp_prime* 
                       (calpha - cbeta) * gradphix[l].y;
        varmoby[l].x = diffusivity*interp_prime* 
                       (calpha - cbeta) * gradphiy[l].x;
        varmoby[l].y = diffusivity*interp_prime* 
                       (calpha - cbeta) * gradphiy[l].y;
   
        mubar = 2.0 * f0AVminv * (calpha - c_alpha_eq);

        dfdphi[l].x = interp_prime*(f_beta - f_alpha + 
                      (calpha - cbeta)*mubar) + 
                      w*g_prime; 
        dfdphi[l].y = 0.0;

      }
    }

    checkCudaErrors(cudaMalloc((void**)&varmobx_d, complex_size));
    checkCudaErrors(cudaMalloc((void**)&varmoby_d, complex_size));

    checkCudaErrors(cudaMemcpy(varmobx_d, varmobx, complex_size,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(varmoby_d, varmoby, complex_size,
                               cudaMemcpyHostToDevice));

    if (cufftExecZ2Z(plan, (cuDoubleComplex*)varmobx_d,
       (cuDoubleComplex*)varmobx_d, CUFFT_FORWARD) != CUFFT_SUCCESS)
       printf("fft failed\n");
    if (cufftExecZ2Z(plan, (cuDoubleComplex*)varmoby_d,
       (cuDoubleComplex*)varmoby_d, CUFFT_FORWARD) != CUFFT_SUCCESS)
       printf("fft failed\n");

    cudaMemcpy(varmobx, varmobx_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(varmoby, varmoby_d, complex_size, cudaMemcpyDeviceToHost);

    cudaFree(varmobx_d);
    cudaFree(varmoby_d);

    //dfdc in Fourier space
    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {

        int l = (j+i*ny);

        double K[2];

        if (i != nx_half)
          K[0] = kx[i];
        else
          K[0] = 0.0;

        if (j != ny_half)
          K[1] = ky[j];
        else
          K[1] = 0.0;

        dfdc[l].x = (K[0] * varmobx[l].y)+
                    (K[1] * varmoby[l].y);

        dfdc[l].y = (K[0] * varmobx[l].x)+
                    (K[1] * varmoby[l].x);

      }
    }

    //Take  dfdphi, comp and dfeldphi to Fourier space 
    cudaMemcpy(comp_d, comp, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dfdphi_d, dfdphi, complex_size, cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan, (cuDoubleComplex*)comp_d,
          (cuDoubleComplex*)comp_d, CUFFT_FORWARD);
    cufftExecZ2Z(plan, (cuDoubleComplex*)dfdphi_d,
          (cuDoubleComplex*)dfdphi_d, CUFFT_FORWARD);

    cudaMemcpy(comp, comp_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dfdphi, dfdphi_d, complex_size, cudaMemcpyDeviceToHost);

    if (elast_int == 1 && count > time_elast){

      cudaMalloc((void**)&dfeldphi_d, complex_size);
          
      cudaMemcpy(dfeldphi_d, dfeldphi, complex_size,
            cudaMemcpyHostToDevice);

      cufftExecZ2Z(plan, (cuDoubleComplex*)dfeldphi_d,
          (cuDoubleComplex*)dfeldphi_d, CUFFT_FORWARD);
          
      cudaMemcpy(dfeldphi, dfeldphi_d, complex_size, 
            cudaMemcpyDeviceToHost);
      cudaFree(dfeldphi_d);

    }

    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
 
        int l = (j+i*ny);

        kpow2 = kx[i]*kx[i] + ky[j]*ky[j];

        //update composition
        lhs = 1.0 + diffusivity*kpow2*dt ;   
        rhs.x = comp[l].x - dt*dfdc[l].x;
        rhs.y = comp[l].y + dt*dfdc[l].y;
        comp[l].x = rhs.x/lhs; 
        comp[l].y = rhs.y/lhs;
  
        //update phi
        lhse = 1.0 + 2.0*relax_coeff*kappa_phi*kpow2*dt;
        if (elast_int == 1 && count > time_elast){
          rhse.x  = phi[l].x - relax_coeff*dt*
                   (dfdphi[l].x + dfeldphi[l].x);
          rhse.y  = phi[l].y - relax_coeff*dt*
                   (dfdphi[l].y + dfeldphi[l].y);
        }
        else{
          rhse.x  = phi[l].x - relax_coeff*dt*
                   (dfdphi[l].x);
          rhse.y  = phi[l].y - relax_coeff*dt*
                   (dfdphi[l].y);
        }

        phi[l].x = rhse.x/lhse;
        phi[l].y = rhse.y/lhse;
        dfdphi[l].x = phi[l].x;
        dfdphi[l].y = phi[l].y;

      }
    }

    cudaMemcpy(comp_d, comp, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dfdphi_d, dfdphi, complex_size, cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan, (cuDoubleComplex*)comp_d,
          (cuDoubleComplex*)comp_d, CUFFT_INVERSE);
    cufftExecZ2Z(plan, (cuDoubleComplex*)dfdphi_d,
           (cuDoubleComplex*)dfdphi_d, CUFFT_INVERSE);

    cudaThreadSynchronize();

    cudaMemcpy(comp, comp_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(dfdphi, dfdphi_d, complex_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        comp[(j+i*ny)].x   *= sizescale;
        comp[(j+i*ny)].y   *= sizescale;
        dfdphi[(j+i*ny)].x *= sizescale;
        dfdphi[(j+i*ny)].y *= sizescale;
      }
    }

    //if (count%20)
    //{
    for (int i = 0; i < nx; i++){
      for (int j = 0; j < ny; j++){
        if (Re(dfdphi[j+i*ny])>=0.05 && 
              Re(dfdphi[j+i*ny])<=0.95){

            double randnum;
            randnum = 2.0*ran(&SEED) - 1.0;
	    dfdphi[j+i*ny].x = dfdphi[j+i*ny].x + randnum*noiselevel;    	
        }
      }
    }
	
    cudaMemcpy(dfdphi_d, dfdphi, complex_size, cudaMemcpyHostToDevice);
    cufftExecZ2Z(plan, (cuDoubleComplex*)dfdphi_d,
           (cuDoubleComplex*)dfdphi_d, CUFFT_FORWARD);
    cudaMemcpy(phi, dfdphi_d, complex_size, cudaMemcpyDeviceToHost);

    //}
    // Check for convergence 
    maxerror = 0.0;

    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {

        int l = (j + i*ny);
        err = fabs (tempreal[l] - Re(comp[l]));
        if (err > maxerror)
        maxerror = err;

      }
    }

    if (fabs(comp[0].x - c0) > 1.0e-03){
      printf("End of simulation\n");
      loop_condition = 0;
    } 

    if (maxerror <= Tolerance) {
      printf ("maxerror=%lf\tnumbersteps=%d\n", maxerror, count);
      loop_condition = 0;
    }

    sim_time = sim_time + dt;

    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < ny; j++) {
        tempreal[(j+i*ny)] = Re(comp[(j+i*ny)]);
      }
    }

    iteration = iteration + 1;
  }//time loop ends

  cudaFree(comp_d);
  cudaFree(dfdphi_d);
}

#include "out_conf.cu"
#include "calc_uzero.cu"
#include "homelast.cu"
#include "inhomelast.cu"
