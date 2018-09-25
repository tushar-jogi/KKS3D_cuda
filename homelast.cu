/**************************************************************************

  Interpolation = 1 : elastic constants as function of composition
  Interpolation = 2 : elastic constants as function of phi 

************************************************************************/
#define TOLERENCE 1.0e-05
#define FALSE 0
#define TRUE  1
void HomElast (void){

   int              ii,jj,converge=1,iter=1;
   int              complex_size, double_size;
   double           avgeigstress_v[6];
   double           nk[3];  
   double           hom_strain_v[6];
   double           omega[6];
   cuDoubleComplex  temp_v[6], stmp_v[6];
   //int              MatrixInvert(double *mat, double *invMat, int order, 
   //               int lda);


   complex_size = sizeof(cuDoubleComplex)*nx*ny;
   double_size  = sizeof(double)*nx*ny;

   str_v0   = (cuDoubleComplex*) malloc (complex_size);
   str_v1   = (cuDoubleComplex*) malloc (complex_size);
   str_v2   = (cuDoubleComplex*) malloc (complex_size);
   str_v3   = (cuDoubleComplex*) malloc (complex_size);
   str_v4   = (cuDoubleComplex*) malloc (complex_size);
   str_v5   = (cuDoubleComplex*) malloc (complex_size);
   ts0      = (cuDoubleComplex*) malloc (complex_size);
   ts1      = (cuDoubleComplex*) malloc (complex_size);
   ts2      = (cuDoubleComplex*) malloc (complex_size);
   ts3      = (cuDoubleComplex*) malloc (complex_size);
   ts4      = (cuDoubleComplex*) malloc (complex_size);
   ts5      = (cuDoubleComplex*) malloc (complex_size);

   for (int m = 0; m < 6; m++) 
     eigenstrain_v[m] = (double*) malloc(double_size); 
   

  /*----------------------------------------------------------------------
   *                        Compliance tensor calculations 
   *---------------------------------------------------------------------*/
   
   /*Inhomogenous part of total stiffness tensor in Voight's form*/ 

   for (ii = 0; ii < 6; ii++)
     avgeigstress_v[ii] = 0.0;

   for (int l = 0; l < (nx*ny); l++){

     double hphi, e_temp;

     e_temp = Re(dfdphi[l]);
           
     hphi = e_temp*e_temp*e_temp*(6.0*e_temp*e_temp - 15.0*e_temp 
            + 10.0);

     eigenstrain_v[0][l] = epszero*hphi;
     eigenstrain_v[1][l] = epszero*hphi;
     eigenstrain_v[2][l] = 0.0;
     eigenstrain_v[3][l] = 0.0;
     eigenstrain_v[4][l] = 0.0;
     eigenstrain_v[5][l] = 0.0;

     avgeigstress_v[0] = avgeigstress_v[0]+
                         Chom11*eigenstrain_v[0][l]+
                         Chom12*eigenstrain_v[1][l]+
                         Chom12*eigenstrain_v[2][l];

     avgeigstress_v[1] = avgeigstress_v[1]+
                         Chom12*eigenstrain_v[0][l]+
                         Chom11*eigenstrain_v[1][l]+
                         Chom12*eigenstrain_v[2][l];

     avgeigstress_v[2] = avgeigstress_v[2]+
                         Chom12*eigenstrain_v[0][l]+
                         Chom12*eigenstrain_v[1][l]+
                         Chom11*eigenstrain_v[2][l];

     avgeigstress_v[3] = avgeigstress_v[3]+
                         Chom44*eigenstrain_v[3][l];

     avgeigstress_v[4] = avgeigstress_v[4]+
                         Chom44*eigenstrain_v[4][l];

     avgeigstress_v[5] = avgeigstress_v[5]+
                         Chom44*eigenstrain_v[5][l];
       
   }

   /*Calculate volume average eigen stress*/
   avgeigstress_v[0] *= sizescale;  
   avgeigstress_v[1] *= sizescale;  
   avgeigstress_v[2] *= sizescale;  
   avgeigstress_v[3] *= sizescale;  
   avgeigstress_v[4] *= sizescale;  
   avgeigstress_v[5] *= sizescale;  

   hom_strain_v[0] = 0.0;
   hom_strain_v[1] = 0.0;
   hom_strain_v[2] = 0.0;
   hom_strain_v[3] = 0.0;
   hom_strain_v[4] = 0.0;
   hom_strain_v[5] = 0.0;
         
   for (ii=0; ii<6; ii++)
     for (jj=0; jj<6; jj++)
        hom_strain_v[ii] += S_v[ii][jj] *
               (sigappl_v[jj] + avgeigstress_v[jj]);

   for (int l=0; l<(nx*ny); l++){
      
      ts0[l].x = Chom11*(eigenstrain_v[0][l]-hom_strain_v[0])+
                 Chom12*(eigenstrain_v[1][l]-hom_strain_v[1])+
                 Chom12*(eigenstrain_v[2][l]-hom_strain_v[2]);

      ts1[l].x = Chom12*(eigenstrain_v[0][l]-hom_strain_v[0])+
                 Chom11*(eigenstrain_v[1][l]-hom_strain_v[1])+ 
                 Chom12*(eigenstrain_v[2][l]-hom_strain_v[2]);

      ts2[l].x = Chom12*(eigenstrain_v[0][l]-hom_strain_v[0])+
                 Chom12*(eigenstrain_v[1][l]-hom_strain_v[1])+
                 Chom11*(eigenstrain_v[2][l]-hom_strain_v[2]);

      ts3[l].x = Chom44*(eigenstrain_v[3][l]-hom_strain_v[3]); 

      ts4[l].x = Chom44*(eigenstrain_v[4][l]-hom_strain_v[4]);

      ts5[l].x = Chom44*(eigenstrain_v[5][l]-hom_strain_v[5]);
             
      ts0[l].y = 0.0;          
      ts1[l].y = 0.0;          
      ts2[l].y = 0.0;          
      ts3[l].y = 0.0;          
      ts4[l].y = 0.0;          
      ts5[l].y = 0.0;          
          
    }

    cudaMalloc((void**)&ts0_d, complex_size);
    cudaMalloc((void**)&ts1_d, complex_size);
    cudaMalloc((void**)&ts2_d, complex_size);
    cudaMalloc((void**)&ts3_d, complex_size);
    cudaMalloc((void**)&ts4_d, complex_size);
    cudaMalloc((void**)&ts5_d, complex_size);

    cudaMemcpy(ts0_d, ts0, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ts1_d, ts1, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ts2_d, ts2, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ts3_d, ts3, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ts4_d, ts4, complex_size, cudaMemcpyHostToDevice);
    cudaMemcpy(ts5_d, ts5, complex_size, cudaMemcpyHostToDevice);

    cufftExecZ2Z(plan,(cuDoubleComplex*)ts0_d,
                (cuDoubleComplex*)ts0_d,CUFFT_FORWARD);
    cufftExecZ2Z(plan,(cuDoubleComplex*)ts1_d,
                (cuDoubleComplex*)ts1_d,CUFFT_FORWARD);
    cufftExecZ2Z(plan,(cuDoubleComplex*)ts2_d,
                (cuDoubleComplex*)ts2_d,CUFFT_FORWARD);
    cufftExecZ2Z(plan,(cuDoubleComplex*)ts3_d,
                (cuDoubleComplex*)ts3_d,CUFFT_FORWARD);
    cufftExecZ2Z(plan,(cuDoubleComplex*)ts4_d,
                (cuDoubleComplex*)ts4_d,CUFFT_FORWARD);
    cufftExecZ2Z(plan,(cuDoubleComplex*)ts5_d,
                (cuDoubleComplex*)ts5_d,CUFFT_FORWARD);

      
    cudaMemcpy(ts0, ts0_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ts1, ts1_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ts2, ts2_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ts3, ts3_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ts4, ts4_d, complex_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ts5, ts5_d, complex_size, cudaMemcpyDeviceToHost);

    cudaFree(ts0_d);
    cudaFree(ts1_d);
    cudaFree(ts2_d);
    cudaFree(ts3_d);
    cudaFree(ts4_d);
    cudaFree(ts5_d);

  /*-----------------------------------------------------------------------
   *                        Refinement of displacement
   *---------------------------------------------------------------------*/
    while (converge != FALSE){

      for (int i = 0; i<(nx); i++){
        for (int j = 0; j < ny; j++){

            nk[0] = kx[i];
            nk[1] = ky[j];
            nk[2] = 0.0;

            int m = (j+i*ny);

            omega[0] = omega_v[0][m];
            omega[1] = omega_v[1][m];
            omega[2] = omega_v[2][m];
            omega[3] = omega_v[3][m];
            omega[4] = omega_v[4][m];
            omega[5] = omega_v[5][m];

            stmp_v[0].x = ts0[m].x;    
            stmp_v[1].x = ts1[m].x;    
            stmp_v[2].x = ts2[m].x;    
            stmp_v[3].x = ts3[m].x;    
            stmp_v[4].x = ts4[m].x;    
            stmp_v[5].x = ts5[m].x;

            stmp_v[0].y = ts0[m].y;    
            stmp_v[1].y = ts1[m].y;    
            stmp_v[2].y = ts2[m].y;    
            stmp_v[3].y = ts3[m].y;    
            stmp_v[4].y = ts4[m].y;    
            stmp_v[5].y = ts5[m].y;

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
           
            unewx[m].x = omega[0] * fk10.y + 
                         omega[5] * fk20.y +
                         omega[4] * fk30.y ;

            unewy[m].x = omega[5] * fk10.y + 
                         omega[1] * fk20.y +
                         omega[3] * fk30.y ;
 
        
            unewx[m].y = -1.0*(omega[0] * fk10.x + 
                               omega[5] * fk20.x +
                               omega[4] * fk30.x );

            unewy[m].y = -1.0*(omega[5] * fk10.x + 
                               omega[1] * fk20.x +
                               omega[3] * fk30.x );
 
        }   
      }

      cudaMalloc((void**)&unewx_d,complex_size);    
      cudaMalloc((void**)&unewy_d,complex_size);    

      cudaMemcpy(unewx_d, unewx, complex_size, cudaMemcpyHostToDevice);
      cudaMemcpy(unewy_d, unewy, complex_size, cudaMemcpyHostToDevice);

      cufftExecZ2Z(plan,(cuDoubleComplex*)unewx_d,
                  (cuDoubleComplex*)unewx_d,CUFFT_INVERSE);
      cufftExecZ2Z(plan,(cuDoubleComplex*)unewy_d,
                  (cuDoubleComplex*)unewy_d,CUFFT_INVERSE);

      cudaMemcpy(unewx, unewx_d, complex_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(unewy, unewy_d, complex_size, cudaMemcpyDeviceToHost);

      cudaFree(unewx_d); 
      cudaFree(unewy_d); 

      for (int l=0; l<(nx*ny); l++){
        
        unewx[l].x *= sizescale;
        unewy[l].x *= sizescale;
        unewz[l].x  = 0.0;

        unewx[l].y *= sizescale;
        unewy[l].y *= sizescale;
        unewz[l].y  = 0.0;
      }
    
      disperror = 0.0;
   
      for (int l=0; l<(nx*ny); l++){
  
        disperror += pow ( (Re( unewx[l] ) - Re( ux[l] )), 2.0) + 
                     pow ( (Re( unewy[l] ) - Re( uy[l] )), 2.0) ;
           
      }

      disperror = sqrt(disperror);

      //printf ("\niter=%d error = %le", iter, disperror);

      if (disperror < TOLERENCE){
        //printf("\nConvergence achieved at %d\n", iter);
        converge = 0;
      }

      iter = iter + 1;

      for (int l=0; l<(nx*ny); l++){
  
            ux[l] = unewx[l];
            uy[l] = unewy[l];
            uz[l] = unewz[l];
      }

    
   }


/*Heterogeneous strain calculation after refinement*/
/*      
   for (int i=0; i<nx; i++){
    for (int j=0; j<ny; j++){
     for (int k=0; k<nz; k++){
    
      int l = k+nz*(j+i*ny);
      nk[0] = kx[i];
      nk[1] = ky[j];
      nk[2] = kz[k];

      str_v0[l] = 0.0 + I*(ux[l]*nk[0]);
      str_v1[l] = 0.0 + I*(uy[l]*nk[1]);
      str_v2[l] = 0.0 + I*(uz[l]*nk[2]);
      str_v3[l] = 0.0 + I*(uy[l]*nk[2] + uz[l]*nk[1]);
      str_v4[l] = 0.0 + I*(ux[l]*nk[2] + uz[l]*nk[0]);
      str_v5[l] = 0.0 + I*(ux[l]*nk[1] + uy[l]*nk[0]);
     }
    }
   }
*/
   cudaMalloc((void**)&unewx_d,complex_size);    
   cudaMalloc((void**)&unewy_d,complex_size);    

   cudaMemcpy(unewx_d,unewx,complex_size,cudaMemcpyHostToDevice);
   cudaMemcpy(unewy_d,unewy,complex_size,cudaMemcpyHostToDevice);

   cufftExecZ2Z(plan,(cuDoubleComplex*)unewx_d,
               (cuDoubleComplex*)unewx_d,CUFFT_FORWARD);
   cufftExecZ2Z(plan,(cuDoubleComplex*)unewy_d,
               (cuDoubleComplex*)unewy_d,CUFFT_FORWARD);

   cudaMemcpy(unewx, unewx_d, complex_size, cudaMemcpyDeviceToHost);
   cudaMemcpy(unewy, unewy_d, complex_size, cudaMemcpyDeviceToHost);

   cudaFree(unewx_d); 
   cudaFree(unewy_d); 


   for (int i=0; i<(nx); i++){
     for (int j=0; j<(ny); j++){

         nk[0] = kx[i];
         nk[1] = ky[j];
         nk[2] = 0.0;

         int m = (j+i*ny);

         str_v0[m] = Complex(-1.0*(unewx[m].y*nk[0]),(unewx[m].x*nk[0]));
         str_v1[m] = Complex(-1.0*(unewy[m].y*nk[1]),(unewx[m].x*nk[1]));
         str_v2[m] = Complex(-1.0*(unewz[m].y*nk[2]),(unewx[m].x*nk[2]));
         str_v3[m] = Complex(-1.0*(unewy[m].y*nk[2] + unewz[m].y*nk[1]),
                                  (unewy[m].x*nk[2] + unewz[m].x*nk[1]));
         str_v4[m] = Complex(-1.0*(unewx[m].y*nk[2] + unewz[m].y*nk[0]),
                                  (unewx[m].x*nk[2] + unewz[m].x*nk[0]));
         str_v5[m] = Complex(-1.0*(unewx[m].y*nk[1] + unewy[m].y*nk[0]),
                                  (unewx[m].x*nk[1] + unewy[m].x*nk[0]));
     }  
   }

   cudaMalloc((void**)&str_v0_d,complex_size);
   cudaMalloc((void**)&str_v1_d,complex_size);
   cudaMalloc((void**)&str_v2_d,complex_size);
   cudaMalloc((void**)&str_v3_d,complex_size);
   cudaMalloc((void**)&str_v4_d,complex_size);
   cudaMalloc((void**)&str_v5_d,complex_size);
  
   cudaMemcpy(str_v0_d,str_v0,complex_size,cudaMemcpyHostToDevice);
   cudaMemcpy(str_v1_d,str_v1,complex_size,cudaMemcpyHostToDevice);
   cudaMemcpy(str_v2_d,str_v2,complex_size,cudaMemcpyHostToDevice);
   cudaMemcpy(str_v3_d,str_v3,complex_size,cudaMemcpyHostToDevice);
   cudaMemcpy(str_v4_d,str_v4,complex_size,cudaMemcpyHostToDevice);
   cudaMemcpy(str_v5_d,str_v5,complex_size,cudaMemcpyHostToDevice);

   cufftExecZ2Z(plan,(cuDoubleComplex*)str_v0_d,
               (cuDoubleComplex*)str_v0_d,CUFFT_FORWARD);
   cufftExecZ2Z(plan,(cuDoubleComplex*)str_v1_d,
               (cuDoubleComplex*)str_v1_d,CUFFT_FORWARD);
   cufftExecZ2Z(plan,(cuDoubleComplex*)str_v2_d,
               (cuDoubleComplex*)str_v2_d,CUFFT_FORWARD);
   cufftExecZ2Z(plan,(cuDoubleComplex*)str_v3_d,
               (cuDoubleComplex*)str_v3_d,CUFFT_FORWARD);
   cufftExecZ2Z(plan,(cuDoubleComplex*)str_v4_d,
               (cuDoubleComplex*)str_v4_d,CUFFT_FORWARD);
   cufftExecZ2Z(plan,(cuDoubleComplex*)str_v5_d,
               (cuDoubleComplex*)str_v5_d,CUFFT_FORWARD);

   cudaMemcpy(str_v0,str_v0_d,complex_size,cudaMemcpyDeviceToHost);
   cudaMemcpy(str_v1,str_v1_d,complex_size,cudaMemcpyDeviceToHost);
   cudaMemcpy(str_v2,str_v2_d,complex_size,cudaMemcpyDeviceToHost);
   cudaMemcpy(str_v3,str_v3_d,complex_size,cudaMemcpyDeviceToHost);
   cudaMemcpy(str_v4,str_v4_d,complex_size,cudaMemcpyDeviceToHost);
   cudaMemcpy(str_v5,str_v5_d,complex_size,cudaMemcpyDeviceToHost);

   cudaFree(str_v0_d);
   cudaFree(str_v1_d);
   cudaFree(str_v2_d);
   cudaFree(str_v3_d);
   cudaFree(str_v4_d);
   cudaFree(str_v5_d);

   for (int l=0; l<(nx*ny); l++){
 
     str_v0[l].x *= sizescale;  
     str_v1[l].x *= sizescale;  
     str_v2[l].x *= sizescale;
     str_v3[l].x *= sizescale; 
     str_v4[l].x *= sizescale;
     str_v5[l].x *= sizescale;

     str_v0[l].y *= sizescale;  
     str_v1[l].y *= sizescale;  
     str_v2[l].y *= sizescale;
     str_v3[l].y *= sizescale; 
     str_v4[l].y *= sizescale;
     str_v5[l].y *= sizescale;

   }


   /* Elastic driving force  */
   for (int l = 0; l < (nx*ny); l++){
    
     double hphi_p, e_temp;

     e_temp = Re(dfdphi[l]);
     hphi_p = (30.0*e_temp*e_temp*(1.0-e_temp)*(1.0-e_temp));           

     dfeldphi[l].x = -1.0*
               (Chom11*(hom_strain_v[0]+str_v0[l].x-eigenstrain_v[0][l])*
                epszero*hphi_p+
                Chom11*(hom_strain_v[1]+str_v1[l].x-eigenstrain_v[1][l])*
                epszero*hphi_p+
                Chom12*(hom_strain_v[1]+str_v1[l].x-eigenstrain_v[1][l])*
                epszero*hphi_p+
                Chom12*(hom_strain_v[0]+str_v0[l].x-eigenstrain_v[0][l])*
                epszero*hphi_p+
                Chom12*(hom_strain_v[2]+str_v2[l].x-eigenstrain_v[2][l])*
                epszero*hphi_p+ 
                Chom12*(hom_strain_v[2]+str_v2[l].x-eigenstrain_v[2][l])*
                epszero*hphi_p
               );

     dfeldphi[l].y = 0.0;
   }

   free (str_v0);
   free (str_v1);
   free (str_v2);
   free (str_v3);
   free (str_v4);
   free (str_v5);
   free (ts0);
   free (ts1);
   free (ts2);
   free (ts3);
   free (ts4);
   free (ts5);
   for (int ii=0; ii<6; ii++)
   free (eigenstrain_v[ii]);


}
