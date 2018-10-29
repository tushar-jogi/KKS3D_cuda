void Output_Conf (int steps)
{
 FILE    *fpt, *fpt2;
 //int     gridcount=0;
 char    fn[100],fn1[100];
 double  min, max;
 //double  delta_cm, delta_cp;
 //double  k_bar,rad;

 sprintf (fn, "conf.%07d", steps);

 fpt = fopen (fn, "w");
 fwrite (comp, sizeof(cuDoubleComplex), nx*ny*nz, fpt);
 fwrite (dfdphi, sizeof(cuDoubleComplex), nx*ny*nz, fpt);
 fclose (fpt);

 min = 1.0;
 for (int i = 0; i < nx; i++) {
   for (int j = 0; j < ny; j++) {
     for (int k = 0; k < nz; k++) {
       if(Re(comp[k+nz*(j+i*ny)]) < min) 
         min = Re(comp[k+nz*(j+i*ny)]);
     }
   }
 }
 printf("Minimum in composition =%lf\n", min);
 max = 0.0;
 for (int i = 0; i < nx; i++) {
   for (int j = 0; j < ny; j++) {
     for (int k = 0; k < nz; k++) {
       if (Re(comp[k+nz*(j+i*ny)]) > max) 
       max = Re(comp[k+nz*(j+i*ny)]);
     }
   }
 }
 printf("Maximum in composition =%lf\n", max);

 for (int i = 0; i < nx; i++){
   for (int j = 0; j < ny; j++){  
     for (int k = 0; k < nz; k++) {
       if (Re(dfdphi[k+nz*(j + i*ny)]) >= 0.5)
       {
        gridcount++;   
       }
     }  
   }      
 }
 
 rad =  pow((3.0*gridcount)/(4.0*3.142857143),1.0/3.0);
 k_bar = 1.0/rad;
 
 //Finding c_alpha and c_beta at phi=0.5
 
 for (int i=nx/2;i<nx;i++){
     
     if (Re(dfdphi[ny/2 + i*ny]) <=0.5){
        x0 = i; 
     }
     if (Re(dfdphi[ny/2 + i*ny]) > 0.5){
         x1 = i;
         break;
     }
 }

 for (int i = nx/2; i < nx; i++){
     
     if ((Re(dfdphi[nz/2+nz*(ny/2+i*ny)]) < 0.05)){
        delta_cm = comp[nz/2+nz*(ny/2+i*ny)].x ;
        break;
     }   
 }

 delta_cp = comp[nz*(ny/2+(nx/2)*ny)].x - 1.0; 

 x2 = x0 + 
   ((x1-x0)/(Re(dfdphi[ny/2+(int)x1*ny]) - Re(dfdphi[ny/2+(int)x0*ny])))*
   (0.5-Re(dfdphi[ny/2+(int)x0*ny]));

 c_x2 = Re(comp[ny/2 + (int)x0*ny]) + 
       ((Re(comp[ny/2+(int)x1*ny]) - Re(comp[ny/2+(int)x0*ny]))/(x1-x0))*
       (x2 - x0);

 if (steps >= time_elast){  
 fpt2 = fopen("deltaC_vs_R.txt","a+");
 fprintf(fpt2,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",rad, k_bar, delta_cm, delta_cp, 
              min, max-1.0);
 fclose(fpt2);
 }

 sprintf (fn, "prof_gp.%07d", steps);
 sprintf (fn1,"1Dprof.%07d",steps);

 fpt = fopen (fn, "w");
 fpt2 = fopen (fn1, "w");

 for (int i = 0; i < nx; i++) {
  for (int j = 0; j < ny; j++) {
    for (int k = 0; k < nz; k++) {

      int l = k + nz*(j+i*ny);
      fprintf(fpt,"%d\t%d\t%d\t%lf\t%lf\n",i,j,k,comp[l].x,dfdphi[l].x);
    }
  }
 }
 fclose(fpt); 
   for (int k = 0; k < nz; k++) {

    int l = (nz/2) + nz*(k+(nx/2)*ny);
    fprintf(fpt2,"%d\t%lf\t%lf\n",k,comp[l].x,dfdphi[l].x);
   }
 fclose(fpt2); 
}
