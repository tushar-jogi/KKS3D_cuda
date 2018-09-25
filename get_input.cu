void Get_Input_Parameters (char *fnin, char *fnout)
{
 FILE *fpin, *fpcout;
 char param[100], fn[100];

 if (!(fpcout = fopen (fnout, "w"))) {
  printf ("File:%s could not be opened \n", fnout);
  exit (1);
 }

 fprintf (fpcout, "The name of this file is : %s \n", fnout);
 fprintf (fpcout, "Input is from            : %s \n", fnin);

 if (!(fpin = fopen (fnin, "r"))) {
  printf ("File: %s could not be opened \n", fnin);
  exit (1);
 }

 if(fscanf (fpin, "%s%d", param,&nx));
 if(fscanf (fpin, "%s%d", param,&ny));
 if(fscanf (fpin, "%s%d", param,&nz));
 if(fscanf (fpin, "%s%le",param,&dx));
 if(fscanf (fpin, "%s%le",param,&dy));
 if(fscanf (fpin, "%s%le",param,&dz));
 if(fscanf (fpin, "%s%le",param,&Ln));
 if(fscanf (fpin, "%s%le",param,&interface_energy));
 if(fscanf (fpin, "%s%le",param,&diffusivity));
 if(fscanf (fpin, "%s%le",param,&relax_coeff));
 if(fscanf (fpin, "%s%le",param,&dt));
 if(fscanf (fpin, "%s%le",param,&T));
 if(fscanf (fpin, "%s%d", param,&num_steps));
 if(fscanf (fpin, "%s%le%s%le%s%le%s%lf", param, &f0A, param, &f0B, param, 
             &Vm, param, &alpha));
 if(fscanf (fpin, "%s%le%s%le" , param , &c_alpha_eq , param , &c_beta_eq));
 if(fscanf (fpin, "%s%lf",param,&c0));
 if(fscanf (fpin, "%s%lf%s%lf",param,&ppt_size, param, &vf));
 if(fscanf (fpin, "%s%le%s%le%s%le", param, &mu_m, param, &nu_m, param,
             &Az_m));
 if(fscanf (fpin, "%s%le%s%le%s%le", param, &mu_p, param, &nu_p, param,
             &Az_p));
 if(fscanf (fpin, "%s%le", param, &epszero ));
 if(fscanf (fpin, "%s%le%le%le%le%le%le",param,&sigappl_v[0],&sigappl_v[1],
            &sigappl_v[2],&sigappl_v[3],&sigappl_v[4],&sigappl_v[5]));
 if(fscanf (fpin, "%s%d", param,&create_nuclei));
 if(fscanf (fpin, "%s%d", param,&t_prof1));
 if(fscanf (fpin, "%s%d", param,&t_prof2));
 if(fscanf (fpin, "%s%d", param,&numsteps_prof1));
 if(fscanf (fpin, "%s%d", param,&numsteps_prof2));
 if(fscanf (fpin, "%s%d", param,&time_elast));
 if(fscanf (fpin, "%s%d", param,&elast_int));
 if(fscanf (fpin, "%s%d", param,&interpolation))
 if(fscanf (fpin, "%s%d", param,&initflag));
 if(fscanf (fpin, "%s%d", param,&initcount));
 fclose (fpin);
  
 printf ("==================================================================\n");
 printf ("*                       KKS 3D ELASTCITY                         *\n");
 printf ("*    Developed at CMS Lab, Department of MSME, IIT Hyderabad     *\n");
 printf ("==================================================================\n");

 printf ("nx=%d\n",nx);
 printf ("ny=%d\n",ny);
 printf ("nz=%d\n",nz);
 printf ("dx=%lf\n",dx);
 printf ("dy=%lf\n",dy);
 printf ("dz=%lf\n",dz);
 printf ("dt=%le\n",dt); 
 printf ("Simulation steps = %d\n", num_steps);
 printf ("Bulk free energy coefficients f0A=%le\t f0B=%le\nVm=%le\t dt=%le\n",
         f0A,f0B, Vm,dt);
 printf ("Interface_energy=%le \tInterface width=%le \n", interface_energy,
         Ln);
 printf ("SUPERSATURATION, c0 = %lf\n", c0);
 printf ("ppt_size = %lf\n", ppt_size);
 printf ("======ELASTICITY PARAMETERS=======\n");
 printf ("mu_m = %le, mu_m = %le, Az_m = %le\n", mu_m, nu_m, Az_m);
 printf ("mu_p = %le, nu_p = %le, Az_p = %le\n", mu_p, nu_p, Az_p);
 printf ("epszero = %le\n", epszero);
 printf ("Applied stress \n");
 printf ("%.3le %.3le %.3le %.3le %.3le %.3le\n", 
     sigappl_v[0], sigappl_v[1], sigappl_v[2],
     sigappl_v[3], sigappl_v[4], sigappl_v[5]);
 printf ("=====MICROSTRUCTURE INFORMATION=====\n");
 printf ("Initflag = %d\n", initflag);

 if(initflag == 1) {

    initcount = 0;
    if (create_nuclei == 0)
      printf ("Single precipitate simulation\n");
    else if (create_nuclei == 1)
      printf ("Random nuclei generated based on CNT\n");
    else if (create_nuclei == 2)
      printf ("Random nuclei generated with given volume fraction\n");

 } else {

    sprintf (fn,"conf.%06d", initcount);     
    printf ("Configuration read from file %s\n",fn);

 }
 
 
 fprintf (fpcout, "nx %d\n", nx);
 fprintf (fpcout, "ny %d\n", ny);
 fprintf (fpcout, "ny %d\n", nz);
 fprintf (fpcout, "dx %lf\n",dx);
 fprintf (fpcout, "dy %lf\n",dy);
 fprintf (fpcout, "dt %lf\n",dt);
 fprintf (fpcout, "num_steps %d\n", num_steps);
 fprintf (fpcout, "f0A   %3.2f  f0B %3.2f Vm   %3.2f   alpha   %3.2f\n", 
          f0A , f0B, Vm , alpha);
 fprintf (fpcout, "c0 %lf\n", c0);
 fprintf (fpcout, "Ppt_size %lf\n vf %lf", ppt_size, vf);
 fprintf (fpcout, "diffusivity %lf\t relax_coeff %lf\t \n", diffusivity,
          relax_coeff );
 fprintf (fpcout,"mu_m = %le, nu_m = %le, Az_m = %le\n", mu_m, nu_m, Az_m);
 fprintf (fpcout,"mu_p = %le, nu_p = %le, Az_p = %le\n", mu_p, nu_p, Az_p);
 fprintf (fpcout,"epszero = %le\n", epszero);
 fprintf (fpcout,"Applied stress \n");
 fprintf (fpcout,"%le %le %le %le %le %le\n", sigappl_v[0], sigappl_v[1], 
                       sigappl_v[2], sigappl_v[3], sigappl_v[4], sigappl_v[5]);
 fprintf (fpcout,"delta = %le", delta);
 fprintf (fpcout, "Calc_interface_energy  %d\n", calc_interface_energy);
 fprintf (fpcout, "initflag %d\n", initflag);
 fprintf (fpcout, "interpolation %d\n", interpolation);
 fprintf (fpcout, "initcount %d\n", initcount);
 fclose  (fpcout);
}
