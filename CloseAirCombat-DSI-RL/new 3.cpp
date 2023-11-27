void sdof_calculate_data(void)
/*--------------------------*/
{

	/*--------------
	 local variables
	 -------------*/


	static double 		e0 = 1., e1 = 0., e2 = 0., e3 = 0.;
	double		ep, e0dot, e1dot, e2dot, e3dot;

	double		trust; /* motor trust before after burner */
	double	  	trust_aft_burn; /* trust after burner */
	double	  	trust_total; /* total trust */

	double 		air_density;
	double		dyn_pres;

	double		lift_coef;
	double		drag_coef;
	double		yaw_coef;
	double		pit_moment_coef;
	double		yaw_moment_coef;
	double		rol_moment_coef;

	double		pit_acc;
	double		roll_acc;
	double		head_acc;

	/*double	  	pi_half		;*/

	double		temp;
#ifdef _OTW
	short int chng_fl;
	float otw_req_alt;
	short int dummy_fl;

#endif
	/*-----------
	 local common
	 ----------*/

#define SB_MULT 1

	float a, b, xcl, cdsb;

	static float sb_wh = 0, sb_brk = 0, sb = 0;

	/*--------
	 execution
 --------*/

 /*-----------------
	 temporary common in
 -----------------*/

 /*pi_half = RCnst_Pi/2	;*/

	if (sdof_init_fl == 1)
	{
		sdof_init_fl = 0;
		e0 = cos(init_heading / 2);
		e1 = 0.;
		e2 = 0.;
		e3 = sin(init_heading / 2);
		sdof_dt = RCnst_Cycle_Time / SDOF_INTEG_NO; /*0.0125;   0.05 */

	}


	/*------------------------------------------------------------
	calculation of trust  = f(RAcs_Throt_Pos_position & after burner)
	limit trust for vel = 300 m/s
	   trust =          (50000./(1. + DSdof_Alt/6000.) +
		   30. * DSdof_Vel_Forw) 	* RSdof_Throttle  ;
			   50000 = engine power for vel = 0;
			   30( Newton)   = step of increasing in engine power
							   for each 1 m/s additional vel.

	NF5_A AFT
		  Change in trust calculation:
			No influence of Vel_Forw (Newton cnst = 0 instead of 30
			and 60)
		  Change in trust calculation: 50000 -> 70000
  ------------------------------------------------------------- */

	if (DSdof_Alt - shm_flt->f_gnd_level * FEET_TO_METER > 50)
	{
		if (DSdof_Vel_Forw < 400)                /* m/s */
		{
			trust = (60000. / (1. + DSdof_Alt / 15000.) +
				0. * DSdof_Vel_Forw) * RSdof_Throttle;

			trust_aft_burn = (100000. / (1. + DSdof_Alt / 15000.) +
				0. * DSdof_Vel_Forw) *
				(0.75 + 0.25 * RSdof_Throttle);

		}
		else
		{
			trust = (60000. / (1. + DSdof_Alt / 15000.) +
				0. * 300) * RSdof_Throttle;

			trust_aft_burn = (100000. / (1. + DSdof_Alt / 15000.) +
				0. * 300) *
				(0.75 + 0.25 * RSdof_Throttle);
		}
	}
	else
	{

		if (DSdof_Vel_Forw < 400)                /* m/s */
		{
			trust = (60000. / (1. + DSdof_Alt / 15000.) +
				30. * DSdof_Vel_Forw) * RSdof_Throttle;

			trust_aft_burn = (100000. / (1. + DSdof_Alt / 15000.) +
				60. * DSdof_Vel_Forw) *
				(0.75 + 0.25 * RSdof_Throttle);

		}
		else
		{
			trust = (60000. / (1. + DSdof_Alt / 15000.) +
				30. * 300) * RSdof_Throttle;

			trust_aft_burn = (100000. / (1. + DSdof_Alt / 15000.) +
				60. * 300) *
				(0.75 + 0.25 * RSdof_Throttle);
		}

	}

	trust_total = trust * (1 - RSdof_After_Burn)
		+ trust_aft_burn * RSdof_After_Burn;

	// Convert the full dry thrust of KFIR to the thrust of L159, L159 does not have After Burner. יורם אמר
//	trust_total = trust * (6330.0/11890.0) * shm_mmi->sdof_param[SDOF_TRUST_FACTOR] / 100.;	

//	trust_total   = trust; // no aft_burn in L159

	trust_total = trust_total * shm_mmi->sdof_param[SDOF_TRUST_FACTOR] / 100.;
	shm_mmi->sdof_param[SDOF_TRUST] = trust_total;

#ifdef	DEBUG
	printf("TRUST %f %f %f \n", trust, trust_aft_burn, trust_total);
#endif

	/*----------------
	  dynamic pressure
		  ---------------*/

	air_density = 1.225 / (1. + 9.62 * pow(10., -5) * DSdof_Alt +
		1.49 * pow(10., -8) * (pow(DSdof_Alt, 2)));

	dyn_pres = 0.5 * air_density * pow(DSdof_Vel_Forw, 2);

#ifdef	DEBUG
	printf("air_dens,dyn_pres  %f %f \n", air_density, dyn_pres);
#endif

	/*---------------------
	rotation matrix
	---------------------*/

	DSdof_Body_In[1][1] = pow(e0, 2) + pow(e1, 2) - pow(e2, 2) - pow(e3, 2);
	DSdof_Body_In[1][2] = 2. * (e1 * e2 - e0 * e3);

	DSdof_Body_In[1][3] = 2. * (e1 * e3 + e0 * e2);

	DSdof_Body_In[2][1] = 2. * (e1 * e2 + e0 * e3);
	DSdof_Body_In[2][2] = pow(e0, 2) - pow(e1, 2) + pow(e2, 2) - pow(e3, 2);
	DSdof_Body_In[2][3] = 2. * (e2 * e3 - e0 * e1);

	DSdof_Body_In[3][1] = 2. * (e1 * e3 - e0 * e2);

	DSdof_Body_In[3][2] = 2. * (e2 * e3 + e0 * e1);
	DSdof_Body_In[3][3] = pow(e0, 2) - pow(e1, 2) - pow(e2, 2) + pow(e3, 2);


#ifdef	DEBUG
	/*	printf ("E %f %f %f %f %f \n" ,e0,e1,e2,e3,i_cntr_cycle_no ) ; */
	printf("E %f %f %f %f    \n", e0, e1, e2, e3);
#endif

#ifdef	DEBUG
	printf("MATRIX %f %f %f \n", DSdof_Body_In[1][1], DSdof_Body_In[1][2], DSdof_Body_In[1][3]);
	printf("MATRIX %f %f %f \n", DSdof_Body_In[2][1], DSdof_Body_In[2][2], DSdof_Body_In[2][3]);
	printf("MATRIX %f %f %f \n", DSdof_Body_In[3][1], DSdof_Body_In[3][2], DSdof_Body_In[3][3]);
#endif

	/*------------------------------
	euler angles DSdof_Pitch = teta
				 DSdof_Head = psi
			 DSdof_Roll = phi
	-----------------------------*/

	DSdof_Pitch = -atan2(DSdof_Body_In[3][1],
		sqrt(1 - pow(DSdof_Body_In[3][1], 2)));

	DSdof_Head = atan2(DSdof_Body_In[2][1], DSdof_Body_In[1][1]);

	DSdof_Roll = atan2(DSdof_Body_In[3][2], DSdof_Body_In[3][3]);

#ifdef	DEBUG
	printf("ANGLES %f %f %f \n", DSdof_Pitch, DSdof_Head, DSdof_Roll);
#endif

	/*-------------------
	aerodynamic equations
	-------------------*/

	DSdof_Aoa = atan2(DSdof_Vel_Down, DSdof_Vel_Forw);

	/*-------------
	side slip angle
	-------------*/

	DSdof_Side_Slip = atan2(DSdof_Vel_Rght, DSdof_Vel_Forw);

#ifdef	DEBUG
	printf("AOA,SIDE_SLIP %f %f \n", DSdof_Aoa, DSdof_Side_Slip);
#endif

	/* increase lift for vel < 220 knots

   xcl
   2   \
		\
		 \
		  \
   1       ------------

	  -------------------->
	 140   220            vel (knots)
 ************************************/

 /*	if(sdof_type == T38_TYPE)
 */
	a = (float)(-1. / 80.);
	/*	else
			a = 0;
	*/

	b = (float)(1 - 200 * a);
	xcl = (float)(a * DSdof_Vel_Forw * METER_TO_KNOTS + b);

	if (xcl < 1)
		xcl = 1;

	if (xcl > 2.5)
		xcl = 2.5;

	/* ---------------------------------------------------------
	  decr AOA => incr lift =>new coef = 5. => old = 6.5 ! 3.5
	------------------------------------------------------------ */

	/*      if(RSpare_Spare[10] == 0)RSpare_Spare[10] = 1.;
		printf("\n********** RSpare_Spare[10]= %f",RSpare_Spare[10]);

		if(sdof_type == T38_TYPE)
			lift_coef = 5. * DSdof_Aoa * xcl + RSpare_Spare[10] * RSdof_Delta_Elev;
		else
			lift_coef = 3.5 * DSdof_Aoa * xcl + RSpare_Spare[10] * RSdof_Delta_Elev;

	*/

	/*------------------------------------------------------------
		   NF5-A  AFT Increase AOA for NF5-A -only for small velocities
		  ---------------------------------------------------------*/

	if (DSdof_Vel_Forw < 250)                /* m/s */
		lift_coef = 2.5 * DSdof_Aoa * xcl + RSdof_Delta_Elev;
	else
		lift_coef = 3.5 * DSdof_Aoa * xcl + RSdof_Delta_Elev;


	/* handle air (speed) breaks & wheels for drag coef
	**************************************************/

	if (shm_acs->s_inp_dig[AIR_BREAK] == BRKS_OUT)
	{
		sb_brk = sb_brk + sdof_dt * SB_MULT;   // after 2 sec breaks out
  //	  printf("\n sb_brk = %f sb = %f",sb_brk, sb);
	}
	//       sb = sb + sdof_dt * 1/5 ;   // after 5 sec breaks out

	if (shm_acs->s_inp_dig[WHEELS] == WH_DN)
	{
		sb_wh = sb_wh + sdof_dt * SB_MULT;   /* after 2 sec breaks out */
//		  printf("\n sb_wh = %f sb = %f",sb_wh, sb);
	}

	if (shm_acs->s_inp_dig[AIR_BREAK] == BRKS_IN)
	{
		sb_brk = sb_brk - sdof_dt * SB_MULT;   // after 2 sec breaks in 
//		  printf("\n sb_brk = %f sb = %f",sb_brk, sb);
	}
	if (shm_acs->s_inp_dig[WHEELS] == WH_UP)
	{
		sb_wh = sb_wh - sdof_dt * SB_MULT;   /* after 2 sec breaks in */
  //	  printf("\n sb_wh = %f sb = %f",sb_wh, sb);
	}


	if (sb_brk < 0)
		sb_brk = 0;

	if (sb_brk > 2)
		sb_brk = 2;

	if (sb_wh < 0)
		sb_wh = 0;

	if (sb_wh > 2)
		sb_wh = 2;

	sb = sb_brk + sb_wh;

	/* increase influence of sb by decreasing constant 0.064  */

/*	cdsb = (float)(0.064 * ( 1 - DSdof_Aoa/ 0.3) * sb) ;

	if(cdsb < 0)
		cdsb = 0;
	if(cdsb > 0.064)
		cdsb = 0.064f;*/

	cdsb = (float)(FAcs_Drag_Factor * (1 - DSdof_Aoa / 0.3) * sb);

	if (cdsb < 0)
		cdsb = 0;
	if (cdsb > FAcs_Drag_Factor)
		cdsb = FAcs_Drag_Factor;


	/* --------------------------------------------------------
	decrease effect of lift coef in drag coef in order to
	avoid loss of speed during high g manoeuver.
		( coef = 0.2 (old 0.35)

	drag_coef = cdsb + (0.025 + 0.2  * (pow(lift_coef,2) )) * 1.25	;
					1.25 = drag coef.It has to be changed when aircraft
			   drag is required to be changed.
		NF5_A AFT     ********** Change in drag coeficient
				   increase effect of lift coef frm 0.2 to 0.6
	--------------------------------------------------------*/

	/*	drag_coef = cdsb + (0.025 + 0.2  * (pow(lift_coef,2) )) * 1.25	;*/

	drag_coef = cdsb + (0.025 + 0.6 * (pow(lift_coef, 2))) * 1.25;


	/* -------------------------------------------------------
   decrease side slip => coef = 1.4 (old = 0.7) together
	   in yaw_coef & yaw_moment_coef ( == incr. area of rudder )
	   0.26 (old = 0.13)
----------------------------------------------------------*/

	yaw_coef = -1.4 * DSdof_Side_Slip;


	/* -----------------------------------------------------------------
		   - decrease a/c fluctuations during flight => coef = 6.4(old = 3.2)
		   - decrease pitch down in roll man. by incr . lift_coef influence
				coef = 0.05 (old = 0.03)
		   - incr. elevator moment during low speed = > 0.28 * xcl(old 0.28)

	NF5A - AFT
		  - decrease a/c fluctuations during flight => Pitch_Rate coef = 10.
																(old = 6.4)
	------------------------------------------------------------------- */

	pit_moment_coef =
		0.005 - 0.05 * lift_coef - 0.28 * xcl *
		(RSdof_Delta_Elev + 0.005 + DSdof_Vel_Forw / 800. * 0.025)
		+ lift_coef * (shm_mmi->sdof_param[SDOF_GRAV_CENTER]/*AC_GRAV_CENTER*/ - 0.49)
		- 10 * DSdof_Pitch_Rate / DSdof_Vel_Forw;


	/* decrease side slip => coef = 1.4 (old = 0.7) together
	  in yaw_coef & yaw_moment_coef ( == incr. area of rudder )
	  0.26 (old = 0.13)
---------------------------------------------------------*/

	yaw_moment_coef = 0.26 * DSdof_Side_Slip - 0.08 * RSdof_Delta_Ailr +
		yaw_coef * (shm_mmi->sdof_param[SDOF_GRAV_CENTER]/*AC_GRAV_CENTER*/ - 0.49) -
		2.8 * DSdof_Head_Rate / DSdof_Vel_Forw;

	/* --------------------------------------------------
	   if roll manuoevering is required to be changed
	   then roll dumper DSdof_Roll_Rate/DSdof_Vel_Forw should
	   be multiplied by a correspondent coef.
------------------------------------------------------ */


	rol_moment_coef = -0.2 * RSdof_Delta_Ailr -
		DSdof_Roll_Rate / DSdof_Vel_Forw;

#ifdef	DEBUG
	printf("COEF %f %f %f \n", lift_coef, drag_coef, yaw_coef);
	printf("MOMENT %f %f %f \n", pit_moment_coef, yaw_moment_coef, rol_moment_coef);
#endif

	/*------------------------------------
	translation and rotation accelerations
	DSdof_Acc_Forw = udot
	DSdof_Acc_Rght = vdot
	DSdof_Acc_Down = wdot
	roll_acc = pdot
	pit_acc = qdot
	rol_acc = rdot
	------------------------------------*/
	DSdof_Acc_Forw = (dyn_pres * shm_mmi->sdof_param[SDOF_WING_SURFACE]/*AC_WING_SURFACE*/ *
		(lift_coef * DSdof_Aoa - drag_coef) + trust_total) / shm_mmi->sdof_param[SDOF_AC_WEIGHT]/*AC_WEIGHT*/
		+ AC_GRAV_ACC * DSdof_Body_In[3][1];


	DSdof_Acc_Rght = dyn_pres * shm_mmi->sdof_param[SDOF_WING_SURFACE]/*AC_WING_SURFACE*/ *
		(yaw_coef - drag_coef * DSdof_Side_Slip) / shm_mmi->sdof_param[SDOF_AC_WEIGHT]/*AC_WEIGHT*/
		+ AC_GRAV_ACC * DSdof_Body_In[3][2]
		- DSdof_Head_Rate * DSdof_Vel_Forw;


	DSdof_Acc_Down = dyn_pres * shm_mmi->sdof_param[SDOF_WING_SURFACE]/*AC_WING_SURFACE*/ *
		(-lift_coef - drag_coef * DSdof_Aoa) / shm_mmi->sdof_param[SDOF_AC_WEIGHT]/*AC_WEIGHT*/
		+ AC_GRAV_ACC * DSdof_Body_In[3][3]
		+ DSdof_Pitch_Rate * DSdof_Vel_Forw;

	/////////////////////////////////////////////
/*	static float alpha =0;
	static float kk = 0.8f;
	static float old_pedal;

	if(shm_acs->f_inp_anlg[PEDAL] > 100)
	{
		if( 0.f < old_pedal && old_pedal < 100.f)
			alpha =  0.75f;
		else
			alpha = alpha + 0.0021816f;
	}

	else if(shm_acs->f_inp_anlg[PEDAL] < -100)
	{
		if(   -100 < old_pedal  && old_pedal < 0)

				alpha =  0.75f;
		else
				alpha = alpha -0.0021816f;// alpha   -0.0021816f/2.f ;
	}

	else
		alpha = 0.f;


	if(alpha != 0)
		printf("\n alpha = %f pedals = %f", alpha, shm_acs->f_inp_anlg[PEDAL]);

	DSdof_Acc_Forw = DSdof_Acc_Forw * cos(alpha * kk) + DSdof_Acc_Rght * sin(alpha * kk);
	DSdof_Acc_Rght = -DSdof_Acc_Forw * sin(alpha * kk) + DSdof_Acc_Rght * cos(alpha * kk);

	old_pedal = shm_acs->f_inp_anlg[PEDAL];*/
	///////////////////////

#ifdef	DEBUG
	printf("A/C ACC %f %f %f \n", DSdof_Acc_Forw, DSdof_Acc_Rght, DSdof_Acc_Down);
#endif

	roll_acc = dyn_pres * shm_mmi->sdof_param[SDOF_WING_CHORD]/*AC_WING_CHORD*/ * shm_mmi->sdof_param[SDOF_WING_SURFACE]/*AC_WING_SURFACE*/ *
		rol_moment_coef / shm_mmi->sdof_param[SDOF_INERTIAL_MOM_X]/*AC_INERTIAL_MOM_X*/;

	pit_acc = dyn_pres * shm_mmi->sdof_param[SDOF_WING_CHORD]/*AC_WING_CHORD*/ * shm_mmi->sdof_param[SDOF_WING_SURFACE]/*AC_WING_SURFACE*/ *
		pit_moment_coef / shm_mmi->sdof_param[SDOF_INERTIAL_MOM_Y]/*AC_INERTIAL_MOM_Y*/;

	head_acc = dyn_pres * shm_mmi->sdof_param[SDOF_WING_CHORD]/*AC_WING_CHORD*/ * shm_mmi->sdof_param[SDOF_WING_SURFACE]/*AC_WING_SURFACE*/ *
		yaw_moment_coef / shm_mmi->sdof_param[SDOF_INERTIAL_MOM_Z]/*AC_INERTIAL_MOM_Z*/ +
		(shm_mmi->sdof_param[SDOF_INERTIAL_MOM_X]/*AC_INERTIAL_MOM_X*/ - shm_mmi->sdof_param[SDOF_INERTIAL_MOM_Y]/*AC_INERTIAL_MOM_Y*/) *
		DSdof_Roll_Rate * DSdof_Pitch_Rate / shm_mmi->sdof_param[SDOF_INERTIAL_MOM_Z]/*AC_INERTIAL_MOM_Z*/;


#ifdef	DEBUG
	printf("ANG ACC %f %f \n", roll_acc, pit_acc, head_acc);
#endif

	/*----------------------
	derivatives of equations
	----------------------*/

	e0dot = 0.5 * (-e1 * DSdof_Roll_Rate
		- e2 * DSdof_Pitch_Rate
		- e3 * DSdof_Head_Rate);

	e1dot = 0.5 * (e0 * DSdof_Roll_Rate
		- e3 * DSdof_Pitch_Rate
		+ e2 * DSdof_Head_Rate);

	e2dot = 0.5 * (+e3 * DSdof_Roll_Rate
		+ e0 * DSdof_Pitch_Rate
		- e1 * DSdof_Head_Rate);

	e3dot = 0.5 * (-e2 * DSdof_Roll_Rate
		+ e1 * DSdof_Pitch_Rate
		+ e0 * DSdof_Head_Rate);

#ifdef	DEBUG
	printf("EDOT %f %f %f %f \n", e0dot, e1dot, e2dot, e3dot);
#endif

	/*----------------------
	translational velocities
	----------------------*/

	DSdof_Vel_N = DSdof_Body_In[1][1] * DSdof_Vel_Forw +
		DSdof_Body_In[1][2] * DSdof_Vel_Rght +
		DSdof_Body_In[1][3] * DSdof_Vel_Down;

	DSdof_Vel_E = DSdof_Body_In[2][1] * DSdof_Vel_Forw +
		DSdof_Body_In[2][2] * DSdof_Vel_Rght +
		DSdof_Body_In[2][3] * DSdof_Vel_Down;

	DSdof_Vel_Dwn = DSdof_Body_In[3][1] * DSdof_Vel_Forw +
		DSdof_Body_In[3][2] * DSdof_Vel_Rght +
		DSdof_Body_In[3][3] * DSdof_Vel_Down;

#ifdef	DEBUG
	printf(" INERTIAL VEL %f %f %f \n", DSdof_Vel_N, DSdof_Vel_E, DSdof_Vel_Dwn);
#endif

	/*----------
	integrations
	----------*/

	if (shm_atm->f_mach_no > 1.5)
	{
		temp = DSdof_Vel_Forw +
			(1.5 * DSdof_Acc_Forw - 0.5 * acc_forw_prv) * sdof_dt;

		if (temp < DSdof_Vel_Forw)
		{
			DSdof_Vel_Forw = temp;
		}
	}
	else
		DSdof_Vel_Forw = DSdof_Vel_Forw +
		(1.5 * DSdof_Acc_Forw - 0.5 * acc_forw_prv) * sdof_dt;

	acc_forw_prv = DSdof_Acc_Forw;

	if (DSdof_Vel_Forw < 10.0) DSdof_Vel_Forw = 10.0;

	DSdof_Vel_Rght = DSdof_Vel_Rght +
		(1.5 * DSdof_Acc_Rght - 0.5 * acc_rght_prv) * sdof_dt;
	acc_rght_prv = DSdof_Acc_Rght;

	DSdof_Vel_Down = DSdof_Vel_Down +
		(1.5 * DSdof_Acc_Down - 0.5 * acc_down_prv) * sdof_dt;
	acc_down_prv = DSdof_Acc_Down;

	DSdof_Roll_Rate = DSdof_Roll_Rate +
		(1.5 * roll_acc - 0.5 * roll_acc_prv) * sdof_dt;
	roll_acc_prv = roll_acc;

	DSdof_Pitch_Rate = DSdof_Pitch_Rate +
		(1.5 * pit_acc - 0.5 * pit_acc_prv) * sdof_dt;
	pit_acc_prv = pit_acc;

	DSdof_Head_Rate = DSdof_Head_Rate +
		(1.5 * head_acc - 0.5 * head_acc_prv) * sdof_dt;
	head_acc_prv = head_acc;

	/*--------------------
	components of rotation
	matrix(EULER ANGLES)
	--------------------*/

	e0 = e0 + (1.5 * e0dot - 0.5 * e0dotp) * sdof_dt;
	e0dotp = e0dot;
	e1 = e1 + (1.5 * e1dot - 0.5 * e1dotp) * sdof_dt;
	e1dotp = e1dot;
	e2 = e2 + (1.5 * e2dot - 0.5 * e2dotp) * sdof_dt;
	e2dotp = e2dot;
	e3 = e3 + (1.5 * e3dot - 0.5 * e3dotp) * sdof_dt;
	e3dotp = e3dot;

	ep = pow(e0, 2) + pow(e1, 2) + pow(e2, 2) + pow(e3, 2) - 1.;

	ep = 1. - 0.5 * ep;

	e0 = e0 * ep;
	e1 = e1 * ep;
	e2 = e2 * ep;
	e3 = e3 * ep;

#ifdef	DEBUG
	printf("E %f %f %f %f \n", e0, e1, e2, e3);
#endif

	DSdof_Alt = DSdof_Alt - DSdof_Vel_Dwn * sdof_dt;

	if (DSdof_Alt < -500.)	DSdof_Alt = -500.0;

#ifdef	DEBUG
	printf("ALT %f \n", DSdof_Alt);
#endif
#ifdef _OTW
	if (DSdof_Vel_Dwn > 0)
	{
		otw_req_chng_ac_alt(&chng_fl, &otw_req_alt);

		if (chng_fl == 1)
		{
			//--------------------------------------
			//  Check if this is the entry after reset
			//  & DSdof_Alt gets already a higher value
			//  Note:
			//  otw_req_alt = terrain_altitude + 40 ft.
			//-----------------------------------------
			if (DSdof_Alt > otw_req_alt + 500)
				dummy_fl = 0;                       // dummy operation 
			else
			{
				DSdof_Alt = otw_req_alt;   // meter 

			   //--------------------------------------
			   //   When take off, reset acelerations 
			   //   in order to shorten the delay 
			   //--------------------------------------

			/////////   if(DSdof_Vel_Dwn > 30 && 
			//////////  shm_acs->s_inp_dig[LANDING_GEAR] ==WHEELS_UP )
			/////////    		sdof_reset();
			}

		}
	}
#endif
	/*-----------------------------------
	calculation of latitude and longitude
	-----------------------------------*/
	if (shm_wnd->e_wnd_exist == WIND_EXIST_E)
	{
		if (REarth_Radius_N != 0)

			DSdof_Lat = DSdof_Lat +
			((DSdof_Vel_N + shm_wnd->f_vel_n * FEET_TO_METER) * sdof_dt
				/ (REarth_Radius_N * FEET_TO_METER + DSdof_Alt)
				) * RAD_TO_DEG;


		if (REarth_Radius_E != 0)

			DSdof_Lon = DSdof_Lon + (
				(DSdof_Vel_E + shm_wnd->f_vel_e * FEET_TO_METER) * sdof_dt
				/ ((REarth_Radius_E * FEET_TO_METER + DSdof_Alt)
					* cos(DSdof_Lat * DEG_TO_RAD))) * RAD_TO_DEG;


	}
	else
	{

		if (REarth_Radius_N != 0)

			DSdof_Lat = DSdof_Lat + (DSdof_Vel_N * sdof_dt
				/ (REarth_Radius_N * FEET_TO_METER + DSdof_Alt)
				) * RAD_TO_DEG;


		if (REarth_Radius_E != 0)

			DSdof_Lon = DSdof_Lon + (DSdof_Vel_E * sdof_dt
				/ ((REarth_Radius_E * FEET_TO_METER + DSdof_Alt)
					* cos(DSdof_Lat * DEG_TO_RAD))) * RAD_TO_DEG;

	}



}
/**************************************************************************/