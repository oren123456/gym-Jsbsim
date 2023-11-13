#include "pch.h"
#include "DSI_FlightEngine.h"

 METER_TO_KNOTS  (float)	(3.280839/1.6887)
 FAcs_Drag_Factor (0.064f)
 MAX_SDOF_PARAM			9
 SDOF_TRUST_FACTOR		0
 SDOF_TRUST				1
 SDOF_INERTIAL_MOM_X		2
 SDOF_INERTIAL_MOM_Y		3
 SDOF_INERTIAL_MOM_Z		4
 SDOF_AC_WEIGHT          5
 SDOF_WING_SURFACE  		6
 SDOF_WING_CHORD    		7
 SDOF_GRAV_CENTER   		8
	AC_GRAV_ACC         9.806     /* m/s^2     */
 PI             3.141592654f
 RAD_TO_DEG      (float)	(180./PI)
 DEG_TO_RAD      (float)	(PI/180.)
 FEET_TO_METER   (float) (1./3.280839)
 Earth_Radius  2.092646982E7	/* earth radius at the equator [feet]*/
 Earth_Radius_M (float)(Earth_Radius* FEET_TO_METER)
 PITCH_CONST (float)(0.10f)
 ROLL_CONST (float)(0.06f)

void DSI_FlightEngine::tick( float cycleTime, float throttle, float stickX, float stickY)
{
	m_throttle = throttle;
	m_ailron_com = stickX;
	m_elevator_com = stickY;
	sdof_calculate_data(cycleTime);

	double obs[12];
	getObs(&obs[0]);
}

void DSI_FlightEngine::getObs(double* obs)
{
	obs[0] = m_lat;		// deg
	obs[1] = m_lon;		// deg
	obs[3] = m_alt;		// meters
	obs[4] = m_pitch;	// rad
	obs[5] = m_roll;	// rad
	obs[6] = m_head;	// rad
	obs[7] = m_vel_N;	// m/s
	obs[8] = m_vel_E;	// m/s
	obs[9] = m_vel_dwn;	// m/s
	obs[10] = m_acc_N;	// ft/s2
	obs[11] = m_acc_E;	// ft/s2
	obs[12] = m_acc_Dwn;	// ft/s2
}


void DSI_FlightEngine::reset(double lat, double lon, double alt, float init_heading, float vel_forw_ms)
{
	m_lat = lat ;
	m_lon = lon;
	m_alt = alt;
	m_vel_forw = vel_forw_ms;

	/*----------------------------
	  Check if profile change was not
			  performed during landing
	------------------------------*/
	m_pitch = 0    /* rad*/;
	m_roll = 0     /* rad*/;
	m_head = init_heading      /* rad*/;

	m_vel_down = 0.00;
	m_vel_rght = 0.00;
	m_vel_N = 0.00;
	m_vel_E = 0.00;
	m_vel_dwn = 0.00;

	m_pitch_rate = 0.00;
	m_roll_rate = 0.00;
	m_head_rate = 0.00;

	m_acc_rght_prv = 0.;
	m_acc_forw_prv = 0.;
	m_acc_down_prv = 0.;
	m_pit_acc_prv = 0.;
	m_roll_acc_prv = 0.;
	m_head_acc_prv = 0.;

	m_e0dotp = 0.;
	m_e1dotp = 0.;
	m_e2dotp = 0.;
	m_e3dotp = 0.;

	m_e0 = cos(init_heading / 2);
	m_e1 = 0.;
	m_e2 = 0.;
	m_e3 = sin(init_heading / 2);
}

void DSI_FlightEngine::sdof_calculate_data(float dt)
{
	double	body_In[4][4];
	/// rotation matrix
	body_In[1][1] = pow(m_e0, 2) + pow(m_e1, 2) - pow(m_e2, 2) - pow(m_e3, 2);
	body_In[1][2] = 2. * (m_e1 * m_e2 - m_e0 * m_e3);
	body_In[1][3] = 2. * (m_e1 * m_e3 + m_e0 * m_e2);
	body_In[2][1] = 2. * (m_e1 * m_e2 + m_e0 * m_e3);
	body_In[2][2] = pow(m_e0, 2) - pow(m_e1, 2) + pow(m_e2, 2) - pow(m_e3, 2);
	body_In[2][3] = 2. * (m_e2 * m_e3 - m_e0 * m_e1);
	body_In[3][1] = 2. * (m_e1 * m_e3 - m_e0 * m_e2);
	body_In[3][2] = 2. * (m_e2 * m_e3 + m_e0 * m_e1);
	body_In[3][3] = pow(m_e0, 2) - pow(m_e1, 2) - pow(m_e2, 2) + pow(m_e3, 2);

	/// euler angles m_pitch = teta			 m_head = psi		 m_roll = phi
	m_pitch = -atan2(body_In[3][1],sqrt(1 - pow(body_In[3][1], 2)));
	m_head = atan2(body_In[2][1], body_In[1][1]);
	m_roll = atan2(body_In[3][2], body_In[3][3]);

	///dynamic pressure
	double air_density = 1.225 / (1. + 9.62 * pow(10., -5) * m_alt + 1.49 * pow(10., -8) * (pow(m_alt, 2)));
	double dyn_pres = 0.5 * air_density * pow(m_vel_forw, 2);

	double	trust; /* motor trust before after burner */
	if (m_vel_forw < 400)                /* m/s */
		trust = (60000. / (1. + m_alt / 15000.) + 0. * m_vel_forw) * m_throttle;
	else
		trust = (60000. / (1. + m_alt / 15000.) + 0. * 300) * m_throttle;

	float a = (float)(-1. / 80.);
	float b = (float)(1 - 200 * a);
	float xcl = (float)(a * m_vel_forw * METER_TO_KNOTS + b);
	if (xcl < 1)	xcl = 1;
	if (xcl > 2.5)	xcl = 2.5;
	/// ///aerodynamic equations
	double aoa = atan2(m_vel_down, m_vel_forw);
	double lift_coef;
	if (m_vel_forw < 250)                /* m/s */
		lift_coef = 2.5 * aoa * xcl + PITCH_CONST * m_elevator_com;
	else
		lift_coef = 3.5 * aoa * xcl + PITCH_CONST * m_elevator_com;

	double cdsb = (float)(FAcs_Drag_Factor * (1 - aoa / 0.3));
	if (cdsb < 0)	cdsb = 0;
	if (cdsb > FAcs_Drag_Factor)
		cdsb = FAcs_Drag_Factor;
	/// decrease effect of lift coef in drag coef in order to avoid loss of speed during high g manoeuver.	( coef = 0.2 (old 0.35)
	double drag_coef = cdsb + (0.025 + 0.6 * (pow(lift_coef, 2))) * 1.25;
	///side slip angle
	double side_slip = atan2(m_vel_rght, m_vel_forw);
	/// decrease side slip => coef = 1.4 (old = 0.7) together in yaw_coef & yaw_moment_coef ( == incr. area of rudder )  0.26 (old = 0.13)
	double yaw_coef = -1.4 * side_slip;
	/// decrease a/c fluctuations during flight => coef = 6.4(old = 3.2)  - decrease pitch down in roll man. by incr . lift_coef influence
	double pit_moment_coef = 0.005 - 0.05 * lift_coef - 0.28 * xcl * (PITCH_CONST * m_elevator_com + 0.005 + m_vel_forw / 800. * 0.025) + lift_coef * (SDOF_GRAV_CENTER - 0.49) - 10 * m_pitch_rate / m_vel_forw;
	/// decrease side slip => coef = 1.4 (old = 0.7) together in yaw_coef & yaw_moment_coef ( == incr. area of rudder )  0.26 (old = 0.13)
	double yaw_moment_coef = 0.26 * side_slip - 0.08 * ROLL_CONST * m_ailron_com + yaw_coef * (SDOF_GRAV_CENTER - 0.49) - 2.8 * m_head_rate / m_vel_forw;
	
	double acc_forw = (dyn_pres * SDOF_WING_SURFACE * (lift_coef * aoa - drag_coef) + trust) / SDOF_AC_WEIGHT	+ AC_GRAV_ACC * body_In[3][1];
	double acc_rght = dyn_pres * SDOF_WING_SURFACE *	(yaw_coef - drag_coef * side_slip) /SDOF_AC_WEIGHT + AC_GRAV_ACC * body_In[3][2]- m_head_rate * m_vel_forw;
	double acc_down = dyn_pres * SDOF_WING_SURFACE *	(-lift_coef - drag_coef * aoa) / SDOF_AC_WEIGHT + AC_GRAV_ACC * body_In[3][3] + m_pitch_rate * m_vel_forw;

	///translational velocities
	double vel_n_prev = m_vel_N;
	m_vel_N = body_In[1][1] * m_vel_forw + body_In[1][2] * m_vel_rght +	body_In[1][3] * m_vel_down;
	double vel_e_prev = m_vel_E;
	m_vel_E = body_In[2][1] * m_vel_forw + body_In[2][2] * m_vel_rght +	body_In[2][3] * m_vel_down;
	double vel_dwn_prev = m_vel_dwn;
	m_vel_dwn = body_In[3][1] * m_vel_forw + body_In[3][2] * m_vel_rght + body_In[3][3] * m_vel_down;

	m_acc_N = (m_vel_N - vel_n_prev) / dt;  /* ft/s2 */
	m_acc_E = (m_vel_E - vel_e_prev) / dt;  /* ft/s2 */
	m_acc_Dwn = (m_vel_dwn - vel_dwn_prev) / dt;

	///integrations
	m_vel_forw +=	(1.5 * acc_forw - 0.5 * m_acc_forw_prv) * dt;
	m_acc_forw_prv = acc_forw;

	if (m_vel_forw < 10.0) m_vel_forw = 10.0;

	/// if roll manuoevering is required to be changed then roll dumper m_roll_rate/vel_forw should be multiplied by a correspondent coef.
	double rol_moment_coef = -0.2 * ROLL_CONST * m_ailron_com - m_roll_rate / m_vel_forw;
	double roll_acc = dyn_pres * SDOF_WING_CHORD * SDOF_WING_SURFACE * rol_moment_coef / SDOF_INERTIAL_MOM_X;
	double pit_acc = dyn_pres * SDOF_WING_CHORD * SDOF_WING_SURFACE * pit_moment_coef / SDOF_INERTIAL_MOM_Y;
	double head_acc = dyn_pres * SDOF_WING_CHORD * SDOF_WING_SURFACE * yaw_moment_coef / SDOF_INERTIAL_MOM_Z +
		(SDOF_INERTIAL_MOM_X - SDOF_INERTIAL_MOM_Y) * m_roll_rate * m_pitch_rate / SDOF_INERTIAL_MOM_Z;

	m_vel_rght +=	(1.5 * acc_rght - 0.5 * m_acc_rght_prv) * dt;
	m_acc_rght_prv = acc_rght;
	m_vel_down += (1.5 * acc_down - 0.5 * m_acc_down_prv) * dt;
	m_acc_down_prv = acc_down;
	m_roll_rate += (1.5 * roll_acc - 0.5 * m_roll_acc_prv) * dt;
	m_roll_acc_prv = roll_acc;
	m_pitch_rate += (1.5 * pit_acc - 0.5 * m_pit_acc_prv) * dt;
	m_pit_acc_prv = pit_acc;
	m_head_rate +=	(1.5 * head_acc - 0.5 * m_head_acc_prv) * dt;
	m_head_acc_prv = head_acc;

	///derivatives of equations
	double e0dot = 0.5 * (-m_e1 * m_roll_rate - m_e2 * m_pitch_rate - m_e3 * m_head_rate);
	double e1dot = 0.5 * (m_e0 * m_roll_rate - m_e3 * m_pitch_rate + m_e2 * m_head_rate);
	double e2dot = 0.5 * (+m_e3 * m_roll_rate + m_e0 * m_pitch_rate - m_e1 * m_head_rate);
	double e3dot = 0.5 * (-m_e2 * m_roll_rate + m_e1 * m_pitch_rate + m_e0 * m_head_rate);

	/// components of rotation	matrix(EULER ANGLES)
	m_e0 = m_e0 + (1.5 * e0dot - 0.5 * m_e0dotp) * dt;
	m_e0dotp = e0dot;
	m_e1 = m_e1 + (1.5 * e1dot - 0.5 * m_e1dotp) * dt;
	m_e1dotp = e1dot;
	m_e2 = m_e2 + (1.5 * e2dot - 0.5 * m_e2dotp) * dt;
	m_e2dotp = e2dot;
	m_e3 = m_e3 + (1.5 * e3dot - 0.5 * m_e3dotp) * dt;
	m_e3dotp = e3dot;

	double ep = pow(m_e0, 2) + pow(m_e1, 2) + pow(m_e2, 2) + pow(m_e3, 2) - 1.;
	ep = 1. - 0.5 * ep;
	m_e0 = m_e0 * ep;
	m_e1 = m_e1 * ep;
	m_e2 = m_e2 * ep;
	m_e3 = m_e3 * ep;

	m_alt -= m_vel_dwn * dt;
	m_lat += (m_vel_N * dt	/ (Earth_Radius_M + m_alt)) * RAD_TO_DEG;
	m_lon += (m_vel_E * dt	/ ((Earth_Radius_M + m_alt) * cos(m_lat * DEG_TO_RAD))) * RAD_TO_DEG;
}
