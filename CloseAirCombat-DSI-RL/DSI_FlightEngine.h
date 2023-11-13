#pragma once


#ifdef  DSIFLIGHTENGINE_EXPORTS
 DLL_EXPORT __declspec(dllexport)
#else
 DLL_EXPORT __declspec(dllimport)
#endif


class DLL_EXPORT  DSI_FlightEngine
{
public:

	void tick(float, float, float, float);

	void sdof_calculate_data(float);

	void reset(double lat, double lon, double alt, float, float);

	void getObs(double* obs);

private:
	double 	m_e0 = 1., m_e1 = 0., m_e2 = 0., m_e3 = 0.;
	double	m_e0dotp, m_e1dotp, m_e2dotp, m_e3dotp;
	
	double	m_acc_forw_prv;
	double	m_acc_rght_prv;
	double	m_acc_down_prv;
	double	m_pit_acc_prv ;
	double	m_roll_acc_prv ;
	double	m_head_acc_prv ;
	double	m_vel_rght;
	double	m_vel_down;

	double	m_pitch_rate;
	double	m_head_rate;
	double	m_roll_rate;
	double	m_vel_forw; // m/s

	float	m_throttle; // between 0 to 1
	float	m_elevator_com; // from - 1 to + 1
	float	m_ailron_com; // from - 1 to + 1

	float	m_delta_elev;
	float	m_delta_ailr;

private:
	double	m_lat; // deg
	double	m_lon; // deg
	double	m_alt; // meters

	double	m_pitch; // rad
	double	m_roll; // rad
	double	m_head; //rad

	double	m_vel_N; // m/s
	double	m_vel_E; // m/s
	double	m_vel_dwn; // m/s

	double	m_acc_N;
	double	m_acc_E;
	double	m_acc_Dwn;
};