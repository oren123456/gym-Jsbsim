import math

METER_TO_KNOTS = 3.280839 / 1.6887
FAcs_Drag_Factor = 0.064
MAX_SDOF_PARAM = 9
SDOF_TRUST_FACTOR = 0
SDOF_TRUST = 1
SDOF_INERTIAL_MOM_X = 2
SDOF_INERTIAL_MOM_Y = 3
SDOF_INERTIAL_MOM_Z = 4
SDOF_AC_WEIGHT = 5
SDOF_WING_SURFACE = 6
SDOF_WING_CHORD = 7
SDOF_GRAV_CENTER = 8
AC_GRAV_ACC = 9.806  # m/s^2
RAD_TO_DEG = 180. / math.pi
DEG_TO_RAD = math.pi / 180
FEET_TO_METER = 1. / 3.280839
Earth_Radius = 2.092646982E7  # earth radius at the equator [feet]
Earth_Radius_M = Earth_Radius * FEET_TO_METER
PITCH_CONST = 0.10
ROLL_CONST = 0.06


class DSIExec:

    def __init__(self):
        self.alt = None
        self.v_west_mps = None
        self.v_up_mps = None
        self.v_north_mps = None

        self.long = None
        self.lat = None
        self.vel_forw = None
        self.e3 = None
        self.e2 = None
        self.e1 = None
        self.e0 = None
        self.e3dotp = None
        self.e2dotp = None
        self.e1dotp = None
        self.e0dotp = None,
        self.vel_rght = None
        self.vel_down = None
        self.head_rate = None
        self.roll_rate = None
        self.pitch_rate = None
        self.head_acc_prv = None
        self.roll_acc_prv = None
        self.pit_acc_prv = None
        self.acc_down_prv = None
        self.acc_forw_prv = None
        self.acc_rght_prv = None
        self.dt = None
        self._fields = {}

    def set_dt(self, dt):
        self.dt = dt

    def run_ic(self, ):
        # print(f"run_ic.")
        self.lat = self._fields['ic/lat-geod-deg']
        self.long = self._fields['ic/long-gc-deg']
        # self._fields["position/h_sl_ft"] = self._fields['ic/h-sl-ft']
        self.alt = self._fields['ic/h-sl-ft'] * 0.3048  # meters
        self.vel_forw = self._fields['ic/u-fps'] * 0.3048  # m/s
        self._fields["attitude/pitch-rad"] = 0  # rad
        self._fields["attitude/roll-rad"] = 0  # rad
        init_heading = self._fields['ic/psi-true-deg'] * 0.017453  # rad
        self._fields["attitude/heading_true_rad"] = init_heading  # rad

        self.vel_down = 0.00
        self.vel_rght = 0.00
        # self._fields["velocities/v-north-fps"] = 0.00
        # self._fields["velocities/v-west-fps"] = 0.00
        # self._fields["velocities/v-up-fps"] = 0.00
        self.v_up_mps = 0.0
        self.v_north_mps = 0.0
        self.v_west_mps = 0.0

        self.pitch_rate = 0.00
        self.roll_rate = 0.00
        self.head_rate = 0.00

        self.acc_rght_prv = 0.
        self.acc_forw_prv = 0.
        self.acc_down_prv = 0.
        self.head_acc_prv = 0.
        self.pit_acc_prv = 0.
        self.roll_acc_prv = 0.

        self.e0dotp = 0.
        self.e1dotp = 0.
        self.e2dotp = 0.
        self.e3dotp = 0.

        self.e0 = math.cos(init_heading / 2)
        self.e1 = 0.
        self.e2 = 0.
        self.e3 = math.sin(init_heading / 2)

        air_density = 1.225 / (1. + 9.62 * pow(10., -5) * self.alt + 1.49 * pow(10., -8) * (pow(self.alt, 2)))
        dyn_pres = 0.5 * air_density * pow(self.vel_forw, 2)
        a = -1. / 80
        b = 1 - 200 * a
        xcl = a * self.vel_forw * METER_TO_KNOTS + b
        if xcl < 1:
            xcl = 1
        if xcl > 2.5:
            xcl = 2.5
        # aerodynamic equations
        aoa = math.atan2(self.vel_down, self.vel_forw)
        if self.vel_forw < 250:  # m/s
            lift_coef = 2.5 * aoa * xcl
        else:
            lift_coef = 3.5 * aoa * xcl

        cdsb = FAcs_Drag_Factor * (1 - aoa / 0.3)
        if cdsb < 0:
            cdsb = 0
        if cdsb > FAcs_Drag_Factor:
            cdsb = FAcs_Drag_Factor

        drag_coef = cdsb + (0.025 + 0.6 * (pow(lift_coef, 2))) * 1.25



        return True

    def set_property_value(self, field_names, value):
        self._fields[field_names] = value

    def get_property_value(self, field_names):
        if field_names not in self._fields:
            self._fields[field_names] = 0
        return self._fields[field_names]

    def run(self):
        return self.sdof_calculate_data()

    def sdof_calculate_data(self):
        # print(f"sdof_calculate_data.")
        # double body_In[4][4];
        body_in = [[0.0] * 4 for _ in range(4)]
        # rotation matrix
        body_in[1][1] = pow(self.e0, 2) + pow(self.e1, 2) - pow(self.e2, 2) - pow(self.e3, 2)
        body_in[1][2] = 2. * (self.e1 * self.e2 - self.e0 * self.e3)
        body_in[1][3] = 2. * (self.e1 * self.e3 + self.e0 * self.e2)
        body_in[2][1] = 2. * (self.e1 * self.e2 + self.e0 * self.e3)
        body_in[2][2] = pow(self.e0, 2) - pow(self.e1, 2) + pow(self.e2, 2) - pow(self.e3, 2)
        body_in[2][3] = 2. * (self.e2 * self.e3 - self.e0 * self.e1)
        body_in[3][1] = 2. * (self.e1 * self.e3 - self.e0 * self.e2)
        body_in[3][2] = 2. * (self.e2 * self.e3 + self.e0 * self.e1)
        body_in[3][3] = pow(self.e0, 2) - pow(self.e1, 2) - pow(self.e2, 2) + pow(self.e3, 2)

        # euler angles pitch = teta head = psi roll = phi

        # print( body_in[3][1])
        # print(math.sqrt(1 - pow(body_in[3][1], 2)))
        # self._fields["attitude/pitch-rad"] = -math.atan2(body_in[3][1], math.sqrt(1 - pow(body_in[3][1], 2)))
        self._fields["attitude/heading_true_rad"] = math.atan2(body_in[2][1], body_in[1][1])
        self._fields["attitude/roll-rad"] = math.atan2(body_in[3][2], body_in[3][3])



        a = -1. / 80
        b = 1 - 200 * a
        xcl = a * self.vel_forw * METER_TO_KNOTS + b
        if xcl < 1:
            xcl = 1
        if xcl > 2.5:
            xcl = 2.5
        # aerodynamic equations
        aoa = math.atan2(self.vel_down, self.vel_forw)
        if self.vel_forw < 250:  # m/s
            lift_coef = 2.5 * aoa * xcl + PITCH_CONST * self._fields["fcs/elevator-cmd-norm"]
        else:
            lift_coef = 3.5 * aoa * xcl + PITCH_CONST * self._fields["fcs/elevator-cmd-norm"]

        cdsb = FAcs_Drag_Factor * (1 - aoa / 0.3)
        if cdsb < 0:
            cdsb = 0
        if cdsb > FAcs_Drag_Factor:
            cdsb = FAcs_Drag_Factor

        drag_coef = cdsb + (0.025 + 0.6 * (pow(lift_coef, 2))) * 1.25
        side_slip = math.atan2(self.vel_rght, self.vel_forw)
        yaw_coef = -1.4 * side_slip
        pit_moment_coef = 0.005 - 0.05 * lift_coef - 0.28 * xcl * (
                PITCH_CONST * self._fields[
            "fcs/elevator-cmd-norm"] + 0.005 + self.vel_forw / 800. * 0.025) + lift_coef * (
                                  SDOF_GRAV_CENTER - 0.49) - 10 * self.pitch_rate / self.vel_forw
        yaw_moment_coef = 0.26 * side_slip - 0.08 * ROLL_CONST * self._fields["fcs/aileron-cmd-norm"] + yaw_coef * (
                SDOF_GRAV_CENTER - 0.49) - 2.8 * self.head_rate / self.vel_forw

        # dynamic pressure
        air_density = 1.225 / (1. + 9.62 * pow(10., -5) * self.alt + 1.49 * pow(10., -8) * (pow(self.alt, 2)))
        dyn_pres = 0.5 * air_density * pow(self.vel_forw, 2)
        trust = (60000. / (1. + self.alt / 15000.)) * self._fields["fcs/throttle-cmd-norm"]

        acc_forw = (dyn_pres * SDOF_WING_SURFACE * ( lift_coef * aoa - drag_coef) + trust) / SDOF_AC_WEIGHT + AC_GRAV_ACC * body_in[3][1]
        acc_rght = dyn_pres * SDOF_WING_SURFACE * (yaw_coef - drag_coef * side_slip) / SDOF_AC_WEIGHT + AC_GRAV_ACC * \
                   body_in[3][2] - self.head_rate * self.vel_forw
        acc_down = dyn_pres * SDOF_WING_SURFACE * (-lift_coef - drag_coef * aoa) / SDOF_AC_WEIGHT + AC_GRAV_ACC * \
                   body_in[3][3] + self.pitch_rate * self.vel_forw

        vel_n_prev = self.v_north_mps
        self.v_north_mps = body_in[1][1] * self.vel_forw + body_in[1][2] * self.vel_rght + body_in[1][3] * self.vel_down
        vel_e_prev = self.v_west_mps
        self.v_west_mps = body_in[2][1] * self.vel_forw + body_in[2][2] * self.vel_rght + body_in[2][3] * self.vel_down
        vel_dwn_prev = self.v_up_mps
        self.v_up_mps = body_in[3][1] * self.vel_forw + body_in[3][2] * self.vel_rght + body_in[3][3] * self.vel_down

        # self.acc_N = (self._fields["velocities/v-north-fps"] - vel_n_prev) / self.dt  # ft/s2
        # self.acc_E = (self._fields["velocities/v-west-fps"] - vel_e_prev) / self.dt  # ft/s2
        # self.acc_Dwn = (self._fields["velocities/v-up-fps"] - vel_dwn_prev) / self.dt

        # integrations
        self.vel_forw += (1.5 * acc_forw - 0.5 * self.acc_forw_prv) * self.dt
        self.acc_forw_prv = acc_forw
        if self.vel_forw < 10.0:
            self.vel_forw = 10.0

        # if roll manuoevering is required to be changed then roll dumper roll_rate/vel_forw should be multiplied by a
        # correspondent coef.
        rol_moment_coef = -0.2 * ROLL_CONST * self._fields["fcs/aileron-cmd-norm"] - self.roll_rate / self.vel_forw
        roll_acc = dyn_pres * SDOF_WING_CHORD * SDOF_WING_SURFACE * rol_moment_coef / SDOF_INERTIAL_MOM_X
        pit_acc = dyn_pres * SDOF_WING_CHORD * SDOF_WING_SURFACE * pit_moment_coef / SDOF_INERTIAL_MOM_Y
        head_acc = dyn_pres * SDOF_WING_CHORD * SDOF_WING_SURFACE * yaw_moment_coef / SDOF_INERTIAL_MOM_Z + (
                SDOF_INERTIAL_MOM_X - SDOF_INERTIAL_MOM_Y) * self.roll_rate * self.pitch_rate / SDOF_INERTIAL_MOM_Z

        self.vel_rght += (1.5 * acc_rght - 0.5 * self.acc_rght_prv) * self.dt
        self.acc_rght_prv = acc_rght
        self.vel_down += (1.5 * acc_down - 0.5 * self.acc_down_prv) * self.dt
        self.acc_down_prv = acc_down
        self.roll_rate += (1.5 * roll_acc - 0.5 * self.roll_acc_prv) * self.dt
        self.roll_acc_prv = roll_acc
        self.pitch_rate += (1.5 * pit_acc - 0.5 * self.pit_acc_prv) * self.dt
        self.pit_acc_prv = pit_acc
        self.head_rate += (1.5 * head_acc - 0.5 * self.head_acc_prv) * self.dt
        self.head_acc_prv = head_acc

        # derivatives of equations
        e0dot = 0.5 * (-self.e1 * self.roll_rate - self.e2 * self.pitch_rate - self.e3 * self.head_rate)
        e1dot = 0.5 * (self.e0 * self.roll_rate - self.e3 * self.pitch_rate + self.e2 * self.head_rate)
        e2dot = 0.5 * (self.e3 * self.roll_rate + self.e0 * self.pitch_rate - self.e1 * self.head_rate)
        e3dot = 0.5 * (-self.e2 * self.roll_rate + self.e1 * self.pitch_rate + self.e0 * self.head_rate)

        print(e0dot)
        # components of rotation matrix(EULER ANGLES)
        self.e0 += (1.5 * e0dot - 0.5 * self.e0dotp) * self.dt
        self.e0dotp = e0dot
        self.e1 += (1.5 * e1dot - 0.5 * self.e1dotp) * self.dt
        self.e1dotp = e1dot
        self.e2 += (1.5 * e2dot - 0.5 * self.e2dotp) * self.dt
        self.e2dotp = e2dot
        self.e3 += (1.5 * e3dot - 0.5 * self.e3dotp) * self.dt
        self.e3dotp = e3dot

        ep = pow(self.e0, 2) + pow(self.e1, 2) + pow(self.e2, 2) + pow(self.e3, 2) - 1.
        ep = 1. - 0.5 * ep
        self.e0 *= ep
        self.e1 *= ep
        self.e2 *= ep
        self.e3 *= ep
        # print(ep)
        # print(f" {self.e0}  {self.e1}  {self.e2} {self.e3}")


        self._fields["position/h_sl_ft"] = self.alt * 3.28084 - self.v_up_mps * self.dt
        self.lat += (self.v_north_mps * self.dt / (Earth_Radius_M + self._fields["position/h_sl_ft"])) * RAD_TO_DEG
        self.long += (self._fields['velocities/v-west-fps'] * self.dt / (
                (Earth_Radius_M + self._fields["position/h_sl_ft"]) * math.cos(self.lat * DEG_TO_RAD))) * RAD_TO_DEG
        self._fields["velocities/v-up-fps"] = self.v_up_mps * 3.28084
        self._fields["velocities/v-north-fps"] = self.v_north_mps * 3.28084
        self._fields["velocities/v-west-fps"] = self.v_west_mps * 3.28084
        return True
