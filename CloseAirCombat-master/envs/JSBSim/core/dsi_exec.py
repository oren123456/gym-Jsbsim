import ctypes
import multiprocessing.shared_memory
import sys
import math
import time

print(sys.version)

MAX_AA_TARGETS_D = 56
MAX_GROUND_TARGETS_D = 10
MAX_RWR_TARGETS_D = 10

START_LAT = 32.02 * math.pi / 180
START_LON = 34.85 * math.pi / 180
START_ALT = 2000


class StrLatLonAltType(ctypes.Structure):
    _fields_ = [
        ("lat", ctypes.c_float),  # rad
        ("lon", ctypes.c_float),  # rad
        ("alt", ctypes.c_float),  # ft
    ]


class StrVecType(ctypes.Structure):
    _fields_ = [
        ("north", ctypes.c_float),  # [ft/sec]
        ("west", ctypes.c_float),  # [ft/sec]
        ("up", ctypes.c_float),  # [ft/sec]
    ]


class StrAATgtDataType(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("pos", StrLatLonAltType),
        ("vel", StrVecType),
        ("color", ctypes.c_int),
        ("tgtId", ctypes.c_int),
        ("dlGroup", ctypes.c_ubyte),
        ("dlNumber", ctypes.c_ubyte),
    ]


class StrAcDataType(ctypes.Structure):
    _fields_ = [
        ("acType", ctypes.c_int),
        ("pos", StrLatLonAltType),
        ("heading", ctypes.c_float),  # [rad]
        ("magHeading", ctypes.c_float),  # [rad]
        ("yaw", ctypes.c_float),  # [rad]
        ("pitch", ctypes.c_float),  # [rad]
        ("roll", ctypes.c_float),  # [rad]
        ("gNum", ctypes.c_float),
        ("vel", StrVecType),  # [ft/sec]
        ("acc", StrVecType),  # [ft/sec]
        ("dlGroup", ctypes.c_ubyte),
        ("dlNumber", ctypes.c_ubyte),
    ]


class StrRwrGroundTgtType(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("pos", StrLatLonAltType),
        ("killRangeMeter", ctypes.c_float),
        ("mslLaunch", ctypes.c_int),
        ("searchTrack", ctypes.c_int),
    ]


class StrGroundTgtType(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("pos", StrLatLonAltType),
        ("killRangeMeter", ctypes.c_float),
    ]


class StrWindDataType(ctypes.Structure):
    _fields_ = [
        ("windExist", ctypes.c_int),
        ("windType", ctypes.c_int),
        ("direction", ctypes.c_float),  # [deg]
        ("constVel", ctypes.c_float),  # [ft/sec]
        ("linearVel0", ctypes.c_float),  # [ft/sec]
        ("linearVel10", ctypes.c_float),  # at 10000 ft [ft/sec]
    ]


class StrDsiCtrlType(ctypes.Structure):
    _fields_ = [
        ("real-joystick-control", ctypes.c_int),
        ("reset-scen", ctypes.c_int),
        ("start-pos", StrLatLonAltType),
        ("start-velocity", ctypes.c_float),  # [ft/sec]
        ("start-heading", ctypes.c_float),  # [rad]
        ("windData", StrWindDataType),
        ("episodeNum", ctypes.c_uint),
        ("scenarioNum", ctypes.c_uint),
        ("acSetDestination", ctypes.c_int),
        ("acDestination", StrLatLonAltType),
    ]


class StrAcToRlType(ctypes.Structure):
    _fields_ = [
        ("time-tag", ctypes.c_ulong),
        ("counter", ctypes.c_ulong),
        ("avnMode", ctypes.c_int),
        ("avnSubMode", ctypes.c_int),
        ("ac-data", StrAcDataType),
        ("aaTgtsNum", ctypes.c_int),
        ("aaTgts", StrAATgtDataType * MAX_AA_TARGETS_D),
        ("rwrTgtsNum", ctypes.c_int),
        ("rwrTgts", StrRwrGroundTgtType * MAX_RWR_TARGETS_D),
        ("grndTgtsNum", ctypes.c_int),
        ("grndTgts", StrGroundTgtType * MAX_GROUND_TARGETS_D),
        ("fuelQty", ctypes.c_float),
        ("acDestination", StrLatLonAltType),
    ]


class StrRlToDsiType(ctypes.Structure):
    _fields_ = [
        ("time-tag", ctypes.c_ulong),
        ("counter", ctypes.c_ulong),
        ("joystick-x", ctypes.c_float),  # [1370, 2590]
        ("joystick-y", ctypes.c_float),  # [1380, 2390]
        ("throttle", ctypes.c_float),  # [2060, 2700]
        ("dsi-ctrl-data", StrDsiCtrlType),
    ]


class StrDsiToRlType(ctypes.Structure):
    _fields_ = [
        ("time-tag", ctypes.c_ulong),
        ("counter", ctypes.c_ulong),
        ("joystick-x", ctypes.c_float),  # [1370, 2590]
        ("joystick-y", ctypes.c_float),  # [1380, 2390]
        ("throttle", ctypes.c_float),  # [2060, 2700]
        ("episode-num", ctypes.c_uint),
        ("scenario-num", ctypes.c_uint),
    ]


class StrMainType(ctypes.Structure):
    _fields_ = [
        ("dsi-rl", StrDsiToRlType),
        ("ac-rl", StrAcToRlType),
        ("rl-dsi", StrRlToDsiType),
    ]




# Define the data structure to match shared memory structure
def get_shared_memory(shm_name, data_size):
    try:
        shared_mem = multiprocessing.shared_memory.SharedMemory(name=shm_name, create=False, size=data_size)
        return shared_mem  # Return the data and the shared memory object
    except FileNotFoundError:
        print(f"Shared memory '{shm_name}' not found.")
        return None, None
    except Exception as e:
        print(f"Error reading shared memory '{shm_name}': {e}")
        return None, None


normalize_from_to = {
                "joystick-x": [-1, 1, 1370, 2590],
                "joystick-y": [-1, 1, 1380, 2390],
                "throttle": [0, 1, 2060, 2700]
            }

def mapToRange(input_val, ranges):
    # Ensure the input is within the specified range
    in_min, in_max, out_min, out_max = ranges
    if input_val < in_min:
        input = in_min
    elif input_val > in_max:
        input_Val = in_max;
    # Perform linear interpolation
    return (input_val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def set_property_value(structure, field_names, value):
    field_name, *rest = field_names.split('.')
    try:
        if rest:
            field = getattr(structure, field_name)
            set_property_value(field, '.'.join(rest), value)
        else:
            if field_name in normalize_from_to:
                value = mapToRange (value, normalize_from_to[field_name] )
            setattr(structure, field_name, value)
    except AttributeError:
        raise ValueError(f"prop name unhandled: ({field_names})")


def get_property_value(struct, field_names):
    field_name, *rest = field_names.split('.')
    try:
        field = getattr(struct, field_name)
        return get_property_value(field, '.'.join(rest)) if rest else field
    except AttributeError:
        raise ValueError(f"prop name unhandled: ({field_names})")


class DSIExec:
    extraData = {}

    def __init__(self, read_from_ac, write_to_dsi, read_from_dsi):
        super().__init__()
        # self.mainStruct = StrMainType()
        self.mainType = {}

        self.ac_shared_memory = get_shared_memory(read_from_ac, ctypes.sizeof(StrAcToRlType))
        self.mainType["ac-rl"] = StrAcToRlType.from_buffer(self.ac_shared_memory.buf)
        # setattr(self.mainType, "ac-rl", StrAcToRlType.from_buffer(self.ac_shared_memory.buf))

        self.dsi_shared_memory = get_shared_memory(read_from_dsi, ctypes.sizeof(StrDsiToRlType))
        self.mainType["dsi-rl"] = StrDsiToRlType.from_buffer(self.dsi_shared_memory.buf)
        # setattr(self.mainType, "dsi-rl", StrDsiToRlType.from_buffer(self.dsi_shared_memory.buf))

        self.rl_shared_memory = get_shared_memory(write_to_dsi, ctypes.sizeof(StrRlToDsiType))
        self.mainType["rl-dsi"] = StrRlToDsiType.from_buffer(self.rl_shared_memory.buf)
        # rl_shared_memory_data = StrRlToDsiType.from_buffer(self.rl_shared_memory.buf)
        # b_ptr = ctypes.pointer(self.mainStruct.test)
        # ctypes.memmove(ctypes.addressof(b_ptr.contents), ctypes.addressof(rl_shared_memory_data), ctypes.sizeof(StrRlToDsiType))

        # setattr(self.mainType, "rl-dsi", StrRlToDsiType.from_buffer(self.rl_shared_memory.buf))
        # self.rl_shared_memory_data = StrRlToDsiType.from_buffer(self.rl_shared_memory.buf)
        # storage["a"] = StrRlToDsiType.from_buffer(self.rl_shared_memory.buf)

        # if not getattr(self.mainType, "ac-rl") or not getattr(self.mainType, "dsi-rl") or not getattr(self.mainType, "rl-dsi"):
        if not self.mainType["ac-rl"] or not self.mainType["dsi-rl"] or not self.mainType["rl-dsi"]:
            raise RuntimeError("Failed to connect to DSI Shared Memory.")

    def run(self):
        counter1 = self.get_property_value("dsi-rl.counter")
        counter2 = self.get_property_value("rl-dsi.counter")
        # print(time.time())
        while counter1 != counter2:
            counter1 = self.get_property_value("dsi-rl.counter")
            # print(f"{counter1} {counter2}" )
        return True

    def close(self):
        return True
        # self.ac_shared_memory.close()
        # self.dsi_shared_memory.close()
        # self.rl_shared_memory.close()

    def set_property_value(self, field_names, value):
        field_name, *rest = field_names.split('.')
        if field_name == "extra-data":
            self.extraData[field_names] = value
        else:
            # print(f"{rest} {value}")
            counter = self.get_property_value("dsi-rl.counter")
            set_property_value(self.mainType["rl-dsi"], "counter", counter + 1)
            set_property_value(self.mainType[field_name], '.'.join(rest), value)

    def get_property_value(self, field_names):
        field_name, *rest = field_names.split('.')
        if field_name == "extra-data":
            if field_names not in self.extraData:
                self.extraData[field_names] = 0
            return self.extraData[field_names]
        else:
            return get_property_value(self.mainType[field_name], '.'.join(rest))

    def test_all(self):
        self.set_property_value("rl-dsi.dsi-ctrl-data.start-heading", 0)
        self.set_property_value("rl-dsi.dsi-ctrl-data.reset-scen", 1)
        self.set_property_value("rl-dsi.joystick-x", 1)

        while True:
            print(self.get_property_value("dsi-rl.joystick-x"))


if __name__ == '__main__':
    msg_per_sec = .01
    a = DSIExec("AcToRl.shm", "RlToDsi.shm", "DsiToRl.shm")
    a.test_all()

    # ctypes.memmove(ctypes.byref(self.rl_shared_memory_data), ctypes.byref(data), ctypes.sizeof(StrRlToDsiType))
    # print(self.rl_shared_memory_data)
    # print(aa)
    # while True:
    #     self.set_property_value("rl-dsi.dsi-ctrl-data.reset-scen", 1)
    #     self.set_property_value("rl-dsi.joystick-x", 23)
    #     # self.set_property_value("extra-data.dsiCtrlData.counter", 2)
    #     # print(getattr(self.dsiToRl_sm_data, "counter"))
    #     # print(self.get_property_value("acToRl.ac-data.pos.lat"))
    #     # print(self.get_property_value("extra-data.dsiCtrlData.counter"))
    #     print(self.get_property_value("dsi-rl.joystick-x"))

    # setattr(self.mainType, "rl-dsi", self.rl_shared_memory_data)
    # aa = getattr(self.mainType, "rl-dsi")
    # t = getattr(self.mainType["rl-dsi"], "dsi-ctrl-data")
    # t = getattr(self.mainStruct.test, "dsi-ctrl-data")
    # setattr(t, "start-heading", 3.14)
    # setattr(t, "reset-scen", 1)

    # def rad_to_dms(radians, d_type):
    #     # Determine the direction based on d_type
    #     if d_type == "lat":
    #         direction = 'N' if radians >= 0 else 'S'
    #     elif d_type == "lon":
    #         direction = 'E' if radians >= 0 else 'W'
    #     else:
    #         raise ValueError("Invalid d_type. Use 'lat' for latitude or 'long' for longitude.")
    #
    #     # Convert radians to degrees
    #     degrees = abs(radians) * (180 / math.pi)
    #
    #     # Extract the whole degrees
    #     whole_degrees = int(degrees)
    #
    #     # Calculate the remaining decimal part
    #     decimal_part = degrees - whole_degrees
    #
    #     # Convert the decimal part to minutes and seconds
    #     minutes = int(decimal_part * 60)
    #     seconds = int((decimal_part * 60 - minutes) * 60)  # Convert to integer
    #
    #     # Format the result as "dd:mm:sss Direction"
    #     result = f"{whole_degrees:02d}:{minutes:02d}.{seconds:04d}{direction}"
    #     if d_type == "long":
    #         result = f"{whole_degrees:03d}:{minutes:02d}.{seconds:04d}{direction}"
    #     return result

    # for field_info in structure._fields_:
    #     if field_info[0] == field_name:
    #         setattr(structure, field_name, value)
    #         return

    # except AttributeError:
    #     raise ValueError(f"prop name unhandled: ({name})")

    # def reset_dsi(self):
    #     data = StrRlToDsiType()
    #     data.dsiCtrlData.startHeading = 0
    #     start_pos = StrLatLonAltType()
    #     start_pos.lat = START_LAT
    #     start_pos.lon = START_LON
    #     start_pos.alt = START_ALT
    #     data.dsiCtrlData.startPos = start_pos
    #     data.dsiCtrlData.resetScen = 1
    #     ctypes.memmove(ctypes.byref(self.rlToDsi_sm_data), ctypes.byref(data), ctypes.sizeof(StrRlToDsiType))

    # if self.acToRl_sm_data:
    #     self.data["position/lat-gc-rad"] = getattr(self.acToRl_sm_data.ac-data.pos, "lat")
    #     self.data["position/long-gc-rad"] = getattr(self.acToRl_sm_data.acdata.pos, "lon")
    #     self.data["position/h-sl-ft"] = getattr(self.acToRl_sm_data.ac-data.pos, "alt")
    #     self.data["attitude/roll-rad"] = getattr(self.acToRl_sm_data.ac-data, "roll")
    #     self.data["attitude/pitch-rad"] = getattr(self.acToRl_sm_data.ac-data, "pitch")
    #     self.data["attitude/heading-true-rad"] = getattr(self.acToRl_sm_data.ac-data, "yaw")
    #     self.data["velocities/v-north-fps"] = getattr(self.acToRl_sm_data.vel, "North")
    #     self.data["velocities/v-west-fps"] = getattr(self.acToRl_sm_data.vel, "West")
    #     self.data["velocities/v-up-fps"] = getattr(self.acToRl_sm_data.vel, "Up")
    # else:
    #     raise ValueError("Shared memory acToRl_sm_data not availble. Check if DSI is running...")
    #
    # if self.rlToDsi_sm_data:
    #     act = StrRlToDsiType()
    #     act.joystick-x = 1
    #     act.joystickY = -2
    #     act.throttle = 1
    #     act.counter = 2
    #     ctypes.memmove(ctypes.byref(self.rlToDsi_sm_data), ctypes.byref(act), ctypes.sizeof(StrRlToDsiType))
    # else:
    #     raise ValueError("Shared memory rlToDsi_sm_data not availble. Check if DSI is running...")

    #
    # def get_ac_state(self):
    #     if self.acToRl_sm_data:
    #         print(rad_to_dms(getattr(self.acToRl_sm_data.ac-data.pos, "lat"), "lat"))
    #         print(rad_to_dms(getattr(self.acToRl_sm_data.ac-data.pos, "lon"), "lon"))
    #         print(getattr(self.acToRl_sm_data.ac-data.pos, "alt"))
    #     else:
    #         print("Shared memory acToRl_sm_data not availble. Check if DSI is running...")

    # def get_dsi_state(self):
    #     if self.dsiToRl_sm_data:
    #         print(getattr(self.dsiToRl_sm_data, "joystick-x"))
    #         print(getattr(self.dsiToRl_sm_data, "counter"))
    #     else:
    #         print("Shared memory dsiToRl_shared_memory not availble. Check if DSI is running...")

# def update_dsi(self):
#        self.rlToDsi_sm_data.joystick-x = 11
# self.action_var = [
#     c.fcs_aileron_cmd_norm,
#     c.fcs_elevator_cmd_norm,
#     c.fcs_rudder_cmd_norm,
#     c.fcs_throttle_cmd_norm,
# ]

# if self.rlToDsi_sm_data:
#     data = StrRlToDsiType()
#     data.joystick-x = 1
#     data.joystickY = -2
#     data.throttle = 1
#     data.counter = 2
#     ctypes.memmove(ctypes.byref(self.rlToDsi_sm_data), ctypes.byref(a), ctypes.sizeof(StrRlToDsiType))
# else:
#     print("Shared memory rlToDsi_sm_data not availble. Check if DSI is running...")

# a.dsiCtrlData.startHeading = 0
# startPos = StrLatLonAltType()
# startPos.lat = 32.02 * math.pi / 180
# startPos.lon = 34.85 * math.pi / 180
# startPos.alt = 2000
# a.dsiCtrlData.startPos = startPos
# a.dsiCtrlData.resetScen = 1
# cc = 0

# if r_shared_memory_data and r2_shared_memory_data and w_shared_memory_data:
#         cc = cc + 1
#         a = StrRlToDsiType()
#         a.joystick-x = 1
#         a.joystickY = -2
#         a.counter = cc
#         a.throttle = 1
#         ctypes.memmove(ctypes.byref(w_shared_memory_data), ctypes.byref(a), ctypes.sizeof(StrRlToDsiType))
#         aa = "cc:" + cc.__str__()
#         print(aa)
#
#         # print(getattr(r2_shared_memory_data, "joystick-x"))
#         aaa = "counter:" + str(getattr(r2_shared_memory_data, "counter"))
#         print(aaa)
#
#         # print(rad_to_dms(getattr(r_shared_memory_data.ac-data.pos, "lat"), "lat"))
#         # print(rad_to_dms(getattr(r_shared_memory_data.ac-data.pos, "lon"), "lon"))
#         # print(getattr(r_shared_memory_data.ac-data.pos, "alt"))
#
#         # print(getattr(r_shared_memory_data.ac-data ,"yaw")*180/math.pi%360)
#         # print(getattr(r_shared_memory_data.ac-data, "pitch") * 180 / math.pi )
#         # print(getattr(r_shared_memory_data.ac-data, "roll") * 180 / math.pi)
#         # print(getattr(r_shared_memory_data.ac-data, "gNum"))
#
#         # print(getattr(r_shared_memory_data.ac-data.vel, "North"))
#         # print(getattr(r_shared_memory_data.ac-data.vel, "West"))
#         # print(getattr(r_shared_memory_data.ac-data.vel, "Up"))
#
#         # print(getattr(r_shared_memory_data.ac-data.acc, "North"))
#         # print(getattr(r_shared_memory_data.ac-data.acc, "West"))
#         # print(getattr(r_shared_memory_data.ac-data.acc, "Up"))
#
#         # for field_name, field_type in r_shared_memory_data.ac-data.vel._fields_:
#         #     field_value = getattr(r_shared_memory_data.ac-data.vel, field_name)
#         #     print(f"{field_name}: {field_value}")
#
#         # ("heading", ctypes.c_float),
#         # ("magHeading", ctypes.c_float),
#         # ("yaw", ctypes.c_float),
#         # ("pitch", ctypes.c_float),
#         # ("roll", ctypes.c_float),
#         # ("gNum", ctypes.c_float),
#         # ("vel", StrVecType),
#         # ("acc", StrVecType),
#
#         # value_d_latit = rad_to_dms(shared_memory_data.d_latit, "lat")
#         # value_d_longit = rad_to_dms(shared_memory_data.d_longit, "long")
#         # value_f_alt_sea = round((shared_memory_data.f_alt_sea * 0.3048), 2)   # converted feet to meters
#         # value_f_vel_n = round(shared_memory_data.f_vel_n, 2)
#         # value_f_vel_e = round(shared_memory_data.f_vel_e, 2)
#         # value_f_f_vel_up = round(shared_memory_data.f_vel_up, 2)
#
#         # print("LAT:", value_d_latit, "  LONG:", value_d_longit, "  ALT:", value_f_alt_sea)
#         # print("V_N:", value_f_vel_n, "  V_E:", value_f_vel_e, "  V_UP:", value_f_f_vel_up)
#         # print("_________________________________________________")
#
#         # shared_memory.close()     # When done with the shared memory segment, close it
#
#         # a = {"lat": value_d_latit, "long": value_d_longit, "alt": value_f_alt_sea, "vel_n": value_f_vel_n,
#         #      "vel_e": value_f_vel_e, "vel_up": value_f_f_vel_up}
#         # print(getattr(r_shared_memory_data.ac-data.pos, "lat"))
#         # for field_name, field_type in shared_memory_data.ac-data.pos._fields_:
#         #     field_value = getattr(shared_memory_data.ac-data.pos, field_name)
#         #     print(f"{field_name}: {field_value}")
#         time.sleep(intervals)
#
#     if r_shared_memory_data is None:
#         print("Shared memory not availble. Check if DSI is running...")
#         return None
