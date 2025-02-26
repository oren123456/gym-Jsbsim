import time
import math
import threading
import io
import socket
import jsbsim
import tkinter as tk
from opendis.dis7 import EntityStatePdu, Vector3Float, EntityType, EulerAngles
from opendis.DataOutputStream import DataOutputStream
from opendis.PduFactory import createPdu
from opendis.RangeCoordinates import *

# Initialize UDP communication
DIS_IP = "127.0.0.1"
DIS_PORT = 3001

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_recv.bind(("", DIS_PORT))

# Initialize JSBSim
fdm = jsbsim.FGFDMExec(None)
fdm.set_debug_level(0)
fdm.load_model("f16")
fdm.run_ic()

time_step = 5  # Configurable time step
fdm.set_dt(time_step)

# GPS Helper
gps = GPS()

running = threading.Event()


# GUI for modifying entity_pdu parameters
def update_params():
    entity_pdu.entityID.entityID = int(entity_id_var.get())
    entity_pdu.entityID.siteID = int(site_id_var.get())
    entity_pdu.entityID.applicationID = int(app_id_var.get())
    entity_pdu.marking.setString(marking_var.get())
    entity_pdu.exerciseID = int(exercise_id_var.get())


def gui_thread():
    global received_data_var
    global entity_id_var, site_id_var, app_id_var, marking_var, exercise_id_var
    root = tk.Tk()

    received_data_var = tk.StringVar(value="No data received yet")
    root.title("Entity PDU Editor")

    tk.Label(root, text="Entity ID:").grid(row=0, column=0)
    entity_id_var = tk.StringVar(value="42")
    tk.Entry(root, textvariable=entity_id_var).grid(row=0, column=1)

    tk.Label(root, text="Site ID:").grid(row=1, column=0)
    site_id_var = tk.StringVar(value="17")
    tk.Entry(root, textvariable=site_id_var).grid(row=1, column=1)

    tk.Label(root, text="Application ID:").grid(row=2, column=0)
    app_id_var = tk.StringVar(value="23")
    tk.Entry(root, textvariable=app_id_var).grid(row=2, column=1)

    tk.Label(root, text="Marking:").grid(row=3, column=0)
    marking_var = tk.StringVar(value="Igor3d")
    tk.Entry(root, textvariable=marking_var).grid(row=3, column=1)

    tk.Label(root, text="Exercise ID:").grid(row=4, column=0)
    exercise_id_var = tk.StringVar(value="11")
    tk.Entry(root, textvariable=exercise_id_var).grid(row=4, column=1)

    tk.Button(root, text="Update", command=update_params).grid(row=5, column=0, columnspan=2)
    tk.Label(root, textvariable=received_data_var, wraplength=400, justify='left').grid(row=6, column=0, columnspan=2)

    def on_close():
        global running
        print("Closing application...")
        running.clear()
        root.quit()
        exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def send():
    global entity_pdu
    entity_pdu = EntityStatePdu()
    # Wait for GUI to initialize before calling update_params()
    while 'entity_id_var' not in globals():
        time.sleep(0.5)

    update_params()
    threading.Thread(target=recv_pdu, daemon=True).start()  # Start recv_pdu after send begins

    running.set()  # Start loop
    while running.is_set():
        fdm.run()
        lat = fdm.get_property_value("position/lat-geod-deg")
        lon = fdm.get_property_value("position/long-gc-deg")
        alt = fdm.get_property_value("position/h-sl-ft") * 0.3048
        pos_x, pos_y, pos_z = gps.lla2ecef((deg2rad(lat), deg2rad(lon), alt))

        vel_x = fdm.get_property_value("velocities/v-north-fps") * 0.3048
        vel_y = fdm.get_property_value("velocities/v-east-fps") * 0.3048
        vel_z = fdm.get_property_value("velocities/v-down-fps") * 0.3048

        entity_pdu.entityLocation.x = pos_x
        entity_pdu.entityLocation.y = pos_y
        entity_pdu.entityLocation.z = pos_z
        entity_pdu.entityLinearVelocity = Vector3Float(vel_x, vel_y, vel_z)

        buffer = io.BytesIO()
        output_stream = DataOutputStream(buffer)
        entity_pdu.serialize(output_stream)
        pdu_bytes = buffer.getvalue()
        sock_send.sendto(pdu_bytes, (DIS_IP, DIS_PORT))

        print(f"Sent EntityStatePdu | Location: ({lat:.2f}, {lon:.2f}, {alt:.1f}m)")
        time.sleep(time_step)
    print("Closing sock_send")
    sock_send.close()


def recv_pdu():

    running.set()  # Start loop
    while running.is_set():
        data = sock_recv.recv(1024)
        received_pdu = createPdu(data)

        if received_pdu is None:
            print("Received invalid or unsupported PDU, ignoring...")
            continue

        if not isinstance(received_pdu, EntityStatePdu):
            print("Received a non-EntityStatePdu, ignoring...")
            continue

        # Extract position and orientation
        pos_x, pos_y, pos_z = received_pdu.entityLocation.x, received_pdu.entityLocation.y, received_pdu.entityLocation.z
        vel_x, vel_y, vel_z = received_pdu.entityLinearVelocity.x, received_pdu.entityLinearVelocity.y, received_pdu.entityLinearVelocity.z
        psi, theta, phi = received_pdu.entityOrientation.psi, received_pdu.entityOrientation.theta, received_pdu.entityOrientation.phi

        # Convert ECEF back to Lat/Lon/Alt
        lat, lon, alt = gps.ecef2lla((pos_x, pos_y, pos_z))

        # Print received values
        received_data_var.set(f"Received EntityStatePdu:\n"
                              f" Latitude  : {rad2deg(lat):.2f} degrees\n"
                              f" Longitude : {rad2deg(lon):.2f} degrees\n"
                              f" Altitude  : {alt:.1f} meters\n"
                              f" Velocity  : ({vel_x:.2f}, {vel_y:.2f}, {vel_z:.2f}) m/s\n"
                              f" Orientation: Yaw {rad2deg(psi):.2f}, Pitch {rad2deg(theta):.2f}, Roll {rad2deg(phi):.2f} degrees\n"
                              f" Entity ID  : {received_pdu.entityID.entityID}\n"
                              f" Site ID    : {received_pdu.entityID.siteID}\n"
                              f" App ID     : {received_pdu.entityID.applicationID}\n"
                              f" Force ID   : {received_pdu.forceId}\n"
                              f" Marking    : {received_pdu.marking.characters}\n degrees")
    print("Closing sock_recv")
    sock_recv.close()


threading.Thread(target=gui_thread, daemon=True).start()
send()
