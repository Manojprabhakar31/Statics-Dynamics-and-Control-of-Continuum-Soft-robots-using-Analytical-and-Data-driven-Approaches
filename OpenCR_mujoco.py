import math
import numpy as np
from typing import List, Dict
import xml.etree.ElementTree as ET
import mujoco
from xml.dom import minidom
# =====================================
# OpenCR Mujoco UNIVERSITY OF TORONTO
# =====================================

# Helper function to normalize inputs
ACTUATOR_NAMES = [
    "act_seg_0_t0", "act_seg_0_t1", "act_seg_0_t2",  # Segment 0
    "act_seg_1_t0", "act_seg_1_t1", "act_seg_1_t2"   # Segment 1
]

def generate_tdcr_xml_from_config(config: Dict, output_filename: str = "generated_tdcr.xml"):
    """Generate a MuJoCo TDCR XML from the robot configuration."""
    # 1. READ CONFIGURATION
    num_segments = int(config["num_segments"])
    links_per_segment_cfg = config["links_per_segment"]
    segment_lengths_cfg = config["segment_lengths"]

    links_per_segment = [int(links_per_segment_cfg[str(i + 1)]) for i in range(num_segments)]
    segment_lengths = [float(segment_lengths_cfg[str(i + 1)]) for i in range(num_segments)]

    if len(links_per_segment) != num_segments:
        raise ValueError("links_per_segment does not match num_segments")
    if len(segment_lengths) != num_segments:
        raise ValueError("segment_lengths does not match num_segments")

    material = config["material_properties"]
    E = float(material["youngs_modulus"])
    nu = float(material["poisson_ratio"])
    density = float(material["density"])
    outer_radius = float(material["outer_radius"])
    inner_radius = float(material.get("inner_radius", 0.0))

    actuation_details = config["actuation_details"]
    tendon_segments = actuation_details["segments"]
    num_tendons_per_seg = []
    tendon_radii = []

    for s in range(num_segments):
        tendon_data = tendon_segments[s]
        num_tendons_per_seg.append(int(tendon_data["number_of_tendons"]))
        tendon_radii.append(float(tendon_data["distance_to_backbone"]))

    segment_angle_offsets = config.get("segment_angle_offsets", [0.0] * num_segments)
    if len(segment_angle_offsets) != num_segments:
        raise ValueError("segment_angle_offsets must have one value per segment")
    segment_angle_offsets = [float(x) for x in segment_angle_offsets]

    joints_per_link = int(config.get("joints_per_link", 3))
    enable_x = joints_per_link >= 1
    enable_y = joints_per_link >= 2
    enable_z = joints_per_link >= 3

    # 2. MATERIAL / SECTION CALCULATIONS
    G = E / (2.0 * (1.0 + nu))
    Ixx = math.pi / 4.0 * (outer_radius**4 - inner_radius**4)
    Iyy = Ixx
    J = math.pi / 2.0 * (outer_radius**4 - inner_radius**4)

    # 3. CALCULATE SEGMENT / LINK PROPERTIES
    segment_properties = []
    total_length = 0.0
    total_links = 0

    for s in range(num_segments):
        Ls = segment_lengths[s]
        Ns = links_per_segment[s]
        li = Ls / Ns
        kx = E * Ixx / li
        ky = E * Iyy / li
        kz = G * J / li
        link_volume = math.pi * (outer_radius**2 - inner_radius**2) * li
        link_mass = density * link_volume

        segment_properties.append({
            "length": Ls,
            "num_links": Ns,
            "link_length": li,
            "kx": kx,
            "ky": ky,
            "kz": kz,
            "link_mass": link_mass
        })
        total_length += Ls
        total_links += Ns

    # 4. MUJOCO ROOT
    mujoco = ET.Element("mujoco")
    ET.SubElement(mujoco, "compiler", angle="radian", autolimits="true")
    gravity = config.get("gravity", "0 0 0")
    ET.SubElement(mujoco, "option", gravity=gravity, noslip_iterations="1")

    # 5. ASSETS
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="1 1 1", rgb2=".6 .8 1", width="256", height="256")

    # 6. DEFAULTS
    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", condim="1", friction="1.0 0.005 0.0001")
    collision_default = ET.SubElement(default, "default", {"class": "tdcr_collision"})
    ET.SubElement(collision_default, "geom", contype="1", conaffinity="1", condim="3", friction="1.0 0.005 0.0001", solimp="0.8 0.95 0.001")

    # 7. JOINT DEFAULTS
    for s in range(num_segments):
        props = segment_properties[s]
        kx, ky, kz = props["kx"], props["ky"], props["kz"]
        damping_str = str(config.get("joint_damping", 0.001))

        ET.SubElement(default, "default", {"class": f"segment_{s}_x_joint"}).append(
            ET.Element("joint", {"type": "hinge", "axis": "0 0 1", "stiffness": str(kz), "damping": damping_str, "limited": "false"})
        )
        ET.SubElement(default, "default", {"class": f"segment_{s}_y_joint"}).append(
            ET.Element("joint", {"type": "hinge", "axis": "0 1 0", "stiffness": str(ky), "damping": damping_str, "limited": "false"})
        )
        ET.SubElement(default, "default", {"class": f"segment_{s}_z_joint"}).append(
            ET.Element("joint", {"type": "hinge", "axis": "1 0 0", "stiffness": str(kx), "damping": damping_str, "limited": "false"})
        )

    # 8. WORLDBODY
    worldbody = ET.SubElement(mujoco, "worldbody")
    mocap_base = ET.SubElement(worldbody, "body", {"name": "mocap_base", "pos": "0 0 0", "mocap": "true"})

    # 9. HELPER: TENDON XY POSITION
    def get_tendon_xy(tendon_index: int, number_of_tendons: int, radius: float, angle_offset: float):
        angle = angle_offset + 2.0 * math.pi * tendon_index / number_of_tendons
        return radius * math.cos(angle), radius * math.sin(angle)

    # 10. LINK INDEX MAP
    segment_link_ranges = []
    current_link = 1
    for s in range(num_segments):
        start_link = current_link
        end_link = current_link + links_per_segment[s] - 1
        segment_link_ranges.append((start_link, end_link))
        current_link = end_link + 1

    # 11. BASE LINK
    first_li = segment_properties[0]["link_length"]
    base_link = ET.SubElement(mocap_base, "body", {"name": "link_0", "pos": "0 0 0"})
    base_mass = density * (math.pi * (outer_radius**2 - inner_radius**2) * (first_li / 2.0))

    ET.SubElement(base_link, "geom", {
        "fromto": f"0 0 0 0 0 {first_li/2:.9f}",
        "size": f"{outer_radius:.9f}",
        "type": "cylinder",
        "rgba": "0.274 0.274 0.274 1.000",
        "mass": f"{base_mass:.12g}",
        "name": "geom_0",
        "class": "tdcr_collision"
    })

    for s in range(num_segments):
        radius, offset, nt = tendon_radii[s], segment_angle_offsets[s], num_tendons_per_seg[s]
        for t in range(nt):
            tx, ty = get_tendon_xy(t, nt, radius, offset)
            ET.SubElement(base_link, "site", {"name": f"l_0_s{s}_s1_t{t}", "pos": f"{tx:.9f} {ty:.9f} 0", "rgba": "0 0 0 0"})
            ET.SubElement(base_link, "site", {"name": f"l_0_s{s}_s2_t{t}", "pos": f"{tx:.9f} {ty:.9f} {first_li/2:.9f}", "rgba": "0 0 0 0"})

    # 12. BUILD KINEMATIC TREE
    parent_body = base_link
    global_link_idx = 1

    for s in range(num_segments):
        props = segment_properties[s]
        li, link_mass = props["link_length"], props["link_mass"]
        start_link, end_link = segment_link_ranges[s]
        nt, radius, offset = num_tendons_per_seg[s], tendon_radii[s], segment_angle_offsets[s]

        for l_in_seg in range(links_per_segment[s]):
            z_pos = li / 2.0 if global_link_idx == 1 else li
            body = ET.SubElement(parent_body, "body", {"name": f"link_{global_link_idx}", "pos": f"0 0 {z_pos:.9f}"})

            is_final_link = (global_link_idx == total_links)
            geom_length = li / 2.0 if is_final_link else li
            geom_mass = link_mass / 2.0 if is_final_link else link_mass
            rgba = "0.274 0.274 0.274 1.000" if s % 2 == 0 else "0.574 0.574 0.574 1.000"

            ET.SubElement(body, "geom", {
                "fromto": f"0 0 0 0 0 {geom_length:.9f}",
                "size": f"{outer_radius:.9f}",
                "type": "cylinder",
                "rgba": rgba,
                "mass": f"{geom_mass:.12g}",
                "name": f"geom_{global_link_idx}",
                "class": "tdcr_collision"
            })

            if enable_x:
                ET.SubElement(body, "joint", {"class": f"segment_{s}_x_joint", "name": f"joint_{global_link_idx}_x"})
            if enable_y:
                ET.SubElement(body, "joint", {"class": f"segment_{s}_y_joint", "name": f"joint_{global_link_idx}_y"})
            if enable_z:
                ET.SubElement(body, "joint", {"class": f"segment_{s}_z_joint", "name": f"joint_{global_link_idx}_z"})

            for t in range(nt):
                tx, ty = get_tendon_xy(t, nt, radius, offset)
                if not is_final_link:
                    ET.SubElement(body, "site", {"name": f"l_{global_link_idx}_s{s}_s1_t{t}", "pos": f"{tx:.9f} {ty:.9f} {li/2:.9f}", "rgba": "0 0 0 0"})
                ET.SubElement(body, "site", {"name": f"l_{global_link_idx}_s{s}_s2_t{t}", "pos": f"{tx:.9f} {ty:.9f} {geom_length:.9f}", "rgba": "0 0 0 0"})

            parent_body = body
            global_link_idx += 1

    # 13. END EFFECTOR
    final_li = segment_properties[-1]["link_length"]
    ee_body = ET.SubElement(parent_body, "body", {"name": "EE_pos", "pos": f"0 0 {final_li/2:.9f}"})
    ET.SubElement(ee_body, "site", {"name": "force_site_tip", "pos": "0 0 0", "size": "0.003", "rgba": "1 0 0 0.8", "type": "sphere"})

    # 14. TENDONS
    tendon_elem = ET.SubElement(mujoco, "tendon")
    colors = ["1 0 0 1", "0 1 0 1", "0 0 1 1", "1 0.5 0 1", "0.5 1 0 1", "0 0.5 1 1"]

    for s in range(num_segments):
        start_link, end_link = segment_link_ranges[s]
        nt = num_tendons_per_seg[s]

        for t in range(nt):
            tendon_name = f"seg_{s}_tendon_{t}"
            color = colors[(s * nt + t) % len(colors)]
            spatial = ET.SubElement(tendon_elem, "spatial", {"name": tendon_name, "width": "0.001", "rgba": color})

            ET.SubElement(spatial, "site", {"site": f"l_0_s{s}_s1_t{t}"})
            ET.SubElement(spatial, "site", {"site": f"l_0_s{s}_s2_t{t}"})

            for l_idx in range(start_link, end_link + 1):
                if l_idx != total_links:
                    ET.SubElement(spatial, "site", {"site": f"l_{l_idx}_s{s}_s1_t{t}"})
                ET.SubElement(spatial, "site", {"site": f"l_{l_idx}_s{s}_s2_t{t}"})

    # 15. ACTUATORS
    actuator_elem = ET.SubElement(mujoco, "actuator")
    actuator_props = config.get("actuator_properties", {})
    force_limited = actuator_props.get("tendon_forcelimited", "true")
    force_range = actuator_props.get("tendon_forcerange", "-100 0")
    pretension = float(actuator_props.get("tendon_pretension", 0.0))

    for s in range(num_segments):
        for t in range(num_tendons_per_seg[s]):
            tendon_name = f"seg_{s}_tendon_{t}"
            actuator_name = f"seg_{s}_ten_{t}"
            ET.SubElement(actuator_elem, "motor", {
                "name": actuator_name,
                "tendon": tendon_name,
                "gear": "1",
                "forcelimited": str(force_limited).lower(),
                "forcerange": force_range
            })

    # 16. KEYFRAME
    ctrl_values = [-pretension for s in range(num_segments) for _ in range(num_tendons_per_seg[s])]
    keyframe = ET.SubElement(mujoco, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "pretension", "ctrl": " ".join(f"{x:g}" for x in ctrl_values)})

    # 17. SELF-COLLISION EXCLUSIONS
    contact = ET.SubElement(mujoco, "contact")
    if config.get("disable_self_collision", True):
        link_names = [f"link_{i}" for i in range(0, total_links + 1)]
        for i in range(len(link_names)):
            for j in range(i + 1, len(link_names)):
                ET.SubElement(contact, "exclude", {"body1": link_names[i], "body2": link_names[j]})

    # 18. XML FORMAT
    xml_bytes = ET.tostring(mujoco, encoding="utf-8")
    xml_str = minidom.parseString(xml_bytes).toprettyxml(indent="    ")
    xml_str = "\n".join(line for line in xml_str.splitlines() if line.strip())

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(xml_str)

    # 19. PRINT CALCULATED PARAMETERS
    print("\n" + "=" * 70)
    print("TDCR MuJoCo XML GENERATED")
    print("=" * 70)
    print(f"Output       : {output_filename}")
    print(f"Segments     : {num_segments}")
    print(f"Total links  : {total_links}")
    print(f"Total length : {total_length:.6f} m")
    print(f"E            : {E:.6e} Pa")
    print(f"nu           : {nu:.6f}")
    print(f"G            : {G:.6e} Pa")
    print(f"Radius       : {outer_radius:.6f} m")
    print(f"Ixx = Iyy    : {Ixx:.6e} m^4")
    print(f"J            : {J:.6e} m^4")

    print("\nSEGMENT PARAMETERS")
    print("-" * 70)
    for s, props in enumerate(segment_properties):
        print(f"Segment {s + 1}")
        print(f"  Ls        = {props['length']:.6f} m")
        print(f"  N         = {props['num_links']}")
        print(f"  li        = {props['link_length']:.6f} m")
        print(f"  kx        = {props['kx']:.12g} N*m/rad")
        print(f"  ky        = {props['ky']:.12g} N*m/rad")
        print(f"  kz        = {props['kz']:.12g} N*m/rad")
        print(f"  link mass = {props['link_mass']:.12g} kg")
    print("=" * 70)
    
# ---------------------------------------------------------------------------
# 2. MuJoCo Helpers
# ---------------------------------------------------------------------------
def load_model(xml_path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = 5e-4
    model.opt.gravity[:] = 0.0

    mujoco.mj_forward(model, data)

    #print(f"MuJoCo Model Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}, ntendon={model.ntendon}")
    #print("Actuators:", [model.actuator(i).name for i in range(model.nu)])
    return model, data


def get_actuator_ids(model):
    ids = []
    # Reads only the actuators defined in the loaded XML file dynamically
    for i in range(model.nu):
        ids.append(i)
    return ids

def apply_tensions(model, data, tensions, act_ids):
    data.ctrl[:] = 0.0
    for i, act_id in enumerate(act_ids):
        data.ctrl[act_id] = -float(tensions[i])


def extract_joint_angles(model, data):
    angles = np.zeros(model.njnt)
    for j in range(model.njnt):
        angles[j] = data.qpos[model.jnt_qposadr[j]]
    return angles


def tendon_lengths(model, data):
    return np.asarray(data.ten_length[:model.ntendon], dtype=float).copy()

# ---------------------------------------------------------------------------
# 3. Interactive Keyboard Controller
# ---------------------------------------------------------------------------
class FlexibleTendonController:
    def __init__(self, num_actuators, step=0.2, maximum=10.0,tension_funcs=None):
        self.num_actuators = num_actuators
        self.tension = np.zeros(num_actuators, dtype=float)
        self.step = step
        self.maximum = maximum
        self.paused = False
        self.quit = False
        self.tension_funcs = tension_funcs

    def key_callback(self, key):
        # Segment 0 (T1, T2, T3)
        if key == ord("1") and self.num_actuators > 0:
            self.tension[0] = min(self.maximum, self.tension[0] + self.step)
        elif key == ord("q") and self.num_actuators > 0:
            self.tension[0] = max(0.0, self.tension[0] - self.step)

        elif key == ord("2") and self.num_actuators > 1:
            self.tension[1] = min(self.maximum, self.tension[1] + self.step)
        elif key == ord("w") and self.num_actuators > 1:
            self.tension[1] = max(0.0, self.tension[1] - self.step)

        elif key == ord("3") and self.num_actuators > 2:
            self.tension[2] = min(self.maximum, self.tension[2] + self.step)
        elif key == ord("e") and self.num_actuators > 2:
            self.tension[2] = max(0.0, self.tension[2] - self.step)

        # Segment 1 (T4, T5, T6 - processed only if model has >3 actuators)
        elif key == ord("4") and self.num_actuators > 3:
            self.tension[3] = min(self.maximum, self.tension[3] + self.step)
        elif key == ord("a") and self.num_actuators > 3:
            self.tension[3] = max(0.0, self.tension[3] - self.step)

        elif key == ord("5") and self.num_actuators > 4:
            self.tension[4] = min(self.maximum, self.tension[4] + self.step)
        elif key == ord("s") and self.num_actuators > 4:
            self.tension[4] = max(0.0, self.tension[4] - self.step)

        elif key == ord("6") and self.num_actuators > 5:
            self.tension[5] = min(self.maximum, self.tension[5] + self.step)
        elif key == ord("d") and self.num_actuators > 5:
            self.tension[5] = max(0.0, self.tension[5] - self.step)

        # Global Controls
        elif key == ord("r"):
            self.tension[:] = 0.0
        elif key == ord(" "):
            self.paused = not self.paused
        elif key == 27:  # ESC
            self.quit = True

    def print_status(self):
      # Segment 0 (Always present)
      s0_str = f"Seg0: [{self.tension[0]:4.1f}, {self.tension[1]:4.1f}, {self.tension[2]:4.1f}] N"
      
      # Segment 1 (Only format if the model has more than 3 actuators)
      if self.num_actuators > 3:
          s1_str = f" | Seg1: [{self.tension[3]:4.1f}, {self.tension[4]:4.1f}, {self.tension[5]:4.1f}] N"
      else:
          s1_str = ""
          
      print(f"\r{s0_str}{s1_str}", end="", flush=True)

    def update_tensions(self, current_time: float):
        """Evaluates time-varying tension functions if provided."""
        if self.tension_funcs is not None and not self.paused:
            for idx, func in enumerate(self.tension_funcs):
                if idx < self.num_actuators and callable(func):
                    # Evaluate dynamic profile and clip within bounds
                    val = float(func(current_time))
                    
                    self.tension[idx] = np.clip(val, 0, self.maximum)
                    #print(self.tension[idx])

def forward_dynamics_interactive(model, data, viewer, controller, act_ids, t_final=60.0, save_dt=0.005):
    nsteps = int(round(t_final / model.opt.timestep))
    save_every = max(1, int(round(save_dt / model.opt.timestep)))

    times, tips, lengths, tensions = [], [], [], []
    qpos_hist, qvel_hist, qacc_hist = [], [], []
    joint_angle_hist, joint_torque_hist, states_hist = [], [], []

    # 1. Pre-fetch End-Effector ID (Site first, fallback to Body)
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "force_site_tip")
    ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")

    # 2. Pre-fetch and sort all link body IDs
    link_body_ids = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and body_name.startswith("link_"):
            link_body_ids.append(i)

    # Sort numerically (link_0, link_1, ..., link_N)
    link_body_ids.sort(
        key=lambda b_id: int(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id).split("_")[1])
    )

    step_count = 0

    while viewer.is_running() and data.time < t_final:
        if controller.quit:
            break

        if not controller.paused:
            # Update time-dependent dynamic tension profiles & apply
            controller.update_tensions(data.time)

            for idx, act_id in enumerate(act_ids):
                if idx < controller.num_actuators:
                    data.ctrl[act_id] = -controller.tension[idx]

            # Step MuJoCo Physics
            mujoco.mj_step(model, data)

            # Save data matching save_dt resolution
            if step_count % save_every == 0:
                # Track End-Effector Tip Position
                if ee_site_id != -1:
                    tip_pos = data.site_xpos[ee_site_id].copy()
                elif ee_body_id != -1:
                    tip_pos = data.xpos[ee_body_id].copy()
                else:
                    # Fallback to the last link position if tip isn't named
                    tip_pos = data.xpos[link_body_ids[-1]].copy() if link_body_ids else np.zeros(3)

                times.append(data.time)
                tips.append(tip_pos)
                tensions.append(controller.tension.copy())

                # Optional helpers
                if "tendon_lengths" in globals():
                    lengths.append(tendon_lengths(model, data))
                if "extract_joint_angles" in globals():
                    joint_angle_hist.append(extract_joint_angles(model, data))

                qpos_hist.append(data.qpos.copy())
                qvel_hist.append(data.qvel.copy())
                qacc_hist.append(data.qacc.copy())
                joint_torque_hist.append(data.qfrc_actuator.copy())

                # Extract backbone curve: Links + End-Effector Tip as final point
                link_positions = [data.xpos[b_id].copy() for b_id in link_body_ids]
                link_positions.append(tip_pos)  # Adds tip for tracking
                states_hist.append(np.array(link_positions))

            step_count += 1

        viewer.sync()

    # Format output dictionary (states shape: [TimeSteps, NumLinks + 1, 3])
    return {
        "time": np.asarray(times),
        "tip": np.asarray(tips),
        "tendon_length": np.asarray(lengths),
        "tension": np.asarray(tensions),
        "qpos": np.asarray(qpos_hist),
        "qvel": np.asarray(qvel_hist),
        "qacc": np.asarray(qacc_hist),
        "joint_angles": np.asarray(joint_angle_hist),
        "joint_torques": np.asarray(joint_torque_hist),
        "states": np.asarray(states_hist),
    }