import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.interpolate import griddata, CubicSpline
import os, sys, time
import uproot3 as uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numbers import Integral
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy.stats import norm, truncnorm, foldnorm
import warnings
from time import perf_counter
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from xgboost import XGBRegressor
from sklearn.preprocessing import QuantileTransformer

plt.style.use("default")

rng = np.random.default_rng(12345)
SIPM_GEOMETRIC_EFFICIENCY = 0.85
SIPM_CELL_SIZE = 3.36
plane_angles = np.array(
    [
        1.745319668152660597e-01,
        4.072425337478433605e-01,
        6.399531006804206612e-01,
        8.726636676129979620e-01,
        1.105374234545575263e00,
        1.338084801478152563e00,
        1.570795368410729864e00,
        1.803505935343307165e00,
        2.036216502275884466e00,
        2.268927069208461766e00,
        2.501637636141039067e00,
        2.734348203073616368e00,
        2.967058770006193225e00,
        3.199769336938770525e00,
        3.432479903871347826e00,
        3.665190470803925127e00,
        3.897901037736502428e00,
        4.130611604669079284e00,
        4.363322171601657473e00,
        4.596032738534233886e00,
        4.828743305466812075e00,
        5.061453872399388487e00,
        5.294164439331966676e00,
        5.526875006264543089e00,
        5.759585573197120389e00,
        5.992296140129697690e00,
        6.225006707062274991e00,
    ]
)
norm_r = 1007.0091186826339


def plot_cyl(file, coords=None, transposed=False):
    def cylinder(r, h, a=0, nt=100, nv=50):
        """
        parametrize the cylinder of radius r, height h, base point a
        """
        theta = np.linspace(0, 2 * np.pi, nt)
        v = np.linspace(a, a + h, nv)
        theta, v = np.meshgrid(theta, v)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = v
        return x, y, z

    colorscale = [[0, "blue"], [1, "blue"]]

    x1, y1, z1 = cylinder(1000, 2800, a=-1400)
    if transposed:
        z1, x1 = x1, z1
    cyl1 = go.Surface(
        x=x1, y=y1, z=z1, colorscale=colorscale, showscale=False, opacity=0.05
    )

    fig = go.Figure()

    # colors = [f'rgba({255}, {int(255 * i / 9)}, 0, 0.6)' for i in range(10)]  # Adjust the colors as needed
    colors = [
        "rgba(255, 0, 0, 0.6)",  # Red
        "rgba(255, 165, 0, 0.6)",  # Orange
        "rgba(255, 255, 0, 0.6)",  # Yellow
        "rgba(0, 255, 0, 0.6)",  # Green
        "rgba(0, 0, 255, 0.6)",  # Blue
        "rgba(75, 0, 130, 0.6)",  # Indigo
        "rgba(128, 0, 128, 0.6)",  # Violet
        "rgba(255, 99, 71, 0.6)",  # Tomato
        "rgba(0, 128, 128, 0.6)",  # Teal
        "rgba(128, 128, 0, 0.6)",  # Olive
    ]
    for num in range(10):
        if coords is not None:
            x = coords[num][0]
            y = coords[num][1]
            z = coords[num][2]
        else:
            x = np.array(
                file["events;1"]["FarichBarrelG4Hits.postStepPosition.x"].array()
            )[num]
            y = np.array(
                file["events;1"]["FarichBarrelG4Hits.postStepPosition.y"].array()
            )[num]
            z = np.array(
                file["events;1"]["FarichBarrelG4Hits.postStepPosition.z"].array()
            )[num]
        if transposed:
            z, x = x, z
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=1, color=colors[num]),
                name=f"Ring {num+1}",
            )
        )

        x = np.array(file["events;1"]["DriftChamberG4Hits.postStepPosition.x"].array())[
            num
        ]
        y = np.array(file["events;1"]["DriftChamberG4Hits.postStepPosition.y"].array())[
            num
        ]
        z = np.array(file["events;1"]["DriftChamberG4Hits.postStepPosition.z"].array())[
            num
        ]
        if transposed:
            z, x = x, z
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=1, color=colors[num]),
                name=f"Drift Chamber {num + 1}",
                line=dict(dash="dash"),
            )
        )

        # x = np.array(file['events;1']['SiStripG4Hits.postStepPosition.x'].array())[num]
        # y = np.array(file['events;1']['SiStripG4Hits.postStepPosition.y'].array())[num]
        # z = np.array(file['events;1']['SiStripG4Hits.postStepPosition.z'].array())[num]
        # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num]), name=f'Si Strips {num + 1}', line=dict(dash='solid')))

        x = np.array(file["events;1"]["TPCG4Hits.postStepPosition.x"].array())[num]
        y = np.array(file["events;1"]["TPCG4Hits.postStepPosition.y"].array())[num]
        z = np.array(file["events;1"]["TPCG4Hits.postStepPosition.z"].array())[num]
        if transposed:
            z, x = x, z
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=1, color=colors[num]),
                name=f"TPC {num + 1}",
                line=dict(dash="solid"),
            )
        )

    fig.add_trace(cyl1)
    # Update the layout to add labels and a title
    scene = dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        aspectmode="cube",
        xaxis=dict(range=[-1005, 1005]),
        yaxis=dict(range=[-1005, 1005]),
        zaxis=dict(range=[-1405, 1405]),
    )
    if transposed:
        scene = dict(
            zaxis_title="X Axis",
            yaxis_title="Y Axis",
            xaxis_title="Z Axis",
            aspectmode="cube",
            zaxis=dict(range=[-1005, 1005]),
            yaxis=dict(range=[-1005, 1005]),
            xaxis=dict(range=[-1405, 1405]),
        )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        scene=scene,
        template="plotly_dark",  # DARK OR LIGHT
        title="3D Scatter Plot",
    )

    # Show the plot
    fig.show()


def plot_event(
    ev_coords, track_coords, tpc_coords, true_direction_coords, transposed=False, num=0
):
    def cylinder(r, h, a=0, nt=100, nv=50):
        """
        parametrize the cylinder of radius r, height h, base point a
        """
        theta = np.linspace(0, 2 * np.pi, nt)
        v = np.linspace(a, a + h, nv)
        theta, v = np.meshgrid(theta, v)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = v
        return x, y, z

    colorscale = [[0, "blue"], [1, "blue"]]

    x1, y1, z1 = cylinder(1000, 2800, a=-1400)
    if transposed:
        z1, x1 = x1, z1
    cyl1 = go.Surface(
        x=x1, y=y1, z=z1, colorscale=colorscale, showscale=False, opacity=0.05
    )

    fig = go.Figure()

    # colors = [f'rgba({255}, {int(255 * i / 9)}, 0, 0.6)' for i in range(10)]  # Adjust the colors as needed
    colors = [
        "rgba(255, 0, 0, 0.6)",  # Red
        "rgba(255, 165, 0, 0.6)",  # Orange
        "rgba(255, 255, 0, 0.6)",  # Yellow
        "rgba(0, 255, 0, 0.6)",  # Green
        "rgba(0, 0, 255, 0.6)",  # Blue
        "rgba(75, 0, 130, 0.6)",  # Indigo
        "rgba(128, 0, 128, 0.6)",  # Violet
        "rgba(255, 99, 71, 0.6)",  # Tomato
        "rgba(0, 128, 128, 0.6)",  # Teal
        "rgba(128, 128, 0, 0.6)",  # Olive
    ]
    x = ev_coords[0]
    y = ev_coords[1]
    z = ev_coords[2]

    if transposed:
        z, x = x, z
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=1, color=colors[num]),
            name=f"Ring {num+1}",
        )
    )

    if track_coords is not None:
        x = track_coords[0]
        y = track_coords[1]
        z = track_coords[2]
        if transposed:
            z, x = x, z
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=1, color=colors[num + 1]),
                name=f"Drift Chamber {num + 1}",
                line=dict(dash="dash"),
            )
        )

    if tpc_coords is not None:
        x = tpc_coords[0]
        y = tpc_coords[1]
        z = tpc_coords[2]
        if transposed:
            z, x = x, z
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=1, color=colors[num + 2]),
                name=f"TPC {num + 1}",
                line=dict(dash="solid"),
            )
        )

    if true_direction_coords is not None:
        x = true_direction_coords[0]
        y = true_direction_coords[1]
        z = true_direction_coords[2]
        if transposed:
            z, x = x, z
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                marker=dict(size=1, color=colors[num + 3]),
                name=f"True Direction {num + 1}",
                line=dict(dash="solid"),
            )
        )

    fig.add_trace(cyl1)
    # Update the layout to add labels and a title
    scene = dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        aspectmode="cube",
        xaxis=dict(range=[-1005, 1005]),
        yaxis=dict(range=[-1005, 1005]),
        zaxis=dict(range=[-1405, 1405]),
    )
    if transposed:
        scene = dict(
            zaxis_title="X Axis",
            yaxis_title="Y Axis",
            xaxis_title="Z Axis",
            aspectmode="cube",
            zaxis=dict(range=[-1005, 1005]),
            yaxis=dict(range=[-1005, 1005]),
            xaxis=dict(range=[-1405, 1405]),
        )
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        scene=scene,
        template="plotly_dark",  # DARK OR LIGHT
        title="3D Scatter Plot",
    )

    # Show the plot
    fig.show()


def sipm_sim(full_coords, sipm_eff):
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        probs = (
            np.array([sipm_eff[wv] for wv in event_coords[3]])
            * SIPM_GEOMETRIC_EFFICIENCY
        )
        random_nums = np.random.random(len(probs))
        idx = np.where(probs - random_nums > 0)[0]
        for j in range(full_coords.shape[1]):
            full_coords[i][j] = event_coords[j][idx]


def lin_move_to_grid(coord, grid):
    distances = cdist(coord[:, np.newaxis], grid[:, np.newaxis], metric="euclidean")
    closest_indices = np.argmin(distances, axis=1)
    # Use the indices to get the closest elements from second_array
    closest_points = grid[closest_indices]
    return closest_points


def rotate_point(point, angle):
    # x, y, _, __ = point
    x = point[0]
    y = point[1]
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    new_x = x * cos_theta - y * sin_theta
    # new_y = x * sin_theta + y * cos_theta
    return new_x


def shift_index_for_event_rotation(idx):
    idx = round(idx)
    if idx > 20:
        return idx - 27
    elif idx < -20:
        return idx + 27
    else:
        return idx


def rotate_event(coords, main_angle):
    angles = np.arctan2(coords[1], coords[0]) % (2 * np.pi)
    angles = lin_move_to_grid(angles, plane_angles)
    idx_to_shift = (angles - main_angle) / 0.2327
    idx_to_shift = np.array(
        [round(shift_index_for_event_rotation(idx)) for idx in idx_to_shift]
    )
    angle_to_rotate = np.pi / 2 - angles
    x = rotate_point(coords, angle_to_rotate) - 2 * idx_to_shift * norm_r * np.sin(
        np.pi / 27
    )
    return np.column_stack((x, np.zeros_like(x) + 1000, coords[2], coords[3]))


def rotate_events(full_coords, main_angles):
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        rotated_event_coords = rotate_event(event_coords, main_angles[i])

        for j in range(4):
            full_coords[i][j] = rotated_event_coords[:, j]


def move_points_to_grid(points, grid_points):
    x_moved = lin_move_to_grid(points[0], grid_points[0])
    z_moved = lin_move_to_grid(points[2], grid_points[1])
    # wv_moved = lin_move_to_grid(points[:,3], grid_points[2])

    return np.column_stack((x_moved, points[1], z_moved, points[3], points[4]))


def remove_duplicate_points(points):
    # Find unique pairs of elements in both arrays
    unique_pairs = set(zip(points[0], points[2]))

    # Find indices of elements forming unique pairs
    indices_to_remove = []
    for pair in unique_pairs:
        indices = np.where((points[0] == pair[0]) & (points[2] == pair[1]))[0]
        if len(indices) > 1:  # If the pair occurs more than once
            indices_to_remove.extend(indices[1:])

    return np.column_stack(
        (
            np.delete(points[0], indices_to_remove),
            np.delete(points[1], indices_to_remove),
            np.delete(points[2], indices_to_remove),
            np.delete(points[3], indices_to_remove),
            np.delete(points[4], indices_to_remove),
        )
    )


def move_events_to_grid(full_coords, grid_points):
    for i in range(full_coords.shape[0]):

        event_coords = full_coords[i]

        grid_event_coords = move_points_to_grid(event_coords, grid_points)
        # print(grid_event_coords.shape)
        grid_set_event_coords = remove_duplicate_points(grid_event_coords.T)

        for j in range(full_coords.shape[1]):
            full_coords[i][j] = grid_set_event_coords[:, j]


def fix_PDE_plot(PDEs, PDE_wvs):
    lin_reg_tmp = LinearRegression()
    lin_reg_tmp.fit([[PDE_wvs[18]], [PDE_wvs[28]]], ([PDEs[18]], [PDEs[28]]))
    PDEs[18:28] = PDE_wvs[18:28] * lin_reg_tmp.coef_[0][0] + lin_reg_tmp.intercept_[0]
    return {key: value for key, value in zip(PDE_wvs, PDEs)}


def rotate_lines(full_coords):
    angles = np.zeros(full_coords.shape[0])
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        rotated_event_coords, angles[i] = rotate_line(event_coords)

        for j in range(3):
            full_coords[i][j] = rotated_event_coords[:, j]
    return angles


def rotate_line(coords):
    angles = np.arctan2(coords[1], coords[0]) % (2 * np.pi)
    # print(angles)
    median_angle = np.median(angles)
    median_angle = lin_move_to_grid(np.array([median_angle]), plane_angles)
    # print(angles)
    angle_to_rotate = np.pi / 2 - median_angle
    # print(angle_to_rotate)
    x, y = rotate_point_on_line(coords, angle_to_rotate)
    return np.column_stack((x, y, coords[2])), median_angle


def rotate_point_on_line(point, angle):
    x, y, _ = point
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta
    return new_x, new_y


def find_intersections(full_coords):
    intersections = np.zeros((full_coords.shape[0], 3))
    zeros = np.zeros((1, 3))
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        pca = PCA(n_components=1)

        if event_coords[0].shape[0] == 1:
            event_coords = [np.insert(arr, 0, 0.0) for arr in event_coords]

        pca.fit(np.column_stack(event_coords))
        line_direction = pca.components_[0]
        line_point = pca.mean_

        # Calculate the parameter t for the intersection with the plane y=1000
        p_y = line_point[1]
        d_y = line_direction[1]
        t = (1000 - p_y) / d_y

        # Find the intersection point
        intersection_point = line_point + t * line_direction

        # print(f"Line direction: {line_direction}")
        # print(f"Point on the line: {line_point}")
        # print(f"Intersection point with the plane y=1000: {intersection_point}")
        for j in range(3):
            intersections[i][j] = intersection_point[j]
    return intersections


def applySpaceCut(edf: pd.DataFrame) -> pd.DataFrame:
    return edf[
        (abs(edf["x_c"] - edf["x_i"]) <= 220) & (abs(edf["y_c"] - edf["y_i"]) <= 220)
    ]


def planeRecalculation(edf: pd.DataFrame, idf: pd.DataFrame):
    R = edf[["x_c", "y_c", "z_c"]].to_numpy()
    R_i = edf[["x_i", "y_i", "z_c"]].to_numpy()
    N = edf[["nx_p", "ny_p", "nz_p"]].to_numpy()
    dist = idf.W / 2 + idf.zdis
    alpha = (float(dist)) / N[:, 2]
    r_d = N * alpha[:, np.newaxis]

    u = R - r_d
    dot = np.sum(N * u, axis=1)
    w = r_d - R_i
    fac = -np.sum(N * w, axis=1) / dot
    u *= fac[:, np.newaxis]

    R_new = r_d + u

    speedOfLight_mmperns = 299.792458
    t_dif = np.sqrt(np.sum((R_new - R) ** 2, axis=1)) / speedOfLight_mmperns
    edf["t_c"] = edf["t_c"] + np.sign(R_new[:, 2] - R[:, 2]) * t_dif

    edf["recalculated_x"] = R_new[:, 0]
    edf["recalculated_y"] = R_new[:, 1]
    edf["recalculated_z"] = R_new[:, 2]


def planeRotation(edf: pd.DataFrame):
    R = edf[["recalculated_x", "recalculated_y", "recalculated_z"]].to_numpy()
    R_i = edf[["x_i", "y_i", "z_c"]].to_numpy()
    N = edf[["nx_p", "ny_p", "nz_p"]].to_numpy()  # N
    M = np.array([0, 0, 1])  # M
    c = np.dot(N, M) / (np.linalg.norm(M) * np.linalg.norm(N, axis=1))
    axis = np.cross(N, np.broadcast_to(M, (N.shape[0], 3))) / np.linalg.norm(
        np.cross(N, np.broadcast_to(M, (N.shape[0], 3))), axis=1, keepdims=True
    )
    x, y, z = axis.T
    s = np.sqrt(1 - c * c)
    C = 1 - c
    rmat = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )
    # print(rmat.shape)
    # print(R.shape)
    # print(rmat[:, :, 0])
    # print(R[0])
    # print(rmat[:, :, 0] @ R[0])
    rotated_R = np.matmul(rmat.transpose((2, 0, 1)), R[:, :, np.newaxis])
    rotated_R = np.squeeze(rotated_R, axis=-1).transpose().T
    rotated_R_i = np.matmul(rmat.transpose((2, 0, 1)), R_i[:, :, np.newaxis])
    rotated_R_i = np.squeeze(rotated_R_i, axis=-1).transpose().T
    # print(rotated_R[0])
    maskR = np.logical_or(abs(rotated_R[:, 0]) >= 5000, abs(rotated_R[:, 1]) >= 5000)
    maskR_i = np.logical_or(
        abs(rotated_R_i[:, 0]) >= 5000, abs(rotated_R_i[:, 1]) >= 5000
    )
    rotated_R[maskR] = [5000, 5000, 0]
    rotated_R_i[maskR_i] = [5000, 5000, 0]
    rotated_n = (rotated_R_i - edf[["x_p", "y_p", "z_p"]].to_numpy()) / np.linalg.norm(
        rotated_R_i - edf[["x_p", "y_p", "z_p"]].to_numpy(), axis=1, keepdims=True
    )
    edf["rotated_x"] = rotated_R[:, 0]
    edf["rotated_y"] = rotated_R[:, 1]
    edf["rotated_z"] = rotated_R[:, 2]
    edf["rotated_x_i"] = rotated_R_i[:, 0]
    edf["rotated_y_i"] = rotated_R_i[:, 1]
    edf["rotated_z_i"] = rotated_R_i[:, 2]
    edf["rotated_nx_p"] = rotated_n[:, 0]
    edf["rotated_ny_p"] = rotated_n[:, 1]
    edf["rotated_nz_p"] = rotated_n[:, 2]


def applySecondSpaceCut(edf: pd.DataFrame) -> pd.DataFrame:
    return edf[
        (abs(edf["rotated_x"] - edf["rotated_x_i"]) <= 220)
        & (abs(edf["rotated_y"] - edf["rotated_y_i"]) <= 220)
    ]


def edf_to_bdf(edf_col: pd.Series, bdf: pd.DataFrame):
    to_bdf = [sub.iloc[0] for _, sub in edf_col.groupby(level=0)]
    bdf[edf_col.name] = pd.Series(to_bdf)


def primaryDirectionRecalculation(edf: pd.DataFrame):
    N = edf.loc[:, ("nx_p", "ny_p", "nz_p")].to_numpy()
    M = []
    theta_ps = []
    for n in N:
        M.append([0, 0, 1])
        dot_product = np.dot(n, [0, 0, 1.0])

        # Calculate the magnitudes (norms)
        mag_vector = np.linalg.norm(n)
        mag_z_axis = np.linalg.norm([0, 0, 1.0])

        # Calculate the cosine of the angle
        cos_theta = dot_product / (mag_vector * mag_z_axis)

        # Handle possible numerical issues with floating point precision
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Calculate the angle in radians
        theta_radians = np.arccos(cos_theta)
        theta_ps.append(theta_radians)
        # print(n)
        # print(C_inv)
        # print(C_inv @ n)
        # break
    M = np.array(M)
    theta_ps = np.array(theta_ps)
    edf["recalculated_nx_p"] = M[:, 0]
    edf["recalculated_ny_p"] = M[:, 1]
    edf["recalculated_nz_p"] = M[:, 2]
    edf["theta_p"] = theta_ps
    edf["rotated_r_c"] = np.sqrt(
        (edf["rotated_x_i"] - edf["rotated_x"]) ** 2
        + (edf["rotated_y_i"] - edf["rotated_y"]) ** 2
    )


def recoAngles(
    edf: pd.DataFrame, idf: pd.DataFrame, rotation_mode=False, for_decay=True
):
    """
    Геометрическая реконструкция углов фотонов относительно направления частицы.
    Из координат срабатываний и частиц вычисляются углы theta_c, phi_c и время вылета фотонов t_c_orig и добавляются к edf.
    Параметр for_decay для использования уже существующей точки пересечения трека вместо её вычисления
    """
    r0 = edf.loc[:, ("x_p", "y_p", "z_p")].to_numpy()
    if rotation_mode:
        r = edf.loc[:, ("rotated_x", "rotated_y", "rotated_z")].to_numpy()
        # n0 = edf.loc[:, ('rotated_nx_p', 'rotated_ny_p', 'rotated_nz_p')].to_numpy()
        n0 = edf.loc[
            :, ("recalculated_nx_p", "recalculated_ny_p", "recalculated_nz_p")
        ].to_numpy()
    else:
        r = edf.loc[:, ("x_c", "y_c", "z_c")].to_numpy()
        n0 = edf.loc[:, ("nx_p", "ny_p", "nz_p")].to_numpy()

    speedOfLight_mmperns = 299.792458  # мм/нс

    # расстояние от радиатора до детектора
    dist = float(idf["distance"])

    # толщина радиатора
    W = float(idf["W"])

    # расстояние от точки вылета частицы до входной плоскости радиатора
    rad_pos = float(idf["zdis"])

    # полное число срабатываний
    N = edf.shape[0]

    # координаты точки пересечения трека с ФД
    if not rotation_mode:
        if for_decay:
            x_i = edf.x_i
            y_i = edf.y_i
        else:
            y_i = (
                r0[:, 1] + (dist + rad_pos) * n0[:, 1] / n0[:, 2]
            )  # r0[:,1] + (dist + W + rad_pos) * n0[:,1] / n0[:,2]   #   r0[:,1] + (dist + rad_pos) * n0[:,1] / n0[:,2]
            x_i = (
                r0[:, 0] + (y_i - r0[:, 1]) * n0[:, 0] / n0[:, 1]
            )  # r0[:,0] + (y_i - r0[:,1]) * n0[:,0] / n0[:,1]    #     r0[:,0] + (dist + rad_pos) * n0[:,0] / n0[:,2]
            edf["x_i"] = x_i
            edf["y_i"] = y_i

        edf["r_p_c"] = np.sqrt(
            (r0[:, 0] - x_i) ** 2 + (r0[:, 1] - y_i) ** 2 + (r0[:, 2] - r[:, 2]) ** 2
        )
        edf["r_c"] = np.sqrt((x_i - edf["x_c"]) ** 2 + (y_i - edf["y_c"]) ** 2)

    if rotation_mode:
        n_mean = float(idf["n_mean"])

        edf["rotated_r_c"] = np.sqrt(
            (edf["rotated_x_i"] - edf["rotated_x"]) ** 2
            + (edf["rotated_y_i"] - edf["rotated_y"]) ** 2
        )

        rotated_r_c = edf["rotated_r_c"].to_numpy()
        # r_p_c = edf['r_p_c'].to_numpy()
        beta = edf["beta"].to_numpy()
        r_p_c = dist  # or + W/2 ???

    # avg_betas = []
    # for _, subentry in edf['beta_from_true_r'].groupby(level=0):
    #   avg_beta = subentry.mean()
    #   for __ in subentry:
    #     avg_betas.append(avg_beta)
    # edf['beta_from_true_r_mean'] = avg_betas
    # косинусы и синусы сферических углов направления частицы
    costheta, sintheta = n0[:, 2], np.sqrt(n0[:, 0] ** 2 + n0[:, 1] ** 2)
    phi = np.arctan2(n0[:, 1], n0[:, 0])
    cosphi, sinphi = np.cos(phi), np.sin(phi)

    # номинальная точка вылета фотонов
    ro = r0 + (W / 2 + rad_pos) / n0[:, 2].reshape(N, 1) * n0

    """
    Преобразование в СК частицы
    𝑢𝑥 = cos 𝜃(𝑣𝑥 cos 𝜙 + 𝑣𝑦 sin 𝜙) − 𝑣𝑧 sin 𝜃,
    𝑢𝑦 = −𝑣𝑥 sin 𝜙 + 𝑣𝑦 cos 𝜙,
    𝑢𝑧 = sin 𝜃(𝑣𝑥 cos 𝜙 + 𝑣𝑦 sin 𝜙) + 𝑣𝑧 cos 𝜃.
    """

    # вектор направления фотона в лабораторной СК
    s = r - ro
    snorm = np.linalg.norm(s, axis=1, keepdims=True)
    v = s / snorm
    if not rotation_mode:
        edf["t_c_orig"] = edf["t_c"] - (snorm / speedOfLight_mmperns).reshape(N)

    # освобождение памяти при необходимости
    # del r0, n0, ro, r, s

    U = np.stack(
        (
            np.stack((costheta * cosphi, costheta * sinphi, -sintheta)),
            np.stack((-sinphi, cosphi, np.full(N, 0.0))),
            np.stack((sintheta * cosphi, sintheta * sinphi, costheta)),
        )
    ).transpose(2, 0, 1)

    # единичный вектор направления фотона в СК частицы
    u = (U @ v.reshape(N, 3, 1)).reshape(N, 3)

    # сферические углы фотона в СК частицы
    if rotation_mode:
        # edf["rotated_theta_c"] = np.arccos(u[:, 2])
        edf["rotated_phi_c"] = np.arctan2(-u[:, 1], -u[:, 0])
    else:
        edf["theta_c"] = np.arccos(u[:, 2])
        edf["phi_c"] = np.arctan2(-u[:, 1], -u[:, 0])


def local_sum_2d(
    event,
    r_slices,
    t_slices,
    square_counts,
    max_index,
    n,
    m,
    timestep,
    t_window_width,
    method="N/r",
):
    cut_event = event[
        (
            event.t_c
            <= np.clip(t_slices[max_index[1]] + t_window_width + timestep * m, 0, 10)
        )
        & (event.t_c >= np.clip(t_slices[max_index[1]] - timestep * m, 0, 10))
        & (event.rotated_r_c <= r_slices[max_index[0] + n])
        & (event.rotated_r_c >= r_slices[max_index[0] - n])
    ]
    return np.mean(cut_event.rotated_r_c)


def local_weighed_sum_2d(
    r_slices, t_slices, square_counts, max_index, n, m, method="N/r"
):
    arr = np.mean(
        square_counts[
            max_index[0] - n : max_index[0] + n + 1,
            np.clip(max_index[1] - m, 0, 50) : np.clip(max_index[1] + m + 1, 0, 50),
        ],
        axis=1,
    )
    if method == "N/r":
        sum_arr = r_slices[max_index[0] - n : max_index[0] + n + 1] ** 2 * arr
        den_arr = r_slices[max_index[0] - n : max_index[0] + n + 1] * arr
    else:
        sum_arr = r_slices[max_index[0] - n : max_index[0] + n + 1] * arr
        den_arr = arr

    weighted_sum = np.sum(sum_arr)
    weighted_den = np.sum(den_arr)

    return weighted_sum / weighted_den


def local_weighed_sum(r_slices, counts, max_index, n, method="N/r"):
    if method == "N/r":
        sum_arr = (
            r_slices[max_index - n : max_index + n + 1] ** 2
            * counts[max_index - n : max_index + n + 1]
        )
        den_arr = (
            r_slices[max_index - n : max_index + n + 1]
            * counts[max_index - n : max_index + n + 1]
        )
    else:
        sum_arr = (
            r_slices[max_index - n : max_index + n + 1]
            * counts[max_index - n : max_index + n + 1]
        )
        den_arr = counts[max_index - n : max_index + n + 1]

    weighted_sum = np.sum(sum_arr)
    weighted_den = np.sum(den_arr)

    return weighted_sum / weighted_den


def pol(x, a, b, c):
    return a * np.exp(((x - b) ** 2) / c**2)


def pol2(x, p0, p1, p2):
    return p0 + p1 * x + p2 * x**2


def d3pol2(X, p0, p1, p2, q0, q1, q2, k0, k1, k2):
    r, theta = X
    # return pol(r, pol2(theta, p0, p1, p2), pol2(theta, q0, q1, q2), pol2(theta, k0, k1, k2))
    return pol(
        r,
        p0 + p1 * theta + p2 * theta**2,
        q0 + q1 * theta + q2 * theta**2,
        k0 + k1 * theta + k2 * theta**2,
    )


def momentum_from_beta(beta, mass):
    return mass * beta / np.sqrt(1 - beta**2)


def momentum_pol(x, a, b, c):
    return momentum_from_beta(pol(x, a, b, c), 139.57)


def momentum_d3pol2(X, p0, p1, p2, q0, q1, q2, k0, k1, k2):
    return momentum_from_beta(d3pol2(X, p0, p1, p2, q0, q1, q2, k0, k1, k2), 139.57)


def rSlidingWindowIntro(
    edf: pd.DataFrame,
    idf: pd.DataFrame,
    bdf: pd.DataFrame,
    avg_sigmas: tuple,
    avg_t_sigmas: tuple,
    step: float,
    method="N/r",
    cal_arr=False,
    t_window_width=2,
    r_width_factor=2,
    t_width_factor=8,
    full_width_t_hist=False,
    num_of_groups=5,
    what_to_group="beta",
):
    r_r_c = edf["rotated_r_c"]
    time_step = float(t_window_width) / t_width_factor

    # Compute beta_step and r_step using NumPy functions
    param_step = np.ptp(edf[what_to_group].values)  # не факт что нужно values

    # Compute param_group_to_bdf and  using NumPy operations
    param_group = np.floor(
        (
            num_of_groups * edf[what_to_group]
            + max(edf[what_to_group])
            - (num_of_groups + 1) * min(edf[what_to_group])
        )
        / param_step
    ).values

    edf["param_group"] = param_group

    edf_to_bdf(edf.param_group, bdf)

    edf_to_bdf(edf.theta_p, bdf)
    bdf["cos_theta_p"] = np.cos(bdf["theta_p"])
    # edf_to_bdf(edf.signal_counts, bdf)
    # edf_to_bdf(edf.beta, bdf)


def calculateSignalCounts(edf: pd.DataFrame, bdf: pd.DataFrame):
    signal_counts = edf["signal"].groupby(level=0).sum()
    bdf["signal_counts"] = signal_counts.values
    edf["signal_counts"] = edf.signal.groupby(level=0).transform("sum").values


def rSlidingWindowLoop1(
    edf: pd.DataFrame,
    idf: pd.DataFrame,
    bdf: pd.DataFrame,
    avg_sigmas: tuple,
    avg_t_sigmas: tuple,
    step: float,
    method="N/r",
    cal_arr=False,
    t_window_width=2,
    r_width_factor=2,
    t_width_factor=8,
    full_width_t_hist=True,
    weighed=True,
):
    mean_cos_theta_p = 0.8535536229610443
    param_step = step
    step = param_step / r_width_factor
    time_step = float(t_window_width) / t_width_factor

    r_slices = np.arange(0, 800, step=step)
    t_slices = np.arange(0, 15, step=time_step)

    n_sigmas = np.ptp(avg_sigmas)
    t_sigmas = np.ptp(avg_t_sigmas)

    all_counts_to_edf = np.zeros((n_sigmas, len(edf)))
    all_calculated_r = np.zeros((n_sigmas, len(edf)))
    all_calculated_r_from_2d = np.zeros((t_sigmas, n_sigmas, len(edf)))
    cur_ind = 0
    for i, (entry, subentry) in enumerate(
        edf[["rotated_r_c", "t_c", "rotated_phi_c", "theta_p"]].groupby(level=0)
    ):
        if np.cos(subentry.theta_p).iat[0] >= mean_cos_theta_p:
            step = param_step / (r_width_factor + 1)
        else:
            step = param_step / r_width_factor

        counts = np.zeros(r_slices.shape)
        square_counts = np.zeros(shape=(r_slices.shape[0], t_slices.shape[0]))

        mask = np.logical_and(subentry.rotated_r_c >= 16, subentry.rotated_r_c <= 80)
        rotated_r_c = subentry.rotated_r_c[mask]
        t_c = subentry.t_c[mask]
        rotated_phi_c = subentry.rotated_phi_c[mask]

        counts, _ = np.histogram(rotated_r_c, bins=r_slices)
        square_counts, _, __ = np.histogram2d(
            rotated_r_c, t_c, bins=(r_slices, t_slices)
        )

        if method == "N/r":
            counts = np.divide(np.add(counts[:-1], counts[1:]), r_slices[1:-1])
            shift = 0
            square_counts[:-1, :] = np.divide(
                np.add(square_counts[:-1, :], square_counts[1:, :]),
                r_slices[1 : -1 - shift * 2, np.newaxis],
            )

        if full_width_t_hist:
            square_counts_but_last = sum(
                [
                    square_counts[:, it : -t_width_factor + 1 + it]
                    for it in range(t_width_factor - 1)
                ]
            )
            square_counts = np.add(
                square_counts_but_last, square_counts[:, t_width_factor - 1 :]
            )

        max_index = np.argmax(counts)

        max_index_2d = np.unravel_index(np.argmax(square_counts), square_counts.shape)

        for j in range(n_sigmas):
            all_counts_to_edf[j][cur_ind : subentry.shape[0] + cur_ind] = counts[
                np.floor_divide(subentry.rotated_r_c, step).astype(int)
            ]  # fixed
            # avg_r_from_slices = local_weighed_sum(r_slices, counts, max_index, j + avg_sigmas[0], method)
            # all_calculated_r[j, cur_ind:subentry.shape[0] + cur_ind] = np.repeat(avg_r_from_slices, subentry.shape[0])
            for t in range(t_sigmas):
                if weighed:
                    avg_r_from_2d_slices = local_weighed_sum_2d(
                        r_slices,
                        t_slices,
                        square_counts,
                        max_index_2d,
                        j + avg_sigmas[0],
                        t + avg_t_sigmas[0],
                    )
                else:
                    avg_r_from_2d_slices = local_sum_2d(
                        subentry,
                        r_slices,
                        t_slices,
                        square_counts,
                        max_index_2d,
                        j + avg_sigmas[0],
                        t + avg_t_sigmas[0],
                        t_window_width=t_window_width,
                        timestep=time_step,
                    )

                all_calculated_r_from_2d[
                    t, j, cur_ind : subentry.shape[0] + cur_ind
                ] = np.repeat(avg_r_from_2d_slices, subentry.shape[0])

        cur_ind += subentry.shape[0]
    for j in range(n_sigmas):
        edf[f"slice_counts_{j + avg_sigmas[0]}_sigms"] = all_counts_to_edf[j]
        # edf[f'unfixed_calculated_r_{j + avg_sigmas[0]}_sigms'] = all_calculated_r[j, :]
        for t in range(t_sigmas):
            edf[
                f"unfixed_calculated_r_2d_{j + avg_sigmas[0]}_rsigms_{t + avg_t_sigmas[0]}_tsigms"
            ] = all_calculated_r_from_2d[t, j, :]
            edf_to_bdf(
                edf[
                    f"unfixed_calculated_r_2d_{j + avg_sigmas[0]}_rsigms_{t + avg_t_sigmas[0]}_tsigms"
                ],
                bdf,
            )


def rSlidingWindowLoop2(
    edf: pd.DataFrame,
    idf: pd.DataFrame,
    bdf: pd.DataFrame,
    avg_sigmas: tuple,
    avg_t_sigmas: tuple,
    step: float,
    method="N/r",
    cal_arr=False,
    t_window_width=2,
    r_width_factor=2,
    t_width_factor=8,
    full_width_t_hist=False,
    param_fit=False,
    calibration_func=pol,
    param_calibration_func=d3pol2,
    target_variable="momentum",
    target_angle="theta_p",
    num_of_theta_intervals=11,
):

    # cal_arr = np.array([np.array([np.array(y) for y in x]) for x in cal_arr])
    error_counter = 0

    bdf["cos_theta_p"] = np.cos(bdf["theta_p"])
    edf["cos_theta_p"] = np.cos(edf["theta_p"])
    theta_interval = np.ptp(bdf[target_angle]) / (num_of_theta_intervals - 1)
    theta_min = min(bdf[target_angle])
    theta_max = max(bdf[target_angle])

    for n_sigms in range(*avg_sigmas):
        for t_sigms in range(*avg_t_sigmas):
            meas_betas = np.zeros(edf.shape[0])
            cur_ind = 0
            # meas_betas = []
            for entry, subentry in edf[
                f"unfixed_calculated_r_2d_{n_sigms}_rsigms_{t_sigms}_tsigms"
            ].groupby(level=0):
                if param_fit:
                    angle = edf[target_angle].loc[entry].iloc[0]
                    # meas_beta = pol(subentry.iloc[0], pol2(angle, *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][0]),
                    #               pol2(angle, *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][1]),
                    #               pol2(angle, *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][2]))
                    meas_beta = param_calibration_func(
                        (subentry.iloc[0], angle),
                        *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]],
                    )
                else:
                    if edf[target_angle].loc[entry].iloc[0] != theta_max:
                        try:
                            # meas_beta = pol(subentry.iloc[0], *(cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][(np.floor(((edf.cos_theta_p[entry].iloc[0]) - theta_min) / theta_interval)).astype(int)]))
                            meas_beta = calibration_func(
                                subentry.iloc[0],
                                *(
                                    cal_arr[n_sigms - avg_sigmas[0]][
                                        t_sigms - avg_t_sigmas[0]
                                    ][
                                        (
                                            np.floor(
                                                (
                                                    (
                                                        edf[target_angle]
                                                        .loc[entry]
                                                        .iloc[0]
                                                    )
                                                    - theta_min
                                                )
                                                / theta_interval
                                            )
                                        ).astype(int)
                                    ]
                                ),
                            )
                        except IndexError:
                            print(edf.cos_theta_p[entry].iloc[0])
                            print(
                                (
                                    np.floor(
                                        ((edf.cos_theta_p[entry].iloc[0]) - theta_min)
                                        / theta_interval
                                    )
                                ).astype(int)
                            )
                            meas_beta = 0
                            error_counter += 1
                    else:
                        # meas_beta = pol(subentry.iloc[0], *(cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][9]))
                        meas_beta = calibration_func(
                            subentry.iloc[0],
                            *(
                                cal_arr[n_sigms - avg_sigmas[0]][
                                    t_sigms - avg_t_sigmas[0]
                                ][num_of_theta_intervals - 2]
                            ),
                        )
                meas_betas[cur_ind : subentry.shape[0] + cur_ind] = np.repeat(
                    meas_beta, subentry.shape[0]
                )
                cur_ind += subentry.shape[0]
            edf[f"beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms"] = np.clip(
                meas_betas, a_min=None, a_max=0.9957
            )
            edf[f"delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"] = (
                edf[f"beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms"] - edf["beta"]
            )
            edf[f"eps_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"] = (
                edf[f"delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"] / edf["beta"] * 100
            )

            edf_to_bdf(edf[f"beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms"], bdf)
            edf_to_bdf(edf[f"delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"], bdf)
            edf_to_bdf(edf[f"eps_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"], bdf)
            print(error_counter)


def save_calibration_plot(fig, dir_to_save, deg_lim, r_sigms, avg_t_sigmas):
    filename = f"rsigm={r_sigms}_t_sigms={avg_t_sigmas[0]}-{avg_t_sigmas[-1] - 1}"
    if deg_lim:
        filename += "_10deg"
    if dir_to_save != "":
        fig.savefig(os.path.join("calibrations_barrel", dir_to_save, f"{filename}"))
        plt.close(fig)


def plot_calibration(
    t_bdf: pd.DataFrame,
    chosen_column: str,
    t_sigms: int,
    momentum_min,
    momentum_max,
    rs,
    pol_param,
    chi2,
    theta_interval_index,
    fig,
    axs,
    avg_sigmas,
    avg_t_sigmas,
    target_variable,
    target_angle,
    calibration_func,
):
    if t_sigms - avg_t_sigmas[0] != 0:
        for_colorbar = axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].hist2d(
            t_bdf[chosen_column],
            t_bdf[target_variable],
            bins=70,
            range=((0, 80), (momentum_min, momentum_max)),
        )
        fig.colorbar(
            for_colorbar[3], ax=axs[theta_interval_index, t_sigms - avg_t_sigmas[0]]
        )
    else:
        for_colorbar = axs[theta_interval_index].hist2d(
            t_bdf[chosen_column],
            t_bdf[target_variable],
            bins=70,
            range=((0, 80), (momentum_min, momentum_max)),
        )
        fig.colorbar(for_colorbar[3], ax=axs[theta_interval_index])
    if t_sigms - avg_t_sigmas[0] != 0:
        # axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].plot(rs, calibration_func(rs, *pol_param), label=r'$\chi^2$ = '+ str(chi2), c='r')
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_xlim((0, 90))
        # axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_ylim((0.955, momentum_max))
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_ylim(
            (momentum_min, momentum_max)
        )
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_xlabel(
            r"$R_{reco}$, mm"
        )
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_ylabel(
            r"$target_{true}$"
        )
    else:
        axs[theta_interval_index].plot(
            rs,
            calibration_func(rs, *pol_param),
            label=r"$\chi^2$ = " + str(round(chi2, 1)),
            c="r",
        )
        axs[theta_interval_index].set_xlim((0, 90))
        # axs[theta_interval_index].set_ylim((0.955, momentum_max))
        axs[theta_interval_index].set_ylim((momentum_min, momentum_max))
        axs[theta_interval_index].set_xlabel(r"$R_{reco}$, mm")
        axs[theta_interval_index].set_ylabel(r"target_{true}$")
        # axs[theta_interval_index].legend(loc="best")


def calibration_loop(
    bdf: pd.DataFrame,
    chosen_column: str,
    r_sigms: int,
    t_sigms: int,
    param_fit: bool,
    num_of_theta_intervals: int,
    to_return_unbinned: np.ndarray,
    errs_tmp: np.ndarray,
    fig,
    axs,
    avg_sigmas,
    avg_t_sigmas,
    target_variable,
    target_angle,
    calibration_func,
    p0,
):
    theta_p_max = max(bdf[target_angle])
    theta_p_min = min(bdf[target_angle])
    momentum_min = min(bdf[target_variable])
    momentum_max = max(bdf[target_variable])

    theta_intervals = np.linspace(theta_p_min, theta_p_max, num=num_of_theta_intervals)
    theta_dif = (theta_intervals[1:] + theta_intervals[:-1]) / 2

    for theta_interval_index in range(num_of_theta_intervals - 1):
        t_bdf = bdf.copy()
        t_bdf = t_bdf[np.isfinite(t_bdf[chosen_column])]
        t_bdf = t_bdf[t_bdf.signal_counts >= 15]  # 15
        t_bdf = t_bdf[t_bdf[target_angle] <= theta_intervals[theta_interval_index + 1]]
        t_bdf = t_bdf[t_bdf[target_angle] >= theta_intervals[theta_interval_index]]

        # t_bdf = t_bdf[t_bdf[target_variable] >= 0.965]
        # t_bdf = t_bdf[t_bdf[chosen_column] <= 65]
        # t_bdf = t_bdf[t_bdf[chosen_column] >= 25]

        pol_param, cov = curve_fit(
            calibration_func,
            t_bdf[chosen_column],
            t_bdf[target_variable],
            maxfev=50000000,
            p0=p0,
        )
        # cs = CubicSpline(t_bdf[chosen_column], t_bdf[target_variable])
        to_return_unbinned[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][
            theta_interval_index
        ] = pol_param
        pol_param_errs = np.sqrt(np.diag(cov))
        if not param_fit:
            errs_tmp[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][
                theta_interval_index
            ] = pol_param_errs

        rs = np.linspace(10, 80, num=50)
        chi2 = np.sum(
            (
                t_bdf[target_variable]
                - calibration_func(t_bdf[chosen_column], *pol_param)
            )
            ** 2
            / calibration_func(t_bdf[chosen_column], *pol_param)
        )
        plot_calibration(
            t_bdf,
            chosen_column,
            t_sigms,
            momentum_min,
            momentum_max,
            rs,
            pol_param,
            chi2,
            theta_interval_index,
            fig,
            axs,
            avg_sigmas,
            avg_t_sigmas,
            target_variable,
            target_angle,
            calibration_func,
        )


def param_fit_calibration(
    bdf: pd.DataFrame,
    chosen_column: str,
    r_sigms: int,
    t_sigms: int,
    avg_sigmas,
    avg_t_sigmas,
    fit_params,
    errs_pararm_fit,
    num_of_calibration_params,
    num_of_param_fit_params,
    target_variable,
    target_angle,
    param_calibration_func,
    p0_c,
):
    t_bdf = bdf.copy()
    t_bdf = t_bdf[np.isfinite(t_bdf[chosen_column])]
    t_bdf = t_bdf[t_bdf.signal_counts >= 5]

    # t_bdf = t_bdf[t_bdf[target_variable] >= 0.965]

    X = (np.array(t_bdf[chosen_column]), np.array(t_bdf[target_angle]))
    fit, errs = curve_fit(param_calibration_func, X, t_bdf[target_variable], p0=p0_c)
    errs_pararm_fit[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]] = np.sqrt(
        np.diag(errs)
    )

    fit_params[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]] = fit
    # for param in range(num_of_calibration_params):
    #     fit_params[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][param] = fit[
    #         param * 3 : param * 3 + 3
    #     ]
    chi2 = np.sum((t_bdf[target_variable] - param_calibration_func(X, *fit)) ** 2)


def rms90(arr):
    arr = arr.dropna()
    lower_limit = np.percentile(arr, 5)
    upper_limit = np.percentile(arr, 95)
    arr_filtered = arr[(arr >= lower_limit) & (arr <= upper_limit)]
    assert arr_filtered.shape
    rms = np.std(arr_filtered)

    return rms


def beta_from_momentum(p, mass):
    return p / np.sqrt(mass * mass + p * p)


def betaGroupsRMS90(bdf: pd.DataFrame, avg_sigmas: tuple, avg_t_sigmas: tuple, n=5):
    beta_sigms = np.full((np.ptp(avg_sigmas), np.ptp(avg_t_sigmas), n), 0.0)
    beta_epss = np.full((np.ptp(avg_sigmas), np.ptp(avg_t_sigmas), n), 0.0)
    beta_sigms_sigms = np.full((np.ptp(avg_sigmas), np.ptp(avg_t_sigmas), n), 0.0)

    masses_mean = np.full((np.ptp(avg_sigmas), np.ptp(avg_t_sigmas), n), 0.0)
    masses_upper = np.full((np.ptp(avg_sigmas), np.ptp(avg_t_sigmas), n), 0.0)
    masses_lower = np.full((np.ptp(avg_sigmas), np.ptp(avg_t_sigmas), n), 0.0)

    for group in range(1, n + 1):
        data = bdf[bdf["param_group"] == group]
        for i in range(np.ptp(avg_sigmas)):
            for j in range(np.ptp(avg_t_sigmas)):
                population_fourth_moment = np.mean(
                    bdf[
                        f"delta_beta_{i + avg_sigmas[0]}_rsigms_{j + avg_t_sigmas[0]}_tsigms"
                    ]
                    ** 4
                )
                sample_fourth_moment = np.mean(
                    data[
                        f"delta_beta_{i + avg_sigmas[0]}_rsigms_{j + avg_t_sigmas[0]}_tsigms"
                    ]
                    ** 4
                )
                # print(np.std(data[f'delta_beta_{i + avg_sigmas[0]}_rsigms_{j + avg_t_sigmas[0]}_tsigms']))
                beta_sigms[i, j, group - 1] = rms90(
                    data[
                        f"delta_beta_{i + avg_sigmas[0]}_rsigms_{j + avg_t_sigmas[0]}_tsigms"
                    ]
                )
                # assert not np.isnan(beta_sigms[i, j, group - 1])
                beta_epss[i, j, group - 1] = rms90(
                    data[
                        f"eps_beta_{i + avg_sigmas[0]}_rsigms_{j + avg_t_sigmas[0]}_tsigms"
                    ]
                )
                beta_sigms_sigms[i, j, group - 1] = np.sqrt(
                    2
                    * np.abs(sample_fourth_moment - population_fourth_moment)
                    / (data.shape[0])
                )
                masses_mean[i, j, group - 1] = np.mean(
                    (
                        data.momentum
                        / data.beta_from_calc_r_4_rsigms_4_tsigms
                        * np.sqrt(1 - data.beta_from_calc_r_4_rsigms_4_tsigms**2)
                    ).dropna()
                )
                masses_upper[i, j, group - 1] = (
                    np.mean(
                        (
                            data.momentum
                            / (
                                data.beta_from_calc_r_4_rsigms_4_tsigms
                                - beta_sigms[i, j, group - 1]
                            )
                            * np.sqrt(
                                1
                                - (
                                    data.beta_from_calc_r_4_rsigms_4_tsigms
                                    - beta_sigms[i, j, group - 1]
                                )
                                ** 2
                            )
                        ).dropna()
                    )
                    - masses_mean[i, j, group - 1]
                )
                masses_lower[i, j, group - 1] = masses_mean[i, j, group - 1] - np.mean(
                    (
                        data.momentum
                        / (
                            data.beta_from_calc_r_4_rsigms_4_tsigms
                            + beta_sigms[i, j, group - 1]
                        )
                        * np.sqrt(
                            1
                            - (
                                data.beta_from_calc_r_4_rsigms_4_tsigms
                                + beta_sigms[i, j, group - 1]
                            )
                            ** 2
                        )
                    ).dropna()
                )

    return (
        beta_sigms,
        beta_epss,
        beta_sigms_sigms,
        masses_mean,
        masses_upper,
        masses_lower,
    )


def plot_final_graph(
    edf,
    beta_sigms,
    beta_sigms_yerr,
    avg_sigmas,
    avg_t_sigmas,
    r_width,
    t_width,
    r_factor,
    t_factor,
    weighed,
    labels=["0"],
    to_save=True,
    deg_lim=False,
    num_of_groups=10,
    iteration=0,
):
    # labels = ['0', '1e3', '1e4', '1e5', '1e6']
    pi_mass = 139.57
    mu_mass = 105.65
    ka_mass = 493.67
    # labels = ["0"]
    labels = ["DCR = " + i + " $Hz/mm^2$" for i in labels]
    colors = ["c", "y", "g", "r", "m"]
    weight = "weighed" if weighed else "unweighed"
    y = np.arange(1, num_of_groups + 1)
    x = (
        y * (max(edf["beta"]) - min(edf["beta"]))
        - max(edf["beta"])
        + (num_of_groups + 1) * min(edf["beta"])
    ) / num_of_groups
    required_separation = [
        (beta_from_momentum(momentum_from_beta(b, pi_mass), mu_mass) - b) / 3 for b in x
    ]
    fig, axs = plt.subplots(
        np.ptp(avg_sigmas),
        np.ptp(avg_t_sigmas),
        figsize=(10 * np.ptp(avg_t_sigmas), 10 * np.ptp(avg_sigmas)),
    )
    title = f"Method: N(r) / r; {weight} Avg\nR Width = {r_width}mm, T Width = {t_width}ns\nR step factor = {r_factor}, T step factor = {t_factor}"
    if deg_lim:
        title += "\n" + r"$\theta_p < 10\deg$"
    # fig.suptitle(title)

    if np.ptp(avg_sigmas) > 1:
        for i in range(np.ptp(avg_sigmas)):
            for j in range(np.ptp(avg_t_sigmas)):
                for k in range(beta_sigms.shape[0]):
                    axs[i, j].plot(x, beta_sigms[k, i, j], label=labels[k], c=colors[k])
                    axs[i, j].errorbar(
                        x,
                        beta_sigms[k, i, j],
                        xerr=[np.diff(x)[0] / 4 for _ in x],
                        linestyle="",
                        c=colors[k],
                    )
                    axs[i, j].errorbar(
                        x,
                        beta_sigms[k, i, j],
                        yerr=beta_sigms_yerr[k, i, j],
                        linestyle="",
                        c=colors[k],
                    )
                axs[i, j].legend(loc="upper right")
                axs[i, j].set_xlabel("Beta Group")
                axs[i, j].set_ylabel(r"RMS90($\Delta\beta$)")
                axs[i, j].set_ylim((0, 0.004))
                axs[i, j].set_title(
                    f"Velocity resoultion for\nr window width = {avg_sigmas[0] + i}$\sigma$\nt window width = {avg_t_sigmas[0] + j}$\sigma$"
                )
                axs[i, j].grid()
    elif np.ptp(avg_t_sigmas) > 1:
        for j in range(np.ptp(avg_t_sigmas)):
            for k in range(beta_sigms.shape[0]):
                axs[j].plot(x, beta_sigms[k, 0, j], label=labels[k], c=colors[k])
                axs[j].errorbar(
                    x,
                    beta_sigms[k, 0, j],
                    xerr=[np.diff(x)[0] / 4 for _ in x],
                    linestyle="",
                    c=colors[k],
                )
                axs[j].errorbar(
                    x,
                    beta_sigms[k, 0, j],
                    yerr=beta_sigms_yerr[k, 0, j],
                    linestyle="",
                    c=colors[k],
                )
            axs[j].legend(loc="upper right")
            axs[j].set_xlabel("Beta Group")
            axs[j].set_ylabel(r"RMS90($\Delta\beta)$")
            axs[j].set_ylim((0, 0.004))
            axs[j].set_title(
                f"Velocity resoultion for\nr window width = {avg_sigmas[0]}$\sigma$\nt window width = {avg_t_sigmas[0] + j}$\sigma$"
            )
            axs[j].grid()
    else:
        for k in range(beta_sigms.shape[0]):
            axs.plot(x, beta_sigms[k, 0, 0], label=labels[k], c=colors[k])
            axs.errorbar(
                x,
                beta_sigms[k, 0, 0],
                xerr=[np.diff(x)[0] / 4 for _ in x],
                linestyle="",
                c=colors[k],
            )
            axs.errorbar(
                x,
                beta_sigms[k, 0, 0],
                yerr=beta_sigms_yerr[k, 0, 0],
                linestyle="",
                c=colors[k],
            )
            axs.plot(x, required_separation, c="r", linestyle="--")
            required_separation = [
                (beta_from_momentum(momentum_from_beta(b, pi_mass), mu_mass) - b) / 2
                for b in x
            ]
        axs.plot(x, required_separation, c="b", linestyle="--")
        axs.legend(loc="upper right")
        axs.set_xlabel("Beta Group")
        axs.set_ylabel(r"RMS90($\Delta\beta$)")
        # axs.set_ylim((0, 0.002))
        # revert back
        axs.set_ylim((0, 0.004))

        axs.set_title(
            f"Velocity resoultion for\nr window width = {avg_sigmas[0]}$\sigma$\nt window width = {avg_t_sigmas[0]}$\sigma$"
        )
        axs.grid()

    if to_save:
        filename = f"{weight}_avg_rw={r_width}_tw={t_width}_rs={r_factor}_ts={t_factor}_rsigms={avg_sigmas[0]}-{avg_sigmas[-1] - 1}_tsigms={avg_t_sigmas[0]}-{avg_t_sigmas[-1] - 1}"
        if deg_lim:
            filename += "_10deg"
        filename += f"_{iteration}"
        filename += ".png"
        fig.savefig(os.path.join("results_barrel", f"{filename}"))
        plt.close(fig)
    else:
        plt.show()


def plot_final_mass_graph(
    edf,
    mass_mean,
    mass_upper,
    mass_lower,
    avg_sigmas,
    avg_t_sigmas,
    r_width,
    t_width,
    r_factor,
    t_factor,
    weighed,
    to_save=True,
    deg_lim=False,
    num_of_groups=10,
    iteration=0,
):
    labels = ["0"]
    labels = ["DCR = " + i + " $Hz/mm^2$" for i in labels]
    colors = ["c", "y", "g", "r", "m"]
    weight = "weighed" if weighed else "unweighed"
    pi_mass = 139.57
    mu_mass = 105.65
    ka_mass = 493.67
    y = np.arange(1, num_of_groups + 1)
    x = (
        y * (max(edf["beta"]) - min(edf["beta"]))
        - max(edf["beta"])
        + (num_of_groups + 1) * min(edf["beta"])
    ) / num_of_groups

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    title = f"Method: N(r) / r; {weight} Avg\nR Width = {r_width}mm, T Width = {t_width}ns\nR step factor = {r_factor}, T step factor = {t_factor}"
    if deg_lim:
        title += "\n" + r"$\theta_p < 10\deg$"
    for k in range(mass_mean.shape[0]):
        axs.plot(x, mass_mean[k, 0], label=labels[k], c=colors[k])
        axs.errorbar(
            x,
            mass_mean[k, 0],
            yerr=np.array(list(zip(mass_lower[k, 0], mass_upper[k, 0]))).T,
            linestyle="",
            c=colors[k],
        )

    axs.plot((min(x), max(x)), (pi_mass, pi_mass), linestyle="--", c="red")
    axs.plot((min(x), max(x)), (mu_mass, mu_mass), linestyle="--", c="red")
    # axs.plot((min(x), max(x)), (ka_mass, ka_mass), linestyle='--', c='red')

    axs.legend(loc="upper right")
    axs.set_xlabel("Beta Group")
    axs.set_ylabel(r"Measured Mass, MeV")
    axs.set_title(
        f"Mass resoultion for\nr window width = {avg_sigmas[0]}$\sigma$\nt window width = {avg_t_sigmas[0]}$\sigma$"
    )
    axs.grid()

    if to_save:
        filename = f"{weight}_avg_rw={r_width}_tw={t_width}_rs={r_factor}_ts={t_factor}_rsigms={avg_sigmas[0]}-{avg_sigmas[-1] - 1}_tsigms={avg_t_sigmas[0]}-{avg_t_sigmas[-1] - 1}"
        if deg_lim:
            filename += "_10deg"
        filename += f"_{iteration}"
        filename += ".png"
        fig.savefig(os.path.join("results_barrel", f"{filename}"))
        plt.close(fig)
    else:
        plt.show()


def exp_like(x, a, b):
    return a * np.exp(b * x)


def exp_like_3param(x, a, b, c):
    return a * np.exp((x - b) / c)


def pol2_exp_like(x, a0, a1, a2, b0, b1, b2):
    r, theta = x
    return exp_like(r, a0 + a1 * theta + a2 * theta**2, b0 + b1 * theta + b2 * theta**2)


def pol3(x, p0, p1, p2, p3):
    return p0 + p1 * x + p2 * x**2 + p3 * x**3


def lin(x, a, b):
    return a * x + b


def pol2_lin(x, a0, a1, a2, b0, b1, b2):
    r, theta = x
    return lin(r, a0 + a1 * theta + a2 * theta**2, b0 + b1 * theta + b2 * theta**2)


def pol2_pol2(x, a0, a1, a2, b0, b1, b2, c0, c1, c2):
    r, theta = x
    return pol2(
        r,
        a0 + a1 * theta + a2 * theta**2,
        b0 + b1 * theta + b2 * theta**2,
        c0 + c1 * theta + c2 * theta**2,
    )


def calibration(
    edf: pd.DataFrame,
    idf: pd.DataFrame,
    bdf: pd.DataFrame,
    avg_sigmas: tuple,
    avg_t_sigmas: tuple,
    step=3.0,
    method="N/r",
    t_window_width=2,
    r_width_factor=2,
    t_width_factor=8,
    full_width_t_hist=False,
    weighed=True,
    deg_lim=False,
    param_fit=False,
    calibration_func=pol,
    param_calibration_func=d3pol2,
    num_of_calibration_params=3,
    num_of_param_fit_params=3,
    target_variable="momentum",
    target_angle="theta_p",
    num_of_theta_intervals=11,
    p0=(100, 1, 100),
    p0_c=(1.219, -0.5588, 0.2946, 864.4, -1922, 1055, -2535, 6572, -3751),
    use_decision_tree=False,
):
    to_return_unbinned = np.full(
        (
            np.ptp(avg_sigmas),
            np.ptp(avg_t_sigmas),
            num_of_theta_intervals - 1,
            num_of_calibration_params,
        ),
        0.0,
    )
    errs_tmp = np.full(
        (
            np.ptp(avg_sigmas),
            np.ptp(avg_t_sigmas),
            num_of_theta_intervals - 1,
            num_of_calibration_params,
        ),
        0.0,
    )

    fit_params = np.full(
        (
            np.ptp(avg_sigmas),
            np.ptp(avg_t_sigmas),
            num_of_calibration_params * num_of_param_fit_params,
        ),
        0.0,
    )
    errs_pararm_fit = np.full(
        (
            np.ptp(avg_sigmas),
            np.ptp(avg_t_sigmas),
            num_of_calibration_params * num_of_param_fit_params,
        ),
        0.0,
    )

    models = np.full(
        (
            np.ptp(avg_sigmas),
            np.ptp(avg_t_sigmas),
        ),
        None,
        dtype=object,
    )

    dir_to_save = f"{'weighed' if weighed else 'unweighed'}_rw={step}_tw={t_window_width}_rs={r_width_factor}_ts={t_width_factor}"
    if not os.path.exists(os.path.join("calibrations_barrel", dir_to_save)):
        os.mkdir(os.path.join("calibrations_barrel", dir_to_save))

    for r_sigms in range(*avg_sigmas):
        fig, axs = plt.subplots(
            num_of_theta_intervals - 1,
            np.ptp(avg_t_sigmas),
            figsize=(16 * np.ptp(avg_t_sigmas), 9 * (num_of_theta_intervals - 1)),
        )
        for t_sigms in range(*avg_t_sigmas):
            chosen_column = f"unfixed_calculated_r_2d_{r_sigms}_rsigms_{t_sigms}_tsigms"

            calibration_loop(
                bdf,
                chosen_column,
                r_sigms,
                t_sigms,
                param_fit,
                num_of_theta_intervals,
                to_return_unbinned,
                errs_tmp,
                fig,
                axs,
                avg_sigmas,
                avg_t_sigmas,
                target_variable,
                target_angle,
                calibration_func,
                p0,
            )
            if param_fit:
                param_fit_calibration(
                    bdf,
                    chosen_column,
                    r_sigms,
                    t_sigms,
                    avg_sigmas,
                    avg_t_sigmas,
                    fit_params,
                    errs_pararm_fit,
                    num_of_calibration_params,
                    num_of_param_fit_params,
                    target_variable,
                    target_angle,
                    param_calibration_func,
                    p0_c=p0_c,
                )
            if use_decision_tree:
                boost_calibraion(
                    bdf,
                    chosen_column,
                    r_sigms,
                    t_sigms,
                    avg_sigmas,
                    avg_t_sigmas,
                    models,
                    target_variable,
                    target_angle,
                )
        save_calibration_plot(fig, dir_to_save, deg_lim, r_sigms, avg_t_sigmas)
    if use_decision_tree:
        # print(models)
        return models, errs_pararm_fit  # TODO errors
    if param_fit:
        return fit_params, errs_pararm_fit
    return to_return_unbinned, errs_tmp


def boost_calibraion(
    bdf: pd.DataFrame,
    chosen_column: str,
    r_sigms: int,
    t_sigms: int,
    avg_sigmas,
    avg_t_sigmas,
    models,
    target_variable,
    target_angle,
):
    t_bdf = bdf.sample(frac=0.8, random_state=42).copy()
    # print(t_bdf.index)
    t_bdf = t_bdf[np.isfinite(t_bdf[chosen_column])]
    t_bdf = t_bdf[t_bdf.signal_counts >= 5]

    # Define features and target
    X = t_bdf[[chosen_column, target_angle]]
    y = t_bdf[target_variable]

    # Initialize and fit the XGBoost regressor
    xgb_model = XGBRegressor(
        objective="reg:squarederror", n_estimators=100, learning_rate=0.1
    )
    xgb_model.fit(X, y)

    # Store fitted model and error
    models[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]] = xgb_model


def rSlidingWindowLoop2Boosted(
    bdf: pd.DataFrame,
    avg_sigmas: tuple,
    avg_t_sigmas: tuple,
    models: np.ndarray,
    target_angle="theta_p",
):
    for n_sigms in range(*avg_sigmas):
        for t_sigms in range(*avg_t_sigmas):
            chosen_column = f"unfixed_calculated_r_2d_{n_sigms}_rsigms_{t_sigms}_tsigms"
            X = bdf[[chosen_column, target_angle]]
            meas_betas = models[n_sigms - avg_sigmas[0]][
                t_sigms - avg_t_sigmas[0]
            ].predict(X)
            bdf[f"beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms"] = np.clip(
                meas_betas, a_min=None, a_max=1  # 0.9957
            )
            bdf[f"delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"] = (
                bdf[f"beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms"] - bdf["beta"]
            )
            bdf[f"eps_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"] = (
                bdf[f"delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms"] / bdf["beta"] * 100
            )


def rSlidingWindow(
    edf: pd.DataFrame,
    idf: pd.DataFrame,
    bdf: pd.DataFrame,
    avg_sigmas: tuple,
    avg_t_sigmas: tuple,
    step=3.0,
    method="N/r",
    cal_arr=False,
    errs=False,
    t_window_width=2,
    r_width_factor=2,
    t_width_factor=8,
    full_width_t_hist=True,
    num_of_groups=5,
    weighed=True,
    deg_lim=False,
    param_fit=False,
    calibration_func=pol,
    param_calibration_func=d3pol2,
    num_of_calibration_params=3,
    num_of_param_fit_params=3,
    target_variable="beta",
    target_angle="cos_theta_p",
    num_of_theta_intervals=11,
    p0=(100, 1, 100),
    p0_c=(1.219, -0.5588, 0.2946, 864.4, -1922, 1055, -2535, 6572, -3751),
    what_to_group="beta",
    use_decision_tree=False,
    dcr=0,
):
    """
    Applies a sliding window approach to calculate effective radius of a Cherenkov circle.
    Applies a calibration method to calculate beta of primary particle.

    Parameters:
    - edf (pd.DataFrame): Full data frame containing hit data to be processed.
    - idf (pd.DataFrame): Data frame containing geometry and detector information.
    - bdf (pd.DataFrame): Compact data frame containing general information about events.
    - avg_sigmas (tuple): How many radius window widths are used for averaging (min, max + 1).
    - avg_t_sigmas (tuple): How many time window widths are used for averaging (min, max + 1).
    - step (float): Step size for the sliding window, defaults to 3.
    - method (str): Radius calculation method, default is 'N/r'.
    - cal_arr (bool): Indicates if a calibration array is provided; if False, calibration is performed.
    - t_window_width (int): Temporal window width for averaging.
    - r_width_factor (int): Radial window width factor.
    - t_width_factor (int): Temporal width factor for windowing.
    - full_width_t_hist (bool): Flag for using full-width histogram in calculations.
    - num_of_groups (int): Number of groups for data partitioning.
    - weighed (bool): Whether to apply weighting in radius calculations.
    - deg_lim (bool): Limit theta_p at 10 degrees.
    - param_fit (bool): Flag to enable parameter fitting in calibration.
    - calibration_func (callable): Function used for calibration.
    - param_calibration_func (callable): Function for parameterized calibration.
    - num_of_calibration_params (int): Number of parameters for calibration.
    - num_of_param_fit_params (int): Number of parameters for parameter fitting.
    - target_variable (str): Variable of interest in the calibration, default is 'beta'.
    - target_angle (str): Target angle variable, default is 'cos_theta_p'.
    - num_of_theta_intervals (int): Number of theta intervals for non-param-fit calibration.
    - p0 (tuple): Initial parameter values for calibration function.
    - p0_c (tuple): Initial parameter values for parameterized calibration function.
    - what_to_group (str): Variable used for grouping, default is 'beta'.

    Returns:
    - tuple: A tuple containing the calibration array (`cal_arr`) and error values (`errs`).
    """
    rSlidingWindowIntro(
        edf,
        idf,
        bdf,
        avg_sigmas,
        avg_t_sigmas,
        step=step,
        method=method,
        cal_arr=cal_arr,
        t_window_width=t_window_width,
        r_width_factor=r_width_factor,
        t_width_factor=t_width_factor,
        num_of_groups=num_of_groups,
        what_to_group=what_to_group,
    )
    rSlidingWindowLoop1(
        edf,
        idf,
        bdf,
        avg_sigmas,
        avg_t_sigmas,
        step=step,
        method=method,
        cal_arr=cal_arr,
        t_window_width=t_window_width,
        r_width_factor=r_width_factor,
        t_width_factor=t_width_factor,
        full_width_t_hist=full_width_t_hist,
        weighed=weighed,
    )
    if dcr < 1e5:
        bdf.dropna(
            subset=[f"unfixed_calculated_r_2d_{avg_sigmas[0]}_rsigms_4_tsigms"],
            inplace=True,
        )
        edf.dropna(
            subset=[f"unfixed_calculated_r_2d_{avg_sigmas[0]}_rsigms_4_tsigms"],
            inplace=True,
        )
    quantile_transformer = QuantileTransformer(
        output_distribution="uniform", random_state=0
    )
    # bdf[f"unfixed_calculated_r_2d_{avg_sigmas[0]}_rsigms_4_tsigms"] = (
    #     quantile_transformer.fit_transform(
    #         bdf[[f"unfixed_calculated_r_2d_{avg_sigmas[0]}_rsigms_4_tsigms"]]
    #     )
    # )

    if cal_arr is False:
        cal_arr, errs = calibration(
            edf,
            idf,
            bdf,
            avg_sigmas=avg_sigmas,
            avg_t_sigmas=avg_t_sigmas,
            step=step,
            t_window_width=t_window_width,
            r_width_factor=r_width_factor,
            t_width_factor=t_width_factor,
            weighed=weighed,
            deg_lim=deg_lim,
            param_fit=param_fit,
            calibration_func=calibration_func,
            param_calibration_func=param_calibration_func,
            num_of_calibration_params=num_of_calibration_params,
            num_of_param_fit_params=num_of_param_fit_params,
            target_variable=target_variable,
            target_angle=target_angle,
            num_of_theta_intervals=num_of_theta_intervals,
            p0=p0,
            p0_c=p0_c,
            use_decision_tree=use_decision_tree,
        )
    if use_decision_tree:
        rSlidingWindowLoop2Boosted(
            bdf,
            avg_sigmas,
            avg_t_sigmas,
            cal_arr,
            target_angle=target_angle,
        )
    else:
        rSlidingWindowLoop2(
            edf,
            idf,
            bdf,
            avg_sigmas,
            avg_t_sigmas,
            step=step,
            method=method,
            cal_arr=cal_arr,
            t_window_width=t_window_width,
            r_width_factor=r_width_factor,
            t_width_factor=t_width_factor,
            param_fit=param_fit,
            calibration_func=calibration_func,
            param_calibration_func=param_calibration_func,
            target_variable=target_variable,
            target_angle=target_angle,
            num_of_theta_intervals=num_of_theta_intervals,
        )

    return cal_arr, errs


def init_coords(file, MAXIMUM_EVENT_GROUP_NUMBER, grid):
    x = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "FarichBarrelG4Hits.postStepPosition.x"
        ].array()
    )
    y = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "FarichBarrelG4Hits.postStepPosition.y"
        ].array()
    )
    z = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "FarichBarrelG4Hits.postStepPosition.z"
        ].array()
    )
    wvs = (
        1239.841
        / np.array(
            file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
                "FarichBarrelG4Hits.energy"
            ].array()
        )
        * 1e-9
    )
    t = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "FarichBarrelG4Hits.localTime"
        ].array()
    )
    x3 = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "allGenParticles.core.p4.px"
        ].array()
    )
    y3 = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "allGenParticles.core.p4.py"
        ].array()
    )
    z3 = np.array(
        file[f"events;{MAXIMUM_EVENT_GROUP_NUMBER}"][
            "allGenParticles.core.p4.pz"
        ].array()
    )
    true_direction_coordinates = np.column_stack((x3, y3, z3))
    for i in range(len(wvs)):
        wvs[i] = lin_move_to_grid(wvs[i], grid[2])
    coordinates = np.column_stack((x, y, z, wvs, t))
    return coordinates, true_direction_coordinates


def init_sipm_eff():
    pdes_tmp = pd.read_csv("PDE.csv", sep=";", names=["A"])
    t1 = []
    t2 = []
    for inedx, row in pdes_tmp.iterrows():
        t1.append(float(row["A"].split(";")[0].replace(",", ".")))
        t2.append(float(row["A"].split(";")[1].replace(",", ".")))
    PDE_wvs = np.linspace(200, 900, 128)
    PDEs = np.interp(PDE_wvs, t1, t2, left=0, right=0)
    sipm_eff = fix_PDE_plot(PDEs, PDE_wvs)
    return sipm_eff, PDE_wvs


def addNoise(
    idf: pd.DataFrame,
    edf: pd.DataFrame,
    bdf: pd.DataFrame,
    noiseTimeRange,
    noisefreqpersqmm,
    shiftSignalTimes=True,
):
    x_grid = np.arange(-250, 250, SIPM_CELL_SIZE)
    y_grid = np.arange(-250, 250, SIPM_CELL_SIZE)

    nevents = bdf.shape[0]
    munoise = (
        (noiseTimeRange[1] - noiseTimeRange[0]) * 1e-9 * noisefreqpersqmm * (500**2)
    )

    print(
        f"    Generate noise with DCR per mm^2 {noisefreqpersqmm}, mean number of hits per event: {munoise:.2f}.",
        end="\n",
    )

    noisehits = rng.poisson(
        munoise, nevents
    )  # генерация массива числа шумовых срабатываний в событиях по пуассоновскому распределению
    Ndc = int(noisehits.sum())  # общее число шумовых срабатываний (скаляр)
    signalhits = bdf["nhits"]  # массив числа сигнальных срабатываний по событиям

    # случайное смещение сигнальных срабатываний в пределах временного окна генерации шума
    if shiftSignalTimes:
        edf["t_c"] += np.repeat(
            rng.uniform(0, noiseTimeRange[1] - 2, nevents), bdf["nhits"]
        )

    edf["signal"] = np.ones(
        signalhits.sum(), bool
    )  # разметка сигнальных срабатываний значением 'signal' True
    if Ndc == 0:  # если нет шумовых срабатываний
        return edf  # возвращаем исходный датафрейм с добавлением колонки 'signal'

    xich = rng.choice(
        x_grid.size, Ndc
    )  # генерация случайных номеров сработавших каналов с возможным повтором
    yich = rng.choice(
        y_grid.size, Ndc
    )  # генерация случайных номеров сработавших каналов с возможным повтором

    xh = x_grid[xich]  # x-координата сработавших каналов
    yh = y_grid[yich]  # y-координата сработавших каналов
    zh = 1000.0  # z-координата срабатываний (скаляр)
    th = rng.uniform(
        noiseTimeRange[0], noiseTimeRange[1], size=Ndc
    )  # генерация времён срабатываний по однородному распределению

    # нумерация шумовых срабатываний по событиям
    ievent = np.repeat(
        bdf.index, noisehits
    )  # массив номеров событий для записи в датафрейм
    ihit = np.zeros(
        Ndc, "int64"
    )  # инициализация массива номеров срабатываний для записи в датафрейм
    index = 0
    for i in range(nevents):
        ihit[index : index + noisehits[i]] = signalhits[i] + np.arange(noisehits[i])
        index += noisehits[i]

    # создание датафрейма с шумовыми срабатываниями того же формата, что hitdf
    noisedf = pd.DataFrame(
        {"x_cn": xh, "y_cn": yh, "z_c": zh, "t_c": th, "signal": np.zeros(Ndc, bool)},
        index=pd.MultiIndex.from_arrays((ievent, ihit), names=("entry", "subentry")),
    )

    # TO DO: случайное смещение кольца в фотодетекторе (сдвиг координат сигнальных хитов).
    # Сложность с реализацией для неравномерной сетки пикселей, т.к. зазоры между матрицами больше зазоров между пикселями в матрице.
    # Проще сделать в моделировании.

    # сливаем сигнальный и шумовой датафрейм и сортируем указатель событий и срабатываний
    hitdf2 = pd.concat((edf, noisedf), copy=False).sort_index(
        level=("entry", "subentry")
    )

    columns_to_fill = [
        "x_i",
        "y_i",
        "true_p",
        "beta",
        "x_p",
        "y_p",
        "z_p",
        "nx_p",
        "ny_p",
        "nz_p",
    ]

    hitdf2[columns_to_fill] = (
        hitdf2.groupby(level="entry")[columns_to_fill]
        .apply(lambda group: group.ffill())
        .reset_index(level=0, drop=True)
    )

    hitdf2["x_c"] = (
        hitdf2.groupby(level="entry")
        .apply(lambda group: group["x_c"].combine_first(group["x_cn"] + group["x_i"]))
        .reset_index(level=0, drop=True)
    )
    hitdf2["y_c"] = (
        hitdf2.groupby(level="entry")
        .apply(lambda group: group["y_c"].combine_first(group["y_cn"] + group["y_i"]))
        .reset_index(level=0, drop=True)
    )

    hitdf2.drop(["x_cn", "y_cn"], axis=1, inplace=True)

    # обновляем количества срабатываний в partdf, добавляя количества шумовых срабатываний по событиям
    bdf["sum_hits"] = bdf["nhits"] + noisehits

    return hitdf2


def uncertainty_introduction_to_direction(true_direction_coordinates):
    for i in range(true_direction_coordinates.shape[0]):
        n = np.array(
            [
                true_direction_coordinates[i][0][0],
                true_direction_coordinates[i][1][0],
                true_direction_coordinates[i][2][0],
            ]
        )
        n_magnitude = np.linalg.norm(n)

        # Угол с осью z
        dot_product = np.dot(n, [0, 0, 1.0])
        cos_theta = dot_product / n_magnitude  # |z| = 1
        theta = np.arccos(cos_theta)

        # Добавляем случайную ошибку в угол (±1%)
        sigma = 0.001
        delta_theta = np.random.normal(0, sigma)
        theta_prime = theta + delta_theta

        # Новое значение cos(theta')
        cos_theta_prime = np.cos(theta_prime)

        # Сохраняем азимутальный угол phi
        phi = np.arctan2(n[1], n[0])

        # Новые координаты вектора
        n_prime = np.zeros(3)
        n_prime[0] = n_magnitude * np.sin(theta_prime) * np.cos(phi)
        n_prime[1] = n_magnitude * np.sin(theta_prime) * np.sin(phi)
        n_prime[2] = n_magnitude * np.cos(theta_prime)
        if np.abs(true_direction_coordinates[i][0][0] - n_prime[0]) > 0.2:
            print(i)
            break

        true_direction_coordinates[i][0][0] = n_prime[0]
        true_direction_coordinates[i][1][0] = n_prime[1]
        true_direction_coordinates[i][2][0] = n_prime[2]


def create_edf(
    filepath_to_first="fullsim_optical_2000_pi_bin_1_FARICH_35mm_no_no_trackers.root",
    num_of_files=10,
    sample_num=None,
    uncertain_angle=False,
    is_mu=False,
    is_ka=False,
):
    datadir = "data"
    sipm_eff, PDE_wvs = init_sipm_eff()
    for key in sipm_eff.keys():
        sipm_eff[key] = sipm_eff[key] / 0.55414 * 0.38

    x_grid = np.arange(
        -3 * norm_r * np.sin(np.pi / 27),
        3 * norm_r * np.sin(np.pi / 27),
        SIPM_CELL_SIZE,
    )
    z_grid = np.arange(-1400, 1400, SIPM_CELL_SIZE)
    grid = (x_grid, z_grid, PDE_wvs)

    split_filepath = filepath_to_first.split("_1_")
    filepath_binned = os.path.join(
        datadir, f"{split_filepath[0]}_{1}_{split_filepath[1]}"
    )
    file_binned = uproot.open(filepath_binned)
    coordinates, true_direction_coordinates = init_coords(
        file_binned, int(str(file_binned.keys()[0]).split(";")[1][:-1]), grid
    )
    if num_of_files > 1:
        for i in range(2, num_of_files + 1):
            filepath_binned = os.path.join(
                datadir, f"{split_filepath[0]}_{i}_{split_filepath[1]}"
            )
            file_binned = uproot.open(filepath_binned)
            coordinates_i, true_direction_coordinates_i = init_coords(
                file_binned, int(str(file_binned.keys()[0]).split(";")[1][:-1]), grid
            )
            coordinates = np.concatenate((coordinates, coordinates_i), axis=0)
            true_direction_coordinates = np.concatenate(
                (true_direction_coordinates, true_direction_coordinates_i), axis=0
            )

    idx_to_drop = []
    for i in range(coordinates.shape[0]):
        if coordinates[i][0].shape[0] == 0:
            idx_to_drop.append(i)

    coordinates = np.delete(coordinates, idx_to_drop, axis=0)
    true_direction_coordinates = np.delete(
        true_direction_coordinates, idx_to_drop, axis=0
    )
    print(coordinates.shape)
    print(true_direction_coordinates.shape)

    # sample_num = 4 # 0 to 4
    if sample_num is not None:
        sample_idx = np.random.permutation(coordinates.shape[0])[
            20000 * sample_num : 20000 * (sample_num + 1)
        ]
        coordinates = coordinates[sample_idx]
        true_direction_coordinates = true_direction_coordinates[sample_idx]

    if uncertain_angle:
        uncertainty_introduction_to_direction(true_direction_coordinates)

    sipm_sim(coordinates, sipm_eff)

    idx_to_drop = []
    for i in range(coordinates.shape[0]):
        if coordinates[i][0].shape[0] == 0:
            idx_to_drop.append(i)
    coordinates = np.delete(coordinates, idx_to_drop, axis=0)
    true_direction_coordinates = np.delete(
        true_direction_coordinates, idx_to_drop, axis=0
    )
    print(coordinates.shape)
    print(true_direction_coordinates.shape)

    main_angles = rotate_lines(true_direction_coordinates)
    intersections = find_intersections(true_direction_coordinates)
    rotate_events(coordinates, main_angles)
    move_events_to_grid(coordinates, grid)
    repeat_nums = np.array([coord[0].shape[0] for coord in coordinates])
    edf = pd.DataFrame(coordinates, columns=["x_c", "y_c", "z_c", "wv_c", "t_c"])

    unraveled_data = {col: [] for col in edf.columns}
    row_indices = []

    # Iterate over the DataFrame and unravel the arrays
    for i, row in edf.iterrows():
        max_length = max(len(row[col]) for col in edf.columns)
        for k in range(max_length):
            row_indices.append((i, k))
            for col in edf.columns:
                if k < len(row[col]):
                    unraveled_data[col].append(row[col][k])
                else:
                    unraveled_data[col].append(
                        np.nan
                    )  # Handle cases where arrays are of different lengths

    # Create a new DataFrame from the unraveled data
    unraveled_df = pd.DataFrame(unraveled_data)

    # Create a MultiIndex for the rows
    multi_index = pd.MultiIndex.from_tuples(row_indices, names=["entry", "subentry"])
    unraveled_df.index = multi_index

    edf = unraveled_df

    edf["x_i"] = np.repeat(intersections[:, 0], repeat_nums, axis=0)
    edf["z_i"] = np.repeat(intersections[:, 2], repeat_nums, axis=0)

    x = y = z = x3 = y3 = z3 = unraveled_data = row_indices = main_angles = (
        intersections
    ) = wvs = coordinates = file = coordinates_low = file_low = 0

    bdf = pd.DataFrame()
    gdf = pd.DataFrame()
    gdf["nhits"] = repeat_nums

    mu_mass = 105.65
    pi_mass = 139.57
    ka_mass = 493.68
    mass = mu_mass if is_mu else (ka_mass if is_ka else pi_mass)
    edf.drop("y_c", axis=1, inplace=True)
    edf.drop("wv_c", axis=1, inplace=True)
    edf.rename(columns={"z_c": "y_c", "z_i": "y_i"}, inplace=True)
    edf["z_c"] = np.zeros(edf.shape[0]) + 1000  # why 2000?
    edf["mass"] = np.ones(edf.shape[0]) * mass
    edf["true_p"] = np.repeat(
        np.linalg.norm(true_direction_coordinates.astype("float"), axis=1) * 1000,
        repeat_nums,
        axis=0,
    )
    edf["beta"] = edf.true_p / np.sqrt(mass**2 + edf.true_p**2)
    edf["x_p"] = np.zeros(edf.shape[0])
    edf["y_p"] = np.zeros(edf.shape[0])
    edf["z_p"] = np.zeros(edf.shape[0])
    edf["nx_p"] = np.repeat(
        (
            true_direction_coordinates
            / np.array(
                [
                    np.linalg.norm(true_direction_coordinates.astype("float"), axis=1)
                    for i in range(3)
                ]
            ).T
        ).astype("float")[:, 0],
        repeat_nums,
        axis=0,
    )
    edf["ny_p"] = np.repeat(
        (
            true_direction_coordinates
            / np.array(
                [
                    np.linalg.norm(true_direction_coordinates.astype("float"), axis=1)
                    for i in range(3)
                ]
            ).T
        ).astype("float")[:, 2],
        repeat_nums,
        axis=0,
    )
    edf["nz_p"] = np.repeat(
        (
            true_direction_coordinates
            / np.array(
                [
                    np.linalg.norm(true_direction_coordinates.astype("float"), axis=1)
                    for i in range(3)
                ]
            ).T
        ).astype("float")[:, 1],
        repeat_nums,
        axis=0,
    )

    true_direction_coordinates = repeat_nums = true_direction_coordinates_low = 0

    return edf, bdf, gdf


def create_edf_for_field(  # needs reworking for using intersection point from farich itself
    filepath_to_first="fullsim_3particles_40000_ka_bin_3_FARICH_35mm_1000_no_trackers.root",
    num_of_files=10,
    uncertain_angle=False,
):
    datadir = "data"
    sipm_eff, PDE_wvs = init_sipm_eff()
    for key in sipm_eff.keys():
        sipm_eff[key] = sipm_eff[key] / 0.55414 * 0.38

    x_grid = np.arange(
        -3 * norm_r * np.sin(np.pi / 27),
        3 * norm_r * np.sin(np.pi / 27),
        SIPM_CELL_SIZE,
    )
    z_grid = np.arange(-1400, 1400, SIPM_CELL_SIZE)
    grid = (x_grid, z_grid, PDE_wvs)

    split_filepath = filepath_to_first.split("_1_")
    filepath_binned = os.path.join(
        datadir, f"{split_filepath[0]}_{1}_{split_filepath[1]}"
    )
    file_binned = uproot.open(filepath_binned)
    coordinates, true_direction_coordinates, intersections, ids = init_coords_for_field(
        file_binned, grid
    )
    if num_of_files > 1:
        for i in range(2, num_of_files + 1):
            filepath_binned = os.path.join(
                datadir, f"{split_filepath[0]}_{i}_{split_filepath[1]}"
            )
            file_binned = uproot.open(filepath_binned)
            coordinates_i, true_direction_coordinates_i, intersections_i, ids_i = (
                init_coords_for_field(file_binned, grid)
            )
            coordinates = np.concatenate((coordinates, coordinates_i), axis=0)
            true_direction_coordinates = np.concatenate(
                (true_direction_coordinates, true_direction_coordinates_i), axis=0
            )
            intersections = np.concatenate((intersections, intersections_i), axis=0)
            ids = np.concatenate((ids, ids_i), axis=0)

    idx_to_drop = []
    for i in range(coordinates.shape[0]):
        if coordinates[i][0].shape[0] == 0:
            idx_to_drop.append(i)

    coordinates = np.delete(coordinates, idx_to_drop, axis=0)
    true_direction_coordinates = np.delete(
        true_direction_coordinates, idx_to_drop, axis=0
    )
    intersections = np.delete(intersections, idx_to_drop, axis=0)
    ids = np.delete(ids, idx_to_drop)
    print(coordinates.shape)
    print(true_direction_coordinates.shape)

    true_direction_coordinates = (
        intersections
        / np.linalg.norm(intersections, axis=1)[:, None]
        * np.linalg.norm(true_direction_coordinates, axis=1)[:, None]
    )

    if uncertain_angle:
        uncertainty_introduction_to_direction(true_direction_coordinates)

    sipm_sim(coordinates, sipm_eff)
    for i, coord in enumerate(coordinates):
        if coord[0].shape[0] == 0:
            coord[0] = np.atleast_1d(np.array(intersections[i][0]))
            coord[1] = np.atleast_1d(np.array(intersections[i][1]))
            coord[2] = np.atleast_1d(np.array(intersections[i][2]))
            coord[3] = np.atleast_1d(np.array(450))
            coord[4] = np.atleast_1d(np.array(0.633))

    # may need to drop that deletion for the sake of keeping events
    # idx_to_drop = []
    # for i in range(coordinates.shape[0]):
    #     if coordinates[i][0].shape[0] == 0:
    #         idx_to_drop.append(i)
    # coordinates = np.delete(coordinates, idx_to_drop, axis=0)
    # true_direction_coordinates = np.delete(
    #     true_direction_coordinates, idx_to_drop, axis=0
    # )
    # print(coordinates.shape)
    # print(true_direction_coordinates.shape)

    main_angles = rotate_lines_for_decay(intersections)
    intersections = find_intersections_for_decay(intersections)
    rotate_events(coordinates, main_angles)
    move_events_to_grid(coordinates, grid)
    repeat_nums = np.array([coord[0].shape[0] for coord in coordinates])
    edf = pd.DataFrame(coordinates, columns=["x_c", "y_c", "z_c", "wv_c", "t_c"])

    unraveled_data = {col: [] for col in edf.columns}
    row_indices = []

    # Iterate over the DataFrame and unravel the arrays
    for i, row in edf.iterrows():
        max_length = max(len(row[col]) for col in edf.columns)
        for k in range(max_length):
            row_indices.append((i, k))
            for col in edf.columns:
                if k < len(row[col]):
                    unraveled_data[col].append(row[col][k])
                else:
                    unraveled_data[col].append(
                        np.nan
                    )  # Handle cases where arrays are of different lengths

    # Create a new DataFrame from the unraveled data
    unraveled_df = pd.DataFrame(unraveled_data)

    # Create a MultiIndex for the rows
    multi_index = pd.MultiIndex.from_tuples(row_indices, names=["entry", "subentry"])
    unraveled_df.index = multi_index

    edf = unraveled_df

    edf["x_i"] = np.repeat(intersections[:, 0], repeat_nums, axis=0)
    edf["z_i"] = np.repeat(intersections[:, 2], repeat_nums, axis=0)

    x = y = z = x3 = y3 = z3 = unraveled_data = row_indices = main_angles = wvs = (
        coordinates
    ) = file = coordinates_low = file_low = 0

    bdf = pd.DataFrame()
    gdf = pd.DataFrame()
    gdf["nhits"] = repeat_nums

    mu_mass = 105.65
    pi_mass = 139.57
    ka_mass = 493.68
    mass = np.array(
        [
            (
                mu_mass
                if np.abs(ids[i]) == 13
                else (ka_mass if np.abs(ids[i]) == 321 else pi_mass)
            )
            for i in range(ids.shape[0])
        ]
    )
    edf.rename(columns={"y_c": "tmp_c"}, inplace=True)
    edf.drop("wv_c", axis=1, inplace=True)
    edf.rename(columns={"z_c": "y_c", "z_i": "y_i"}, inplace=True)
    edf.rename(columns={"tmp_c": "z_c"}, inplace=True)
    # edf["z_c"] = np.zeros(edf.shape[0]) + 1000  # why 2000?
    # edf["mass"] = np.ones(edf.shape[0]) * mass
    edf["mass"] = np.repeat(
        mass,
        repeat_nums,
        axis=0,
    )
    edf["true_p"] = np.repeat(
        np.linalg.norm(true_direction_coordinates.astype("float"), axis=1) * 1000,
        repeat_nums,
        axis=0,
    )
    edf["beta"] = edf.true_p / np.sqrt(edf.mass**2 + edf.true_p**2)
    edf["x_p"] = np.zeros(edf.shape[0])
    edf["y_p"] = np.zeros(edf.shape[0])
    edf["z_p"] = np.zeros(edf.shape[0])
    edf["nx_p"] = np.repeat(
        (
            intersections
            / np.array(
                [
                    np.linalg.norm(intersections.astype("float"), axis=1)
                    for i in range(3)
                ]
            ).T
        ).astype("float")[:, 0],
        repeat_nums,
        axis=0,
    )
    edf["ny_p"] = np.repeat(
        (
            intersections
            / np.array(
                [
                    np.linalg.norm(intersections.astype("float"), axis=1)
                    for i in range(3)
                ]
            ).T
        ).astype("float")[:, 2],
        repeat_nums,
        axis=0,
    )
    edf["nz_p"] = np.repeat(
        (
            intersections
            / np.array(
                [
                    np.linalg.norm(intersections.astype("float"), axis=1)
                    for i in range(3)
                ]
            ).T
        ).astype("float")[:, 1],
        repeat_nums,
        axis=0,
    )

    true_direction_coordinates = repeat_nums = true_direction_coordinates_low = (
        intersections
    ) = 0

    return edf, bdf, gdf


def init_coords_for_field(file, grid):
    primary_pdgid = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.pdgId"].array()
    )
    farich_pdgid = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"]["FarichBarrelG4Hits.pdgId"].array()
    )
    shapes = np.array(
        [
            np.where(
                (farich_pdgid[i] != -22)
                & (farich_pdgid[i] != 11)
                & (primary_pdgid[i][0] in farich_pdgid[i])
            )[0].shape[0]
            for i in range(len(farich_pdgid))
        ]
    )
    good_events = np.where(shapes == 1)[0]
    primary_particle_idx = np.array(
        [
            np.where(farich_pdgid[good_events[i]] == primary_pdgid[good_events[i]])[0][
                0
            ]
            for i in range(len(good_events))
        ]
    )

    x = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.postStepPosition.x"
        ].array()
    )[good_events]
    y = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.postStepPosition.y"
        ].array()
    )[good_events]
    z = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.postStepPosition.z"
        ].array()
    )[good_events]
    wvs = (
        1239.841
        / np.array(
            file[file.keys()[0]]["FarichBarrelG4Hits"][
                "FarichBarrelG4Hits.energy"
            ].array()
        )
        * 1e-9
    )[good_events]
    t = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.localTime"
        ].array()
    )[good_events]

    farich_momentum_x = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.momentum.px"
        ].array()
    )[good_events]
    farich_momentum_y = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.momentum.py"
        ].array()
    )[good_events]
    farich_momentum_z = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.momentum.pz"
        ].array()
    )[good_events]
    farich_pdgid = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"]["FarichBarrelG4Hits.pdgId"].array()
    )[good_events]

    x3 = np.array(
        [
            farich_momentum_x[i][primary_particle_idx[i]]
            for i in range(farich_momentum_x.shape[0])
        ]
    )
    y3 = np.array(
        [
            farich_momentum_y[i][primary_particle_idx[i]]
            for i in range(farich_momentum_y.shape[0])
        ]
    )
    z3 = np.array(
        [
            farich_momentum_z[i][primary_particle_idx[i]]
            for i in range(farich_momentum_z.shape[0])
        ]
    )
    id = np.array(
        [farich_pdgid[i][primary_particle_idx[i]] for i in range(farich_pdgid.shape[0])]
    )

    xi = np.array([x[i][primary_particle_idx[i]] for i in range(x.shape[0])])
    yi = np.array([y[i][primary_particle_idx[i]] for i in range(y.shape[0])])
    zi = np.array([z[i][primary_particle_idx[i]] for i in range(z.shape[0])])

    x3 = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.p4.px"].array()
    )[good_events]
    y3 = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.p4.py"].array()
    )[good_events]
    z3 = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.p4.pz"].array()
    )[good_events]
    x3 = np.array([x3[i][0] for i in range(len(good_events))])
    y3 = np.array([y3[i][0] for i in range(len(good_events))])
    z3 = np.array([z3[i][0] for i in range(len(good_events))])

    true_direction_coordinates = np.column_stack((x3, y3, z3))
    intersections = np.stack((xi, yi, zi), axis=1)
    for i in range(len(wvs)):
        wvs[i] = lin_move_to_grid(wvs[i], grid[2])
    coordinates = np.column_stack((x, y, z, wvs, t))
    return coordinates, true_direction_coordinates, intersections, id


def init_coords_decay(
    file, grid, good_events, primary_particle_idx, primary_particle_in_primary_idx
):
    x = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.postStepPosition.x"
        ].array()
    )[good_events]
    y = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.postStepPosition.y"
        ].array()
    )[good_events]
    z = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.postStepPosition.z"
        ].array()
    )[good_events]
    wvs = (
        1239.841
        / np.array(
            file[file.keys()[0]]["FarichBarrelG4Hits"][
                "FarichBarrelG4Hits.energy"
            ].array()
        )
        * 1e-9
    )[good_events]
    t = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.localTime"
        ].array()
    )[good_events]

    farich_momentum_x = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.momentum.px"
        ].array()
    )[good_events]
    farich_momentum_y = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.momentum.py"
        ].array()
    )[good_events]
    farich_momentum_z = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"][
            "FarichBarrelG4Hits.momentum.pz"
        ].array()
    )[good_events]
    farich_pdgid = np.array(
        file[file.keys()[0]]["FarichBarrelG4Hits"]["FarichBarrelG4Hits.pdgId"].array()
    )[good_events]

    x3 = np.array(
        [
            farich_momentum_x[i][primary_particle_idx[i]]
            for i in range(farich_momentum_x.shape[0])
        ]
    )
    y3 = np.array(
        [
            farich_momentum_y[i][primary_particle_idx[i]]
            for i in range(farich_momentum_y.shape[0])
        ]
    )
    z3 = np.array(
        [
            farich_momentum_z[i][primary_particle_idx[i]]
            for i in range(farich_momentum_z.shape[0])
        ]
    )
    id = np.array(
        [farich_pdgid[i][primary_particle_idx[i]] for i in range(farich_pdgid.shape[0])]
    )

    xi = np.array([x[i][primary_particle_idx[i]] for i in range(x.shape[0])])
    yi = np.array([y[i][primary_particle_idx[i]] for i in range(y.shape[0])])
    zi = np.array([z[i][primary_particle_idx[i]] for i in range(z.shape[0])])

    x3 = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.p4.px"].array()
    )[good_events]
    y3 = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.p4.py"].array()
    )[good_events]
    z3 = np.array(
        file[file.keys()[0]]["allGenParticles"]["allGenParticles.core.p4.pz"].array()
    )[good_events]
    x3 = np.array(
        [x3[i][primary_particle_in_primary_idx[i]] for i in range(len(good_events))]
    )
    y3 = np.array(
        [y3[i][primary_particle_in_primary_idx[i]] for i in range(len(good_events))]
    )
    z3 = np.array(
        [z3[i][primary_particle_in_primary_idx[i]] for i in range(len(good_events))]
    )
    # Объединяем массивы, сохраняя структуру массивов
    true_direction_coordinates = np.stack((x3, y3, z3), axis=1)
    intersections = np.stack((xi, yi, zi), axis=1)
    for i in range(len(wvs)):
        wvs[i] = lin_move_to_grid(wvs[i], grid[2])
    coordinates = np.column_stack((x, y, z, wvs, t))
    return coordinates, true_direction_coordinates, intersections, id


def primary_particle_match_test(
    primary_particles_array, ids_to_check, ids_to_match, verbose=False
):
    for i in range(len(ids_to_check)):
        if primary_particles_array[ids_to_check[i]] != ids_to_match[i]:
            if verbose:
                print(
                    f"Expected {ids_to_match[i]} but got {primary_particles_array[ids_to_check[i]]} in {ids_to_check[i]}-th position"
                )
            return False
    return True


# def find_primary_in_farich(primary_particles_array, farich_particles_array, primary_id, pdg_id, verbose=False):
#     find_index_of_pdg_id = lambda arr: (idx[-1] if (idx := np.flatnonzero(arr == pdg_id)).size and (idx[-1] < arr.size) else False)
#     if primary_particles_array[primary_id] == pdg_id:
#         if find_index_of_pdg_id(farich_particles_array) is not False:
#             if primary_particles_array[primary_id] != farich_particles_array[find_index_ka(farich_pdgid[i])]:
#                 if verbose:
#                     print('Mismatch', i)
#                 mismatches.append(i)
#                 is_good = False
#         else:
#             if verbose:
#                 print('No ka in event', i)
#             is_good = False
#             no_ka.append(i)
#     if is_good:
#         good_events.append(i)
#         primary_particle_idx.append(find_index_ka(farich_pdgid[i]))
#         primary_particle_in_primary_idx.append(2)
#


def find_good_events_in_decay(primary_pdgid, farich_pdgid):
    i = 0
    empty_farich = 0
    important_particle_ind = 3
    mismatches = []
    good_events = []
    good_mu_events = []
    good_ka_events = []
    primary_particle_idx = []
    primary_particle_in_primary_idx = []
    no_mu_in_mu = []
    no_pi_in_pi = []
    no_ka = []
    # find_index = lambda arr: (idx[-1]+1 if (idx := np.flatnonzero(arr == -22)).size and (idx[-1]+1 < arr.size) else 0) # моржовый оператор :)
    find_index = lambda arr: (
        idx[-1] + 1
        if (idx := np.flatnonzero(np.isin(arr, [-22, -11]))).size
        and (idx[-1] + 1 < arr.size)
        else 0
    )
    find_index_mu = lambda arr: (
        idx[-1]
        if (idx := np.flatnonzero(arr == -13)).size and (idx[-1] < arr.size)
        else False
    )
    find_index_pi = lambda arr: (
        idx[-1]
        if (idx := np.flatnonzero(arr == 211)).size and (idx[-1] < arr.size)
        else False
    )
    find_index_ka = lambda arr: (
        idx[-1]
        if (idx := np.flatnonzero(arr == -321)).size and (idx[-1] < arr.size)
        else False
    )

    # need 2 rings per event now...

    for primary_particles in primary_pdgid:
        is_good = True

        if (
            farich_pdgid[i].shape[0] == 0
            or np.flatnonzero(farich_pdgid[i] + 22).shape[0] == 0
        ):
            empty_farich += 1
            is_good = False

        if is_good:
            is_good = primary_particle_match_test(
                primary_particles, [0, 1, 2], [30443, 421, -321], True
            )

        if primary_particles[2] == -321 and is_good:
            if (
                farich_pdgid[i].shape[0] != 0
                and np.flatnonzero(farich_pdgid[i] + 22).shape[0] != 0
            ):
                if find_index_ka(farich_pdgid[i]) is not False:
                    if (
                        primary_particles[2]
                        != farich_pdgid[i][find_index_ka(farich_pdgid[i])]
                    ):
                        print("Mismatch", i)
                        mismatches.append(i)
                        is_good = False
                else:
                    # print('No ka in event', i)
                    is_good = False
                    no_ka.append(i)
            if is_good:
                good_events.append(i)
                primary_particle_idx.append(find_index_ka(farich_pdgid[i]))
                primary_particle_in_primary_idx.append(2)

        if primary_particles[important_particle_ind] == -13 and is_good:
            if (
                farich_pdgid[i].shape[0] != 0
                and np.flatnonzero(farich_pdgid[i] + 22).shape[0] != 0
            ):
                if find_index_mu(farich_pdgid[i]) is not False:
                    if (
                        primary_particles[important_particle_ind]
                        != farich_pdgid[i][find_index_mu(farich_pdgid[i])]
                    ):
                        print("Mismatch", i)
                        mismatches.append(i)
                        is_good = False
                else:
                    # print('No mu in mu event', i)
                    is_good = False
                    no_mu_in_mu.append(i)
            if is_good:
                good_events.append(i)
                primary_particle_idx.append(find_index_mu(farich_pdgid[i]))
                primary_particle_in_primary_idx.append(important_particle_ind)
            if not is_good:
                if len(good_events) != 0:
                    good_events.pop()
                    primary_particle_idx.pop()
                    primary_particle_in_primary_idx.pop()

        elif primary_particles[important_particle_ind] == 211 and is_good:
            if (
                farich_pdgid[i].shape[0] != 0
                and np.flatnonzero(farich_pdgid[i] + 22).shape[0] != 0
            ):
                if find_index_pi(farich_pdgid[i]) is not False:
                    if (
                        primary_particles[important_particle_ind]
                        != farich_pdgid[i][find_index_mu(farich_pdgid[i])]
                    ):
                        print("Mismatch", i)
                        mismatches.append(i)
                        is_good = False
                else:
                    # print('No mu in mu event', i)
                    is_good = False
                    no_pi_in_pi.append(i)
            if is_good:
                good_events.append(i)
                primary_particle_idx.append(find_index_pi(farich_pdgid[i]))
                primary_particle_in_primary_idx.append(important_particle_ind)
            if not is_good:
                if len(good_events) != 0:
                    good_events.pop()
                    primary_particle_idx.pop()
                    primary_particle_in_primary_idx.pop()

        # if is_good:
        #     good_events.append(i)
        #     primary_particle_idx.append(find_index(farich_pdgid[i]))
        i += 1

    print("Empty Farich:", empty_farich)
    print("Mismatches: ", len(mismatches))
    print("Missing K: ", len(no_ka))
    print("Missing Mu in Mu event: ", len(no_mu_in_mu))
    print("Missing Pi in Pi event: ", len(no_pi_in_pi))
    print(
        "Full bad events: ",
        len(mismatches)
        + empty_farich
        + len(set(np.concatenate([no_mu_in_mu, no_ka, no_pi_in_pi]))),
    )
    print(
        "Good Events:",
        i
        - len(mismatches)
        - empty_farich
        - len(set(np.concatenate([no_mu_in_mu, no_ka, no_pi_in_pi]))),
    )
    print(len(good_events))
    print(len(set(good_events)))
    print(-len(set(good_events)) + len(good_events))

    return good_events, primary_particle_idx, primary_particle_in_primary_idx

def find_intersections_for_decay(full_coords):
    intersections = np.zeros((full_coords.shape[0], 3))
    zeros = np.zeros((1, 3))
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i] / 1000
        pca = PCA(n_components=1)

        # if event_coords[0].shape[0] == 1:
        # print(event_coords)
        event_coords = [[0, arr] for arr in event_coords]
        # print(event_coords)
        pca.fit(np.column_stack(event_coords))
        line_direction = pca.components_[0]
        line_point = pca.mean_

        # Calculate the parameter t for the intersection with the plane y=1000
        p_y = line_point[1]
        d_y = line_direction[1]
        t = (1000 - p_y) / d_y

        # Find the intersection point
        intersection_point = line_point + t * line_direction

        # print(f"Line direction: {line_direction}")
        # print(f"Point on the line: {line_point}")
        # print(f"Intersection point with the plane y=1000: {intersection_point}")
        for j in range(3):
            intersections[i][j] = intersection_point[j]
    return intersections

def rotate_lines_for_decay(full_coords):
    angles = np.zeros(full_coords.shape[0])
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        rotated_event_coords, angles[i] = rotate_line_for_decay(event_coords)

        for j in range(3):
            full_coords[i][j] = rotated_event_coords[:, j]
    return angles

# It only fixes angle problems, no reason not to use as main func
def rotate_line_for_decay(coords):
    angles = np.arctan2(coords[1], coords[0]) % (2 * np.pi)
    # print(angles)
    try:
        median_angle = angles
    except IndexError:
        print(angles)
        median_angle = np.median(angles)
    median_angle = lin_move_to_grid(np.array([median_angle]), plane_angles)
    # print(angles)
    angle_to_rotate = np.pi / 2 - median_angle
    # print(angle_to_rotate)
    x, y = rotate_point_on_line(coords, angle_to_rotate)
    return np.column_stack((x, y, coords[2])), median_angle


def create_edf_decay(
        filepath="fullsim_optical_2000_pi_bin_1_FARICH_35mm_no_no_trackers.root",
        good_events=[],
        primary_particle_idx=[],
        primary_particle_in_primary_idx=[],
        uncertain_angle=False,
):
    datadir = "data"
    sipm_eff, PDE_wvs = init_sipm_eff()
    for key in sipm_eff.keys():
        sipm_eff[key] = sipm_eff[key] / 0.55414 * 0.38

    x_grid = np.arange(
        -3 * norm_r * np.sin(np.pi / 27),
        3 * norm_r * np.sin(np.pi / 27),
        SIPM_CELL_SIZE,
    )
    z_grid = np.arange(-1400, 1400, SIPM_CELL_SIZE)
    grid = (x_grid, z_grid, PDE_wvs)

    decay_file = uproot.open(os.path.join(datadir, filepath))
    coordinates, true_direction_coordinates, intersections, ids = init_coords_decay(
        decay_file, grid, good_events, primary_particle_idx, primary_particle_in_primary_idx
    )

    idx_to_drop = []
    for i in range(coordinates.shape[0]):
        if coordinates[i][0].shape[0] == 0:
            idx_to_drop.append(i)

    coordinates = np.delete(coordinates, idx_to_drop, axis=0)
    true_direction_coordinates = np.delete(
        true_direction_coordinates, idx_to_drop, axis=0
    )
    intersections = np.delete(intersections, idx_to_drop, axis=0)
    ids = np.delete(ids, idx_to_drop)
    print(coordinates.shape)
    print(true_direction_coordinates.shape)

    true_direction_coordinates = intersections / np.linalg.norm(intersections, axis=1)[:, None] * np.linalg.norm(
        true_direction_coordinates, axis=1)[:, None]

    if uncertain_angle:
        uncertainty_introduction_to_direction(true_direction_coordinates)

    sipm_sim(coordinates, sipm_eff)
    for i, coord in enumerate(coordinates):
        # print(coord[0].shape)
        # print(coord)
        # if coord[0].shape[0] != 0:
        #     break
        if coord[0].shape[0] == 0:
            coord[0] = np.atleast_1d(np.array(intersections[i][0]))
            coord[1] = np.atleast_1d(np.array(intersections[i][1]))
            coord[2] = np.atleast_1d(np.array(intersections[i][2]))
            coord[3] = np.atleast_1d(np.array(450))
            coord[4] = np.atleast_1d(np.array(0.633))
    # idx_to_drop = []
    # for i in range(coordinates.shape[0]):
    #     if coordinates[i][0].shape[0] == 0:
    #         idx_to_drop.append(i)
    # coordinates = np.delete(coordinates, idx_to_drop, axis=0)
    # true_direction_coordinates = np.delete(
    #     true_direction_coordinates, idx_to_drop, axis=0
    # )
    # intersections = np.delete(intersections, idx_to_drop, axis=0)
    # ids = np.delete(ids, idx_to_drop)
    # print(coordinates.shape)
    # print(true_direction_coordinates.shape)

    main_angles = rotate_lines_for_decay(intersections)  # Rotates intersection points

    intersections = find_intersections_for_decay(
        intersections)  # May need to rewrite both to treat elements as scalasrs and to change reference point from 0

    rotate_events(coordinates, main_angles)  # There are events with extra rings somewhere around angle idx 13-16
    move_events_to_grid(coordinates, grid)
    repeat_nums = np.array([coord[0].shape[0] for coord in coordinates])
    edf = pd.DataFrame(coordinates, columns=["x_c", "y_c", "z_c", "wv_c", "t_c"])

    unraveled_data = {col: [] for col in edf.columns}
    row_indices = []

    # Iterate over the DataFrame and unravel the arrays
    for i, row in edf.iterrows():
        max_length = max(len(row[col]) for col in edf.columns)
        for k in range(max_length):
            row_indices.append((i, k))
            for col in edf.columns:
                if k < len(row[col]):
                    unraveled_data[col].append(row[col][k])
                else:
                    unraveled_data[col].append(
                        np.nan
                    )  # Handle cases where arrays are of different lengths

    # Create a new DataFrame from the unraveled data
    unraveled_df = pd.DataFrame(unraveled_data)

    # Create a MultiIndex for the rows
    multi_index = pd.MultiIndex.from_tuples(row_indices, names=["entry", "subentry"])
    unraveled_df.index = multi_index

    edf = unraveled_df

    edf["x_i"] = np.repeat(intersections[:, 0], repeat_nums, axis=0)
    edf["z_i"] = np.repeat(intersections[:, 2], repeat_nums, axis=0)

    x = y = z = x3 = y3 = z3 = unraveled_data = row_indices = wvs = coordinates = file = coordinates_low = file_low = 0  # = main_angles

    bdf = pd.DataFrame()
    gdf = pd.DataFrame()
    gdf["nhits"] = repeat_nums

    mu_mass = 105.65
    pi_mass = 139.57
    ka_mass = 493.68
    # mass = mu_mass if is_mu else (ka_mass if is_ka else pi_mass)
    mass = np.array(
        [mu_mass if ids[i] == -13 else (ka_mass if ids[i] == -321 else pi_mass) for i in range(ids.shape[0])])
    # edf.drop("y_c", axis=1, inplace=True)
    edf.rename(columns={"y_c": "tmp_c"}, inplace=True)
    edf.drop("wv_c", axis=1, inplace=True)
    edf.rename(columns={"z_c": "y_c", "z_i": "y_i"}, inplace=True)
    edf.rename(columns={"tmp_c": "z_c"}, inplace=True)
    # edf["z_c"] = np.zeros(edf.shape[0]) + 1000
    # edf["mass"] = np.ones(edf.shape[0]) * mass
    edf["mass"] = np.repeat(
        mass,
        repeat_nums,
        axis=0,
    )
    edf["true_p"] = np.repeat(
        np.linalg.norm(true_direction_coordinates.astype("float"), axis=1) * 1000,
        repeat_nums,
        axis=0,
    )
    edf["beta"] = edf.true_p / np.sqrt(edf.mass ** 2 + edf.true_p ** 2)
    edf["x_p"] = np.zeros(edf.shape[0])
    edf["y_p"] = np.zeros(edf.shape[0])
    edf["z_p"] = np.zeros(edf.shape[0])
    edf["nx_p"] = np.repeat(
        (
                intersections
                / np.array(
            [
                np.linalg.norm(intersections.astype("float"), axis=1)
                for i in range(3)
            ]
        ).T
        ).astype("float")[:, 0],
        repeat_nums,
        axis=0,
    )
    edf["ny_p"] = np.repeat(
        (
                intersections
                / np.array(
            [
                np.linalg.norm(intersections.astype("float"), axis=1)
                for i in range(3)
            ]
        ).T
        ).astype("float")[:, 2],
        repeat_nums,
        axis=0,
    )
    edf["nz_p"] = np.repeat(
        (
                intersections
                / np.array(
            [
                np.linalg.norm(intersections.astype("float"), axis=1)
                for i in range(3)
            ]
        ).T
        ).astype("float")[:, 1],
        repeat_nums,
        axis=0,
    )

    true_direction_coordinates = repeat_nums = true_direction_coordinates_low = mass = intersections = 0
    return edf, bdf, gdf, main_angles


def enforce_float32(df):
    return df.astype({col: np.float32 for col in df.select_dtypes(include=['float64']).columns})