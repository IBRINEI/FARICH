import plotly.graph_objects as go
from scipy.interpolate import griddata
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
plt.style.use('default')

SIPM_GEOMETRIC_EFFICIENCY = 0.85
SIPM_CELL_SIZE = 3.36
plane_angles = np.array([1.745319668152660597e-01, 4.072425337478433605e-01, 6.399531006804206612e-01,
                         8.726636676129979620e-01, 1.105374234545575263e+00, 1.338084801478152563e+00,
                         1.570795368410729864e+00, 1.803505935343307165e+00, 2.036216502275884466e+00,
                         2.268927069208461766e+00, 2.501637636141039067e+00, 2.734348203073616368e+00,
                         2.967058770006193225e+00, 3.199769336938770525e+00, 3.432479903871347826e+00,
                         3.665190470803925127e+00, 3.897901037736502428e+00, 4.130611604669079284e+00,
                         4.363322171601657473e+00, 4.596032738534233886e+00, 4.828743305466812075e+00,
                         5.061453872399388487e+00, 5.294164439331966676e+00, 5.526875006264543089e+00,
                         5.759585573197120389e+00, 5.992296140129697690e+00, 6.225006707062274991e+00])
norm_r = 1007.0091186826339


def plot_cyl(file, coords=None, transposed=False):
    def cylinder(r, h, a =0, nt=100, nv =50):
        """
        parametrize the cylinder of radius r, height h, base point a
        """
        theta = np.linspace(0, 2*np.pi, nt)
        v = np.linspace(a, a+h, nv )
        theta, v = np.meshgrid(theta, v)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = v
        return x, y, z

    colorscale = [[0, 'blue'],
                 [1, 'blue']]

    x1, y1, z1 = cylinder(1000, 2800, a=-1400)
    if transposed:
        z1, x1 = x1, z1
    cyl1 = go.Surface(x=x1, y=y1, z=z1,
                     colorscale = colorscale,
                     showscale=False,
                     opacity=0.05)

    fig = go.Figure()

    # colors = [f'rgba({255}, {int(255 * i / 9)}, 0, 0.6)' for i in range(10)]  # Adjust the colors as needed
    colors = [
        'rgba(255, 0, 0, 0.6)',   # Red
        'rgba(255, 165, 0, 0.6)', # Orange
        'rgba(255, 255, 0, 0.6)', # Yellow
        'rgba(0, 255, 0, 0.6)',   # Green
        'rgba(0, 0, 255, 0.6)',   # Blue
        'rgba(75, 0, 130, 0.6)',  # Indigo
        'rgba(128, 0, 128, 0.6)', # Violet
        'rgba(255, 99, 71, 0.6)', # Tomato
        'rgba(0, 128, 128, 0.6)', # Teal
        'rgba(128, 128, 0, 0.6)'  # Olive
    ]
    for num in range(10):
        if coords is not None:
            x = coords[num][0]
            y = coords[num][1]
            z = coords[num][2]
        else:
            x = np.array(file['events;1']['FarichBarrelG4Hits.postStepPosition.x'].array())[num]
            y = np.array(file['events;1']['FarichBarrelG4Hits.postStepPosition.y'].array())[num]
            z = np.array(file['events;1']['FarichBarrelG4Hits.postStepPosition.z'].array())[num]
        if transposed:
            z, x = x, z
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1, color=colors[num]), name=f'Ring {num+1}'))

        x = np.array(file['events;1']['DriftChamberG4Hits.postStepPosition.x'].array())[num]
        y = np.array(file['events;1']['DriftChamberG4Hits.postStepPosition.y'].array())[num]
        z = np.array(file['events;1']['DriftChamberG4Hits.postStepPosition.z'].array())[num]
        if transposed:
            z, x = x, z
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num]), name=f'Drift Chamber {num + 1}', line=dict(dash='dash')))

        # x = np.array(file['events;1']['SiStripG4Hits.postStepPosition.x'].array())[num]
        # y = np.array(file['events;1']['SiStripG4Hits.postStepPosition.y'].array())[num]
        # z = np.array(file['events;1']['SiStripG4Hits.postStepPosition.z'].array())[num]
        # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num]), name=f'Si Strips {num + 1}', line=dict(dash='solid')))

        x = np.array(file['events;1']['TPCG4Hits.postStepPosition.x'].array())[num]
        y = np.array(file['events;1']['TPCG4Hits.postStepPosition.y'].array())[num]
        z = np.array(file['events;1']['TPCG4Hits.postStepPosition.z'].array())[num]
        if transposed:
            z, x = x, z
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num]), name=f'TPC {num + 1}', line=dict(dash='solid')))


    fig.add_trace(cyl1)
    # Update the layout to add labels and a title
    scene = dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis', aspectmode='cube', xaxis=dict(range=[-1005, 1005]), yaxis=dict(range=[-1005, 1005]), zaxis=dict(range=[-1405, 1405]),)
    if transposed:
        scene = dict(zaxis_title='X Axis', yaxis_title='Y Axis', xaxis_title='Z Axis', aspectmode='cube', zaxis=dict(range=[-1005, 1005]), yaxis=dict(range=[-1005, 1005]), xaxis=dict(range=[-1405, 1405]),)
    fig.update_layout(
                        autosize=False,
                        width=1000,
                        height=1000,
                        scene=scene,
                        template='plotly_dark',   # DARK OR LIGHT
                        title='3D Scatter Plot')

    # Show the plot
    fig.show()


def plot_event(ev_coords, track_coords, tpc_coords, true_direction_coords, transposed=False, num=0):
    def cylinder(r, h, a =0, nt=100, nv =50):
        """
        parametrize the cylinder of radius r, height h, base point a
        """
        theta = np.linspace(0, 2*np.pi, nt)
        v = np.linspace(a, a+h, nv )
        theta, v = np.meshgrid(theta, v)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = v
        return x, y, z

    colorscale = [[0, 'blue'],
                 [1, 'blue']]

    x1, y1, z1 = cylinder(1000, 2800, a=-1400)
    if transposed:
        z1, x1 = x1, z1
    cyl1 = go.Surface(x=x1, y=y1, z=z1,
                     colorscale = colorscale,
                     showscale=False,
                     opacity=0.05)

    fig = go.Figure()

    # colors = [f'rgba({255}, {int(255 * i / 9)}, 0, 0.6)' for i in range(10)]  # Adjust the colors as needed
    colors = [
        'rgba(255, 0, 0, 0.6)',   # Red
        'rgba(255, 165, 0, 0.6)', # Orange
        'rgba(255, 255, 0, 0.6)', # Yellow
        'rgba(0, 255, 0, 0.6)',   # Green
        'rgba(0, 0, 255, 0.6)',   # Blue
        'rgba(75, 0, 130, 0.6)',  # Indigo
        'rgba(128, 0, 128, 0.6)', # Violet
        'rgba(255, 99, 71, 0.6)', # Tomato
        'rgba(0, 128, 128, 0.6)', # Teal
        'rgba(128, 128, 0, 0.6)'  # Olive
    ]
    x = ev_coords[0]
    y = ev_coords[1]
    z = ev_coords[2]

    if transposed:
        z, x = x, z
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1, color=colors[num]),
                               name=f'Ring {num+1}'))

    if track_coords is not None:
        x = track_coords[0]
        y = track_coords[1]
        z = track_coords[2]
        if transposed:
            z, x = x, z
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num + 1]),
                                   name=f'Drift Chamber {num + 1}', line=dict(dash='dash')))

    if tpc_coords is not None:
        x = tpc_coords[0]
        y = tpc_coords[1]
        z = tpc_coords[2]
        if transposed:
            z, x = x, z
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num + 2]),
                                   name=f'TPC {num + 1}', line=dict(dash='solid')))

    if true_direction_coords is not None:
        x = true_direction_coords[0]
        y = true_direction_coords[1]
        z = true_direction_coords[2]
        if transposed:
            z, x = x, z
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num + 3]),
                                   name=f'True Direction {num + 1}', line=dict(dash='solid')))

    fig.add_trace(cyl1)
    # Update the layout to add labels and a title
    scene = dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis', aspectmode='cube',
                 xaxis=dict(range=[-1005, 1005]), yaxis=dict(range=[-1005, 1005]), zaxis=dict(range=[-1405, 1405]),)
    if transposed:
        scene = dict(zaxis_title='X Axis', yaxis_title='Y Axis', xaxis_title='Z Axis', aspectmode='cube',
                     zaxis=dict(range=[-1005, 1005]), yaxis=dict(range=[-1005, 1005]), xaxis=dict(range=[-1405, 1405]),)
    fig.update_layout(
                        autosize=False,
                        width=1000,
                        height=1000,
                        scene=scene,
                        template='plotly_dark',   # DARK OR LIGHT
                        title='3D Scatter Plot')

    # Show the plot
    fig.show()


def sipm_sim(full_coords, sipm_eff):
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        probs = np.array([sipm_eff[wv] for wv in event_coords[3]]) * SIPM_GEOMETRIC_EFFICIENCY
        random_nums = np.random.random(len(probs))
        idx = np.where(probs - random_nums > 0)[0]
        for j in range(4):
            full_coords[i][j] = event_coords[j][idx]


def lin_move_to_grid(coord, grid):
    distances = cdist(coord[:, np.newaxis], grid[:, np.newaxis], metric='euclidean')
    closest_indices = np.argmin(distances, axis=1)
    # Use the indices to get the closest elements from second_array
    closest_points = grid[closest_indices]
    return closest_points


def rotate_point(point, angle):
    x, y, _, __ = point
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    new_x = x * cos_theta - y * sin_theta
    # new_y = x * sin_theta + y * cos_theta
    return new_x


def rotate_event(coords, main_angle):
    angles = np.arctan2(coords[1], coords[0]) % (2 * np.pi)
    angles = lin_move_to_grid(angles, plane_angles)
    idx_to_shift = ((angles - main_angle) / 0.2327)
    idx_to_shift = np.array([round(idx) for idx in idx_to_shift])
    # print(idx_to_shift)
    # TODO: fix stuff when on edge of 2pi
    angle_to_rotate = np.pi/2 - angles
    x = rotate_point(coords, angle_to_rotate) - 2 * idx_to_shift * norm_r * np.sin(np.pi / 27)
    return np.column_stack((x, np.zeros_like(x) + 1000, coords[2], coords[3]))


def rotate_events(full_coords, main_angles):
    for i in range(full_coords.shape[0]):
        event_coords = full_coords[i]
        rotated_event_coords = rotate_event(event_coords, main_angles[i])

        for j in range(4):
            full_coords[i][j] = rotated_event_coords[:,j]


def move_points_to_grid(points, grid_points):
    x_moved = lin_move_to_grid(points[0], grid_points[0])
    z_moved = lin_move_to_grid(points[2], grid_points[1])
    # wv_moved = lin_move_to_grid(points[:,3], grid_points[2])

    return np.column_stack((x_moved, points[1], z_moved, points[3]))


def remove_duplicate_points(points):
    # Find unique pairs of elements in both arrays
    unique_pairs = set(zip(points[0], points[2]))

    # Find indices of elements forming unique pairs
    indices_to_remove = []
    for pair in unique_pairs:
        indices = np.where((points[0] == pair[0]) & (points[2] == pair[1]))[0]
        if len(indices) > 1:  # If the pair occurs more than once
            indices_to_remove.extend(indices[1:])

    return np.column_stack((np.delete(points[0], indices_to_remove), np.delete(points[1], indices_to_remove),
                            np.delete(points[2], indices_to_remove), np.delete(points[3], indices_to_remove)))


def move_events_to_grid(full_coords, grid_points):
    for i in range(full_coords.shape[0]):

        event_coords = full_coords[i]

        grid_event_coords = move_points_to_grid(event_coords, grid_points)
        # print(grid_event_coords.shape)
        grid_set_event_coords = remove_duplicate_points(grid_event_coords.T)

        for j in range(4):
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
    angle_to_rotate = (np.pi / 2 - median_angle)
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
            event_coords = [np.insert(arr, 0, 0.) for arr in event_coords]

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
    return edf[(abs(edf['x_c'] - edf['x_i']) <= 220) & (abs(edf['y_c'] - edf['y_i']) <= 220)]


def planeRecalculation(edf: pd.DataFrame, idf: pd.DataFrame):
    R = edf[['x_c', 'y_c', 'z_c']].to_numpy()
    R_i = edf[['x_i', 'y_i', 'z_c']].to_numpy()
    N = edf[['nx_p', 'ny_p', 'nz_p']].to_numpy()
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
    edf['t_c'] = edf['t_c'] + np.sign(R_new[:, 2] - R[:, 2]) * t_dif

    edf['recalculated_x'] = R_new[:, 0]
    edf['recalculated_y'] = R_new[:, 1]
    edf['recalculated_z'] = R_new[:, 2]


def planeRotation(edf: pd.DataFrame):
    R = edf[['recalculated_x', 'recalculated_y', 'recalculated_z']].to_numpy()
    R_i = edf[['x_i', 'y_i', 'z_c']].to_numpy()
    N = edf[['nx_p', 'ny_p', 'nz_p']].to_numpy() # N
    M = np.array([0, 0, 1])                           # M
    c = np.dot(N, M) / (np.linalg.norm(M) * np.linalg.norm(N, axis=1))
    axis = np.cross(N, np.broadcast_to(M, (N.shape[0], 3))) / np.linalg.norm(np.cross(N, np.broadcast_to(M, (N.shape[0], 3))), axis=1, keepdims=True)
    x, y, z = axis.T
    s = np.sqrt(1-c*c)
    C = 1-c
    rmat = np.array([
      [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
      [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
      [z*x*C-y*s, z*y*C+x*s, z*z*C+c]])
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
    maskR_i = np.logical_or(abs(rotated_R_i[:, 0]) >= 5000, abs(rotated_R_i[:, 1]) >= 5000)
    rotated_R[maskR] = [5000, 5000, 0]
    rotated_R_i[maskR_i] = [5000, 5000, 0]
    rotated_n = (rotated_R_i - edf[['x_p', 'y_p', 'z_p']].to_numpy()) / np.linalg.norm(rotated_R_i - edf[['x_p', 'y_p', 'z_p']].to_numpy(), axis=1, keepdims=True)
    edf['rotated_x'] = rotated_R[:,0]
    edf['rotated_y'] = rotated_R[:,1]
    edf['rotated_z'] = rotated_R[:,2]
    edf['rotated_x_i'] = rotated_R_i[:,0]
    edf['rotated_y_i'] = rotated_R_i[:,1]
    edf['rotated_z_i'] = rotated_R_i[:,2]
    edf['rotated_nx_p'] = rotated_n[:,0]
    edf['rotated_ny_p'] = rotated_n[:,1]
    edf['rotated_nz_p'] = rotated_n[:,2]


def applySecondSpaceCut(edf: pd.DataFrame) -> pd.DataFrame:
    return edf[(abs(edf['rotated_x'] - edf['rotated_x_i']) <= 220) & (abs(edf['rotated_y'] - edf['rotated_y_i']) <= 220)]


def edf_to_bdf(edf_col: pd.Series, bdf: pd.DataFrame):
    to_bdf = [sub.iloc[0] for _, sub in edf_col.groupby(level=0)]
    bdf[edf_col.name] = pd.Series(to_bdf)


def primaryDirectionRecalculation(edf: pd.DataFrame):
    N = edf.loc[:, ('nx_p', 'ny_p', 'nz_p')].to_numpy()
    M = []
    theta_ps = []
    for n in N:
        M.append([0, 0, 1])
        dot_product = np.dot(n, [0, 0, 1.])

        # Calculate the magnitudes (norms)
        mag_vector = np.linalg.norm(n)
        mag_z_axis = np.linalg.norm([0, 0, 1.])

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
    edf['recalculated_nx_p'] = M[:, 0]
    edf['recalculated_ny_p'] = M[:, 1]
    edf['recalculated_nz_p'] = M[:, 2]
    edf['theta_p'] = theta_ps


def recoAngles(edf: pd.DataFrame, idf: pd.DataFrame, rotation_mode=False):
    '''
    Геометрическая реконструкция углов фотонов относительно направления частицы.
    Из координат срабатываний и частиц вычисляются углы theta_c, phi_c и время вылета фотонов t_c_orig и добавляются к edf.
    '''
    r0 = edf.loc[:, ('x_p', 'y_p', 'z_p')].to_numpy()
    if rotation_mode:
        r = edf.loc[:, ('rotated_x', 'rotated_y', 'rotated_z')].to_numpy()
    # n0 = edf.loc[:, ('rotated_nx_p', 'rotated_ny_p', 'rotated_nz_p')].to_numpy()
        n0 = edf.loc[:, ('recalculated_nx_p', 'recalculated_ny_p', 'recalculated_nz_p')].to_numpy()
    else:
        r  = edf.loc[:, ('x_c', 'y_c', 'z_c')].to_numpy()
        n0 = edf.loc[:, ('nx_p', 'ny_p', 'nz_p')].to_numpy()

    speedOfLight_mmperns = 299.792458 # мм/нс

    # расстояние от радиатора до детектора
    dist = float(idf['distance'])

    # толщина радиатора
    W = float(idf['W'])

    # расстояние от точки вылета частицы до входной плоскости радиатора
    rad_pos = float(idf['zdis'])

    # полное число срабатываний
    N = edf.shape[0]

    # координаты точки пересечения трека с ФД
    if not rotation_mode:
        y_i = r0[:,1] + (dist + rad_pos) * n0[:,1] / n0[:,2] # r0[:,1] + (dist + W + rad_pos) * n0[:,1] / n0[:,2]   #   r0[:,1] + (dist + rad_pos) * n0[:,1] / n0[:,2]
        x_i = r0[:,0] + (y_i - r0[:,1]) * n0[:,0] / n0[:,1] # r0[:,0] + (y_i - r0[:,1]) * n0[:,0] / n0[:,1]    #     r0[:,0] + (dist + rad_pos) * n0[:,0] / n0[:,2]
        edf['x_i'] = x_i
        edf['y_i'] = y_i
        edf['r_p_c'] = np.sqrt((r0[:,0] - x_i) ** 2 + (r0[:,1] - y_i) ** 2 + (r0[:,2] - r[:,2]) ** 2)
        edf['r_c'] = np.sqrt((x_i - edf['x_c']) ** 2 + (y_i - edf['y_c']) ** 2)

    if rotation_mode:
        n_mean = float(idf['n_mean'])

        edf['rotated_r_c'] = np.sqrt((edf['rotated_x_i'] - edf['rotated_x']) ** 2 + (edf['rotated_y_i'] - edf['rotated_y']) ** 2)

        rotated_r_c = edf['rotated_r_c'].to_numpy()
        # r_p_c = edf['r_p_c'].to_numpy()
        beta = edf['beta'].to_numpy()
        r_p_c = dist # or + W/2 ???


    # avg_betas = []
    # for _, subentry in edf['beta_from_true_r'].groupby(level=0):
    #   avg_beta = subentry.mean()
    #   for __ in subentry:
    #     avg_betas.append(avg_beta)
    # edf['beta_from_true_r_mean'] = avg_betas
    # косинусы и синусы сферических углов направления частицы
    costheta, sintheta = n0[:,2], np.sqrt(n0[:,0]**2+n0[:,1]**2)
    phi = np.arctan2(n0[:,1], n0[:,0])
    cosphi, sinphi = np.cos(phi), np.sin(phi)

    # номинальная точка вылета фотонов
    ro = r0 + (W/2+rad_pos)/n0[:,2].reshape(N,1)*n0

    """
    Преобразование в СК частицы
    𝑢𝑥 = cos 𝜃(𝑣𝑥 cos 𝜙 + 𝑣𝑦 sin 𝜙) − 𝑣𝑧 sin 𝜃,
    𝑢𝑦 = −𝑣𝑥 sin 𝜙 + 𝑣𝑦 cos 𝜙,
    𝑢𝑧 = sin 𝜃(𝑣𝑥 cos 𝜙 + 𝑣𝑦 sin 𝜙) + 𝑣𝑧 cos 𝜃.
    """

    # вектор направления фотона в лабораторной СК
    s = (r-ro)
    snorm = np.linalg.norm(s, axis=1, keepdims=True)
    v = s / snorm
    if not rotation_mode:
        edf['t_c_orig'] = edf['t_c'] - (snorm / speedOfLight_mmperns).reshape(N)

    # освобождение памяти при необходимости
    #del r0, n0, ro, r, s

    U = np.stack((np.stack((costheta*cosphi, costheta*sinphi, -sintheta)),
                np.stack((-sinphi,         cosphi,          np.full(N, 0.))),
                np.stack((sintheta*cosphi, sintheta*sinphi, costheta)))).transpose(2,0,1)

    # единичный вектор направления фотона в СК частицы
    u = (U @ v.reshape(N,3,1)).reshape(N,3)

    # сферические углы фотона в СК частицы
    if rotation_mode:
        edf['rotated_theta_c'] = np.arccos(u[:,2])
        edf['rotated_phi_c'] = np.arctan2(-u[:,1], -u[:,0])
    else:
        edf['theta_c'] = np.arccos(u[:,2])
        edf['phi_c'] = np.arctan2(-u[:,1], -u[:,0])


def local_sum_2d(event, r_slices, t_slices, square_counts, max_index, n, m, timestep, t_window_width, method='N/r'):
    cut_event = event[(event.t_c <= np.clip(t_slices[max_index[1]] + t_window_width + timestep * m, 0, 10)) & (event.t_c >= np.clip(t_slices[max_index[1]] - timestep * m, 0, 10)) &
                    (event.rotated_r_c <= r_slices[max_index[0] + n]) & (event.rotated_r_c >= r_slices[max_index[0] - n])]
    return np.mean(cut_event.rotated_r_c)


def local_weighed_sum_2d(r_slices, t_slices, square_counts, max_index, n, m, method='N/r'):
    arr = np.mean(square_counts[max_index[0] - n:max_index[0] + n + 1, np.clip(max_index[1] - m, 0, 50):np.clip(max_index[1] + m + 1, 0, 50)], axis=1)
    if method == 'N/r':
        sum_arr = r_slices[max_index[0] - n:max_index[0] + n + 1] ** 2 * arr
        den_arr = r_slices[max_index[0] - n:max_index[0] + n + 1] * arr
    else:
        sum_arr = r_slices[max_index[0] - n:max_index[0] + n + 1] * arr
        den_arr = arr

    weighted_sum = np.sum(sum_arr)
    weighted_den = np.sum(den_arr)

    return weighted_sum / weighted_den


def local_weighed_sum(r_slices, counts, max_index, n, method='N/r'):
    if method == 'N/r':
        sum_arr = r_slices[max_index - n:max_index + n + 1] ** 2 * counts[max_index - n:max_index + n + 1]
        den_arr = r_slices[max_index - n:max_index + n + 1] * counts[max_index - n:max_index + n + 1]
    else:
        sum_arr = r_slices[max_index - n:max_index + n + 1] * counts[max_index - n:max_index + n + 1]
        den_arr = counts[max_index - n:max_index + n + 1]

    weighted_sum = np.sum(sum_arr)
    weighted_den = np.sum(den_arr)

    return weighted_sum / weighted_den


def pol(x, a, b, c):
    return a * np.exp((x - b) ** 2 / c ** 2)


def pol2(x, p0, p1, p2):
    return p0 + p1 * x + p2 * x ** 2


def d3pol2(X, p0, p1, p2, q0, q1, q2, k0, k1, k2):
    r, theta = X
    # return pol(r, pol2(theta, p0, p1, p2), pol2(theta, q0, q1, q2), pol2(theta, k0, k1, k2))
    return pol(r, p0 + p1 * theta + p2 * theta ** 2, q0 + q1 * theta + q2 * theta ** 2, k0 + k1 * theta + k2 * theta ** 2)


def momentum_from_beta(beta, mass):
    return mass * beta / np.sqrt(1 - beta ** 2)


def momentum_pol(x, a, b, c):
    return momentum_from_beta(pol(x, a, b, c), 139.57)


def momentum_d3pol2(X, p0, p1, p2, q0, q1, q2, k0, k1, k2):
    return momentum_from_beta(d3pol2(X, p0, p1, p2, q0, q1, q2, k0, k1, k2), 139.57)


def rSlidingWindowIntro(edf: pd.DataFrame, idf: pd.DataFrame, bdf: pd.DataFrame, avg_sigmas: tuple, avg_t_sigmas: tuple, step: float, method='N/r', cal_arr=False, t_window_width=2,
                        r_width_factor=2, t_width_factor=8, full_width_t_hist = False, num_of_groups=5):
    r_r_c = edf['rotated_r_c']
    time_step = float(t_window_width) / t_width_factor
    all_avgs = np.array(r_r_c.groupby(level=0).transform('mean').to_list()).ravel()
    all_dists = np.abs(r_r_c - all_avgs)
    all_sigms = np.array(r_r_c.groupby(level=0).transform('std').to_list()).ravel()

    edf['mean_rotated_r_c'] = all_avgs
    edf['dist_from_mean_rotated_r_c'] = all_dists
    edf['rotated_r_c_sigm'] = all_sigms

    # Compute beta_step and r_step using NumPy functions
    beta_step = np.ptp(edf['beta'].values) # не факт что нужно values

    # Compute beta_intervals using NumPy linspace function
    num_of_groups = num_of_groups
    beta_intervals = np.linspace(edf['beta'].min(), edf['beta'].max(), num=num_of_groups)

    # Compute beta_group_to_bdf and  using NumPy operations
    beta_group = np.floor((num_of_groups * edf['beta'] + max(edf['beta']) - (num_of_groups + 1) * min(edf['beta'])) / beta_step).values

    edf['beta_group'] = beta_group

    edf_to_bdf(edf.beta_group, bdf)

    edf_to_bdf(edf.theta_p, bdf)
    bdf['cos_theta_p'] = np.cos(bdf['theta_p'])
    # edf_to_bdf(edf.signal_counts, bdf)
    edf_to_bdf(edf.beta, bdf)


def calculateSignalCounts(edf: pd.DataFrame, bdf: pd.DataFrame):
    signal_counts = edf['signal'].groupby(level=0).sum()
    bdf['signal_counts'] = signal_counts.values
    edf['signal_counts'] = edf.signal.groupby(level=0).transform('sum').values


def rSlidingWindowLoop1(edf: pd.DataFrame, idf: pd.DataFrame, bdf: pd.DataFrame, avg_sigmas: tuple, avg_t_sigmas: tuple, step: float, method='N/r', cal_arr=False, t_window_width=2,
                        r_width_factor=2, t_width_factor=8, full_width_t_hist=True, weighed=True):
    mean_cos_theta_p = 0.8535536229610443
    param_step = step
    step = param_step / r_width_factor
    time_step = float(t_window_width) / t_width_factor

    r_slices = np.arange(0, 800, step=step)
    t_slices = np.arange(0, 15, step=time_step)
    phi_slices = np.array([-np.pi, -2.013, -0.671, 0, 0.671, 2.013, np.pi])

    n_sigmas = np.ptp(avg_sigmas)
    t_sigmas = np.ptp(avg_t_sigmas)
    # all_counts_to_edf = np.zeros((n_sigmas, len(edf)))
    all_counts_to_edf = np.zeros((n_sigmas, len(edf)))
    all_calculated_r = np.zeros((n_sigmas, len(edf)))
    all_calculated_r_from_2d = np.zeros((t_sigmas, n_sigmas, len(edf)))
    cur_ind = 0
    for i, (entry, subentry) in enumerate(edf[['rotated_r_c', 't_c', 'rotated_phi_c', 'theta_p']].groupby(level=0)):
        if np.cos(subentry.theta_p).iat[0] >= mean_cos_theta_p:
            step = param_step / (r_width_factor + 1)
            r_slices = np.arange(0, 800, step=step)
        else:
            step = param_step / r_width_factor
            r_slices = np.arange(0, 800, step=step)

        counts = np.zeros(r_slices.shape)
        square_counts = np.zeros(shape=(r_slices.shape[0], t_slices.shape[0]))

        mask = np.logical_and(subentry.rotated_r_c >= 16, subentry.rotated_r_c <= 80)
        rotated_r_c = subentry.rotated_r_c[mask]
        t_c = subentry.t_c[mask]
        rotated_phi_c = subentry.rotated_phi_c[mask]

        counts, _ = np.histogram(rotated_r_c, bins=r_slices)
        square_counts, _, __ = np.histogram2d(rotated_r_c, t_c, bins=(r_slices, t_slices))

        if method == 'N/r':
            counts = np.divide(np.add(counts[:-1], counts[1:]), r_slices[1:-1])
            shift = 0
            square_counts[:-1, :] = np.divide(np.add(square_counts[:-1, :], square_counts[1:, :]), r_slices[1:-1-shift*2, np.newaxis])

        if full_width_t_hist:
            square_counts_but_last = sum([square_counts[:, it : -t_width_factor + 1 + it] for it in range(t_width_factor - 1)])
            square_counts = np.add(square_counts_but_last, square_counts[:, t_width_factor - 1:])

        max_index = np.argmax(counts)

        max_index_2d = np.unravel_index(np.argmax(square_counts), square_counts.shape)

        for j in range(n_sigmas):
            all_counts_to_edf[j][cur_ind:subentry.shape[0] + cur_ind] = counts[np.floor_divide(subentry.rotated_r_c, step).astype(int)] # fixed
            # avg_r_from_slices = local_weighed_sum(r_slices, counts, max_index, j + avg_sigmas[0], method)
            # all_calculated_r[j, cur_ind:subentry.shape[0] + cur_ind] = np.repeat(avg_r_from_slices, subentry.shape[0])
            for t in range(t_sigmas):
                if weighed:
                    avg_r_from_2d_slices = local_weighed_sum_2d(r_slices, t_slices, square_counts, max_index_2d, j + avg_sigmas[0], t + avg_t_sigmas[0])
                else:
                    avg_r_from_2d_slices = local_sum_2d(subentry, r_slices, t_slices, square_counts, max_index_2d, j + avg_sigmas[0], t + avg_t_sigmas[0], t_window_width=t_window_width, timestep=time_step)

                all_calculated_r_from_2d[t, j, cur_ind:subentry.shape[0] + cur_ind] = np.repeat(avg_r_from_2d_slices, subentry.shape[0])

        cur_ind += subentry.shape[0]
    for j in range(n_sigmas):
        edf[f'slice_counts_{j + avg_sigmas[0]}_sigms'] = all_counts_to_edf[j]
        # edf[f'unfixed_calculated_r_{j + avg_sigmas[0]}_sigms'] = all_calculated_r[j, :]
        for t in range(t_sigmas):
            edf[f'unfixed_calculated_r_2d_{j + avg_sigmas[0]}_rsigms_{t + avg_t_sigmas[0]}_tsigms'] = all_calculated_r_from_2d[t, j, :]
            edf_to_bdf(edf[f'unfixed_calculated_r_2d_{j + avg_sigmas[0]}_rsigms_{t + avg_t_sigmas[0]}_tsigms'], bdf)


def rSlidingWindowLoop2(edf: pd.DataFrame, idf: pd.DataFrame, bdf: pd.DataFrame, avg_sigmas: tuple, avg_t_sigmas: tuple,
                        step: float, method='N/r', cal_arr=False, t_window_width=2,
                        r_width_factor=2, t_width_factor=8, full_width_t_hist=False, param_fit=False,
                        calibration_func=pol, param_calibration_func=d3pol2,
                        target_variable='momentum', target_angle='theta_p', num_of_theta_intervals=11):

  # cal_arr = np.array([np.array([np.array(y) for y in x]) for x in cal_arr])
    error_counter = 0

    bdf['cos_theta_p'] = np.cos(bdf['theta_p'])
    edf['cos_theta_p'] = np.cos(edf['theta_p'])
    theta_interval = np.ptp(bdf[target_angle]) / (num_of_theta_intervals - 1)
    theta_min = min(bdf[target_angle])
    theta_max = max(bdf[target_angle])

    for n_sigms in range(*avg_sigmas):
        for t_sigms in range(*avg_t_sigmas):
            meas_betas = np.zeros(edf.shape[0])
            cur_ind = 0
            # meas_betas = []
            for entry, subentry in edf[f'unfixed_calculated_r_2d_{n_sigms}_rsigms_{t_sigms}_tsigms'].groupby(level=0):
                if param_fit:
                    angle = edf[target_angle].loc[entry].iloc[0]
                    # meas_beta = pol(subentry.iloc[0], pol2(angle, *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][0]),
                    #               pol2(angle, *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][1]),
                    #               pol2(angle, *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][2]))
                    meas_beta = param_calibration_func(subentry.iloc[0],
                                                       *cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]])
                else:
                    if edf[target_angle].loc[entry].iloc[0] != theta_max:
                        try:
                            # meas_beta = pol(subentry.iloc[0], *(cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][(np.floor(((edf.cos_theta_p[entry].iloc[0]) - theta_min) / theta_interval)).astype(int)]))
                            meas_beta = calibration_func(subentry.iloc[0], *(cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][(np.floor(((edf[target_angle].loc[entry].iloc[0]) - theta_min) / theta_interval)).astype(int)]))
                        except IndexError:
                            print(edf.cos_theta_p[entry].iloc[0])
                            print((np.floor(((edf.cos_theta_p[entry].iloc[0]) - theta_min) / theta_interval)).astype(int))
                            meas_beta = 0
                            error_counter += 1
                    else:
                        # meas_beta = pol(subentry.iloc[0], *(cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][9]))
                        meas_beta = calibration_func(subentry.iloc[0], *(cal_arr[n_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][num_of_theta_intervals - 2]))
                meas_betas[cur_ind: subentry.shape[0] + cur_ind] = np.repeat(meas_beta, subentry.shape[0])
                cur_ind += subentry.shape[0]
            edf[f'beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms'] = meas_betas
            edf[f'delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms'] = edf[f'beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms'] - edf['beta']
            edf[f'eps_beta_{n_sigms}_rsigms_{t_sigms}_tsigms'] = edf[f'delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms'] / edf['beta'] * 100

            edf_to_bdf(edf[f'beta_from_calc_r_{n_sigms}_rsigms_{t_sigms}_tsigms'], bdf)
            edf_to_bdf(edf[f'delta_beta_{n_sigms}_rsigms_{t_sigms}_tsigms'], bdf)
            edf_to_bdf(edf[f'eps_beta_{n_sigms}_rsigms_{t_sigms}_tsigms'], bdf)
            print(error_counter)


def save_calibration_plot(fig, dir_to_save, deg_lim, r_sigms, avg_t_sigmas):
    filename = f'rsigm={r_sigms}_t_sigms={avg_t_sigmas[0]}-{avg_t_sigmas[-1] - 1}'
    if deg_lim:
        filename += '_10deg'
    if dir_to_save != '':
        fig.savefig(os.path.join('calibrations_barrel', dir_to_save, f'{filename}'))
        plt.close(fig)


def plot_calibration(t_bdf: pd.DataFrame, chosen_column: str, t_sigms: int, momentum_min, momentum_max, rs, pol_param,
                     chi2, theta_interval_index, fig, axs, avg_sigmas, avg_t_sigmas, target_variable, target_angle,
                     calibration_func):
    if t_sigms - avg_t_sigmas[0] != 0:
        for_colorbar = axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].hist2d(
            t_bdf[chosen_column], t_bdf[target_variable], bins=70, range=((0, 80), (momentum_min, momentum_max)))
        fig.colorbar(for_colorbar[3], ax=axs[theta_interval_index, t_sigms - avg_t_sigmas[0]])
    else:
        for_colorbar = axs[theta_interval_index].hist2d(t_bdf[chosen_column], t_bdf[target_variable], bins=70,
                                                        range=((0, 80), (momentum_min, momentum_max)))
        fig.colorbar(for_colorbar[3], ax=axs[theta_interval_index])
    if t_sigms - avg_t_sigmas[0] != 0:
        # axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].plot(rs, calibration_func(rs, *pol_param), label=r'$\chi^2$ = '+ str(chi2), c='r')
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_xlim((0, 90))
        # axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_ylim((0.955, momentum_max))
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_ylim((momentum_min, momentum_max))
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_xlabel(r'$R_{reco}$, mm')
        axs[theta_interval_index, t_sigms - avg_t_sigmas[0]].set_ylabel(r'$target_{true}$')
    else:
        axs[theta_interval_index].plot(rs, calibration_func(rs, *pol_param), label=r'$\chi^2$ = ' + str(chi2), c='r')
        axs[theta_interval_index].set_xlim((0, 90))
        # axs[theta_interval_index].set_ylim((0.955, momentum_max))
        axs[theta_interval_index].set_ylim((momentum_min, momentum_max))
        axs[theta_interval_index].set_xlabel(r'$R_{reco}$, mm')
        axs[theta_interval_index].set_ylabel(r'target_{true}$')


def calibration_loop(bdf: pd.DataFrame, chosen_column: str, r_sigms: int, t_sigms: int, num_of_theta_intervals: int,
                     to_return_unbinned: np.ndarray, errs_tmp: np.ndarray, fig, axs, avg_sigmas, avg_t_sigmas,
                     target_variable, target_angle, calibration_func, p0):
    theta_p_max = max(bdf[target_angle])
    theta_p_min = min(bdf[target_angle])
    momentum_min = min(bdf[target_variable])
    momentum_max = max(bdf[target_variable])

    theta_intervals = np.linspace(theta_p_min, theta_p_max, num=num_of_theta_intervals)
    theta_dif = (theta_intervals[1:] + theta_intervals[:-1]) / 2

    for theta_interval_index in range(num_of_theta_intervals - 1):
        t_bdf = bdf.copy()
        t_bdf = t_bdf[np.isfinite(t_bdf[chosen_column])]
        t_bdf = t_bdf[t_bdf.signal_counts >= 15]
        t_bdf = t_bdf[t_bdf[target_angle] <= theta_intervals[theta_interval_index + 1]]
        t_bdf = t_bdf[t_bdf[target_angle] >= theta_intervals[theta_interval_index]]
        # t_bdf = t_bdf[t_bdf[chosen_column] <= 65]
        # t_bdf = t_bdf[t_bdf[chosen_column] >= 25]

        pol_param, cov = curve_fit(calibration_func, t_bdf[chosen_column], t_bdf[target_variable], maxfev=500000, p0=p0)
        to_return_unbinned[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][theta_interval_index] = pol_param
        pol_param_errs = np.sqrt(np.diag(cov))
        errs_tmp[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][theta_interval_index] = pol_param_errs
        rs = np.linspace(10, 80, num=50)
        chi2 = np.sum((t_bdf[target_variable] - calibration_func(t_bdf[chosen_column], *pol_param)) ** 2)
        plot_calibration(t_bdf, chosen_column, t_sigms, momentum_min, momentum_max, rs, pol_param, chi2,
                         theta_interval_index, fig, axs, avg_sigmas, avg_t_sigmas, target_variable, target_angle,
                         calibration_func)


def param_fit_calibration(bdf: pd.DataFrame, chosen_column: str, r_sigms: int, t_sigms: int, avg_sigmas, avg_t_sigmas,
                          fit_params, target_variable, target_angle, param_calibration_func):
    t_bdf = bdf.copy()
    t_bdf = t_bdf[np.isfinite(t_bdf[chosen_column])]
    t_bdf = t_bdf[t_bdf.signal_counts >= 5]
    X = (np.array(t_bdf[chosen_column]), np.array(t_bdf[target_angle]))
    fit, errs = curve_fit(param_calibration_func, X, t_bdf[target_variable],
                          p0=(1.219, -0.5588, 0.2946, 864.4, -1922, 1055, -2535, 6572, -3751))
    errs = np.sqrt(np.diag(errs))
    for param in range(3):
        fit_params[r_sigms - avg_sigmas[0]][t_sigms - avg_t_sigmas[0]][param] = fit[param * 3: param * 3 + 3]
    chi2 = np.sum((t_bdf[target_variable] - param_calibration_func(X, *fit)) ** 2)


