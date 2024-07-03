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
plane_angles = np.array([1.745319668152660597e-01, 4.072425337478433605e-01, 6.399531006804206612e-01, 8.726636676129979620e-01, 1.105374234545575263e+00, 1.338084801478152563e+00, 1.570795368410729864e+00, 1.803505935343307165e+00, 2.036216502275884466e+00, 2.268927069208461766e+00, 2.501637636141039067e+00, 2.734348203073616368e+00, 2.967058770006193225e+00, 3.199769336938770525e+00, 3.432479903871347826e+00, 3.665190470803925127e+00, 3.897901037736502428e+00, 4.130611604669079284e+00, 4.363322171601657473e+00, 4.596032738534233886e+00, 4.828743305466812075e+00, 5.061453872399388487e+00, 5.294164439331966676e+00, 5.526875006264543089e+00, 5.759585573197120389e+00, 5.992296140129697690e+00, 6.225006707062274991e+00])
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


def plot_event(ev_coords, track_coords, tpc_coords, transposed=False, num=0):
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
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1, color=colors[num]), name=f'Ring {num+1}'))

    x = track_coords[0]
    y = track_coords[1]
    z = track_coords[2]
    if transposed:
        z, x = x, z
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num + 1]), name=f'Drift Chamber {num + 1}', line=dict(dash='dash')))

    # x = np.array(file['events;1']['SiStripG4Hits.postStepPosition.x'].array())[num]
    # y = np.array(file['events;1']['SiStripG4Hits.postStepPosition.y'].array())[num]
    # z = np.array(file['events;1']['SiStripG4Hits.postStepPosition.z'].array())[num]
    # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num]), name=f'Si Strips {num + 1}', line=dict(dash='solid')))

    x = tpc_coords[0]
    y = tpc_coords[1]
    z = tpc_coords[2]
    if transposed:
        z, x = x, z
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, marker=dict(size=1, color=colors[num + 2]), name=f'TPC {num + 1}', line=dict(dash='solid')))


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