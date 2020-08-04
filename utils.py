import matplotlib.pyplot as plt2
from torch import sqrt,norm,sum,cat,sigmoid,exp,tensor
from torch.nn.functional import softplus
from scipy.special import expit

def get_contours(Z, x, y, n_contour=10):
    levels = np.linspace(0, np.max(Z), n_contour)
    cont = plt2.contourf(x, y, Z, levels=np.linspace(0, np.max(Z), n_contour + 1), nchunk=10)
    plt2.close()
    contours = [cs.get_paths() for cs in cont.collections]
    levels = [np.repeat(levels[i], len(contours[i])) for i in range(n_contour)]
    contours = [item for sublist in contours for item in sublist]
    contours = [c.vertices for c in contours]
    contours_x = [list(c[:, 0]) for c in contours]
    contours_y = [list(c[:, 1]) for c in contours]
    return contours_x, contours_y, levels


def get_pitch_control_proportion(Z):
    pc = np.sum(Z) / np.prod(Z.shape) * 100
    return '%s' % float('%.3g' % pc) + '%'


def get_pitch_control(home_pos,
                      away_pos,
                      home_v,
                      away_v,
                      ball_pos,
                      target_position,
                      x_grid,
                      y_grid,
                      xT_values,
                      transition,
                      value):
    reaction_time = 0.7
    max_player_speed = 5.
    average_ball_speed = 15.
    sigma = np.pi / np.sqrt(3.) / 0.45

    ball_travel_time = norm(target_position - ball_pos, dim=2) / average_ball_speed
    ts = 0.5 * (ti + 1) * 10 + ball_travel_time[:, :, None]

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    r_reaction_home = home_pos + reaction_time * home_v
    r_reaction_away = away_pos + reaction_time * away_v

    tti_home = reaction_time + sqrt(
        sum((target_position - r_reaction_home) ** 2.,
            dim=3)) / max_player_speed

    tti_away = reaction_time + sqrt(
        sum((target_position - r_reaction_away) ** 2.,
            dim=3)) / max_player_speed

    tti = cat([tti_home, tti_away])

    tmp1 = sigma * (ts - tti[:, :, :, None])
    tmp2 = (ball_travel_time[None, :, :, None] - tti[:, :, :, None]) * sigma

    lamb = 4.3 * sigmoid(tmp1)
    Lamb = 4.3 / sigma * (softplus(tmp1) - softplus(tmp2))

    S = exp(-sum(Lamb, dim=0))
    h = sum(lamb[:11], dim=0)
    integrand = S * h
    out = 5 * sum(integrand * wi, dim=2)
    out = out.numpy()

    if transition == 'Yes':
        gaussian_x = np.exp(-0.5 * (x_grid - np.array(ball_pos[0][0])) ** 2 / (23.9 / 105 * 100) ** 2)
        gaussian_y = np.exp(-0.5 * (y_grid - np.array(ball_pos[0][1])) ** 2 / (23.92 / 68 * 100) ** 2)
        gaussian = np.outer(gaussian_x, gaussian_y)
        transition_prob = gaussian * out
        out = out * transition_prob
    if value == 'Yes':
        out = out * xT_values

    return out


def get_pitch_control_fernandez(home_pos,
                                away_pos,
                                home_v,
                                away_v,
                                ball_pos,
                                target_position,
                                x_grid,
                                y_grid,
                                xT_values,
                                transition,
                                value):
    s_home = np.linalg.norm(home_v, axis=1, keepdims=True)
    s_away = np.linalg.norm(away_v, axis=1, keepdims=True)
    theta_home = np.arccos(home_v[:, 0, None] / s_home)
    theta_away = np.arccos(away_v[:, 0, None] / s_away)
    mu_home = home_pos + 0.5 * home_v
    mu_away = away_pos + 0.5 * away_v
    Srat_home = np.minimum((s_home / 13.0) ** 2, 1.)
    Srat_away = np.minimum((s_away / 13.0) ** 2, 1.)
    Ri_home = np.minimum(4 + np.linalg.norm(ball_pos - home_pos, axis=1, keepdims=True) ** 3 / 972, 10.)
    Ri_away = np.minimum(4 + np.linalg.norm(ball_pos - away_pos, axis=1, keepdims=True) ** 3 / 972, 10.)
    RSinv_home = np.empty((11, 2, 2))
    RSinv_away = np.empty((11, 2, 2))
    S1_home = 2 / ((1 + Srat_home) * Ri_home)
    S2_home = 2 / ((1 - Srat_home) * Ri_home)
    S1_away = 2 / ((1 + Srat_away) * Ri_away)
    S2_away = 2 / ((1 - Srat_away) * Ri_away)
    RSinv_home[:, 0, 0] = (S1_home * np.cos(theta_home))[:, 0]
    RSinv_home[:, 1, 0] = (S1_home * np.sin(theta_home))[:, 0]
    RSinv_home[:, 0, 1] = - (S2_home * np.sin(theta_home))[:, 0]
    RSinv_home[:, 1, 1] = (S2_home * np.cos(theta_home))[:, 0]
    RSinv_away[:, 0, 0] = (S1_away * np.cos(theta_away))[:, 0]
    RSinv_away[:, 1, 0] = (S1_away * np.sin(theta_away))[:, 0]
    RSinv_away[:, 0, 1] = - (S2_away * np.sin(theta_away))[:, 0]
    RSinv_away[:, 1, 1] = (S2_away * np.cos(theta_away))[:, 0]
    denominators_h = np.exp(-0.5 * np.sum((np.matmul(home_pos[:, None, :] - mu_home[:, None, :], RSinv_home)) ** 2, -1))
    denominators_a = np.exp(-0.5 * np.sum((np.matmul(away_pos[:, None, :] - mu_away[:, None, :], RSinv_away)) ** 2, -1))
    x_min_mu_h = mu_home[:, None, None, :] - target_position[None, :, :, :]
    x_min_mu_a = mu_away[:, None, None, :] - target_position[None, :, :, :]
    mm_h = np.matmul(RSinv_home[:, None, None, :], x_min_mu_h[:, :, :, :, None])[..., 0]
    mm_a = np.matmul(RSinv_away[:, None, None, :], x_min_mu_a[:, :, :, :, None])[..., 0]
    infl_h = np.exp(-0.5 * np.sum(mm_h ** 2, -1)) / denominators_h[..., None]
    infl_a = np.exp(-0.5 * np.sum(mm_a ** 2, -1)) / denominators_a[..., None]

    out = expit(np.sum(infl_h, 0) - np.sum(infl_a, 0))

    if transition == 'Yes':
        gaussian_x = np.exp(-0.5 * (x_grid - ball_pos[0][0]) ** 2 / (23.9 / 105 * 100) ** 2)
        gaussian_y = np.exp(-0.5 * (y_grid - ball_pos[0][1]) ** 2 / (23.92 / 68 * 100) ** 2)
        gaussian = np.outer(gaussian_x, gaussian_y)
        transition_prob = gaussian * out
        transition_prob /= np.sum(4 * transition_prob)
        out = out * transition_prob
    if value == 'Yes':
        out = out * xT_values

    return out


def get_pitch_control_fernandez_modified(home_pos,
                                         away_pos,
                                         home_v,
                                         away_v,
                                         ball_pos,
                                         target_position,
                                         x_grid,
                                         y_grid,
                                         xT_values,
                                         transition,
                                         value):
    s_home = np.linalg.norm(home_v, axis=1, keepdims=True)
    s_away = np.linalg.norm(away_v, axis=1, keepdims=True)
    theta_home = np.arccos(home_v[:, 0, None] / s_home)
    theta_away = np.arccos(away_v[:, 0, None] / s_away)
    mu_home = home_pos + 0.5 * home_v
    mu_away = away_pos + 0.5 * away_v
    Srat_home = np.minimum((s_home / 13.0) ** 2, 1.)
    Srat_away = np.minimum((s_away / 13.0) ** 2, 1.)
    Ri_home = np.minimum(4 + np.linalg.norm(ball_pos - home_pos, axis=1, keepdims=True) ** 3 / 972, 10.)
    Ri_away = np.minimum(4 + np.linalg.norm(ball_pos - away_pos, axis=1, keepdims=True) ** 3 / 972, 10.)
    RSinv_home = np.empty((11, 2, 2))
    RSinv_away = np.empty((11, 2, 2))
    S1_home = 2 / ((1 + Srat_home) * Ri_home)
    S2_home = 2 / ((1 - Srat_home) * Ri_home)
    S1_away = 2 / ((1 + Srat_away) * Ri_away)
    S2_away = 2 / ((1 - Srat_away) * Ri_away)
    RSinv_home[:, 0, 0] = (S1_home * np.cos(theta_home))[:, 0]
    RSinv_home[:, 1, 0] = (S1_home * np.sin(theta_home))[:, 0]
    RSinv_home[:, 0, 1] = - (S2_home * np.sin(theta_home))[:, 0]
    RSinv_home[:, 1, 1] = (S2_home * np.cos(theta_home))[:, 0]
    RSinv_away[:, 0, 0] = (S1_away * np.cos(theta_away))[:, 0]
    RSinv_away[:, 1, 0] = (S1_away * np.sin(theta_away))[:, 0]
    RSinv_away[:, 0, 1] = - (S2_away * np.sin(theta_away))[:, 0]
    RSinv_away[:, 1, 1] = (S2_away * np.cos(theta_away))[:, 0]
    denominators_h = np.exp(-0.5 * np.sum((np.matmul(home_pos[:, None, :] - mu_home[:, None, :], RSinv_home)) ** 2, -1))
    denominators_a = np.exp(-0.5 * np.sum((np.matmul(away_pos[:, None, :] - mu_away[:, None, :], RSinv_away)) ** 2, -1))
    x_min_mu_h = mu_home[:, None, None, :] - target_position[None, :, :, :]
    x_min_mu_a = mu_away[:, None, None, :] - target_position[None, :, :, :]
    mm_h = np.matmul(RSinv_home[:, None, None, :], x_min_mu_h[:, :, :, :, None])[..., 0]
    mm_a = np.matmul(RSinv_away[:, None, None, :], x_min_mu_a[:, :, :, :, None])[..., 0]
    infl_h = np.exp(-0.5 * np.sum(mm_h ** 2, -1)) / denominators_h[..., None]
    infl_a = np.exp(-0.5 * np.sum(mm_a ** 2, -1)) / denominators_a[..., None]
    infl_all = np.concatenate([infl_h, infl_a])
    infl_all /= np.sum(infl_all, 0)

    out = np.sum(infl_all[:11], 0)

    if transition == 'Yes':
        gaussian_x = np.exp(-0.5 * (x_grid - ball_pos[0][0]) ** 2 / (23.9 / 105 * 100) ** 2)
        gaussian_y = np.exp(-0.5 * (y_grid - ball_pos[0][1]) ** 2 / (23.92 / 68 * 100) ** 2)
        gaussian = np.outer(gaussian_x, gaussian_y)
        transition_prob = gaussian * out
        transition_prob /= np.sum(4 * transition_prob)
        out = out * transition_prob
    if value == 'Yes':
        out = out * xT_values

    return out