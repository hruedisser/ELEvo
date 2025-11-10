import numpy as np
import datetime


def donki_kinematics(
        i, 
        donki_cat, 
        position_df, 
        gamma = None,
        gamma_min = 0.2,
        gamma_max = 2,
        v_sw = None, 
        v_sw_init = 450,
        v_sw_delta = 100,
        speed_delta = 115,
        res_in_minutes=30,
        kindays=15,
        n_ensemble=100000,
        f = 0.7,
        distance_0 = 21.5 # in solar radii
        ):
    
    kinminutes = int(kindays * 24 * 60 / res_in_minutes)  # number of time steps for kinematics

    distance_0 = distance_0 * 695510  # in km, 1 R_Sun = 695510 km

    print(f"The resolution is set to {res_in_minutes} minutes.")
    

    event = donki_cat[i]

    t_0 = event.launch_time

    lon = event.longitude
    lat = event.latitude
    half_width = event.half_width
    v_0 = event.initial_speed

    earth_lon = position_df['lon'].loc[position_df['timestamp'] == t_0].values # we can assume that the Earth position does not change significantly during the CME propagation time
    earth_r = position_df['r'].loc[position_df['timestamp'] == t_0].values # in AU

    # check if the Earth position is available
    if len(earth_lon) == 0 or len(earth_r) == 0:
        # choose the closest available Earth position
        closest_index = position_df['timestamp'].sub(t_0).abs().idxmin()
        earth_lon = position_df['lon'].iloc[closest_index]
        earth_r = position_df['r'].iloc[closest_index]
        
        print(f"Warning: Earth position at launch time {t_0} not found, using closest available position: lon={earth_lon}, r={earth_r} at timestamp {position_df['timestamp'].iloc[closest_index]}")
    else:
        print(f"Earth position at launch time {t_0}: lon={earth_lon}, r={earth_r}")


    half_width_rad = np.deg2rad(half_width)  # convert half width to radians
    lon_rad = np.deg2rad(lon)
    earth_lon_rad = np.deg2rad(earth_lon)

    if np.abs(lon_rad) + np.abs(earth_lon_rad) > np.pi and np.sign(lon_rad) != np.sign(earth_lon_rad):
        abs_delta_lon = np.abs(lon_rad - (earth_lon_rad + 2 * np.pi *np.sign(earth_lon_rad)))
    else:
        abs_delta_lon = np.abs(lon_rad - earth_lon_rad)


    cme_r = np.zeros([kinminutes, 3]) # radial distance for each time step, mean, lower and upper bounds
    cme_v = np.zeros([kinminutes, 3]) # velocity for each time step, mean, lower and upper bounds

    cme_hit = 1 if abs_delta_lon < half_width_rad else 0 # check if the CME hits the Earth

    if cme_hit == 1:
        print(f'Processing CME {i+1} - hits Earth')
    else:
        print(f'Processing CME {i+1} - does not hit Earth')
    
    # Initialize the ensemble parameters

    if gamma is None:
        gamma_sigma = gamma_max - gamma_min
        gamma = np.abs(np.random.normal(loc=0, scale=gamma_sigma/2, size=n_ensemble)) + gamma_min # make sure gamma is between gamma_min and gamma_max with a half-normal distribution

    if v_sw is None:
        v_sw = np.random.normal(loc=v_sw_init, scale=v_sw_delta/2, size=n_ensemble) # solar wind speed, normally distributed around v_sw_init with a standard deviation of v_sw_delta

    v = np.random.normal(loc=v_0, scale=speed_delta/2, size=n_ensemble) # CME speed, normally distributed around v_0 with a standard deviation of speed_delta

    # rescale parameters

    gamma = gamma * 1e-7

    # compute kinematics

    dv = v - v_sw # difference between CME speed and solar wind speed for each ensemble member

    bg_sgn = np.where(v > v_sw, 1, -1) # sign of the background speed, 1 if CME is faster than solar wind, -1 otherwise

    dt = np.arange(kinminutes) * res_in_minutes * 60  # in seconds
    dt = np.vstack([dt] * n_ensemble)
    dt = np.transpose(dt)

    cme_r_ensemble = (bg_sgn / gamma) * np.log1p(bg_sgn * gamma * dv * dt) + v_sw * dt + distance_0 # (timestamps, ensemblemembers)
    cme_v_ensemble = dv / (1 + bg_sgn * gamma * dv * dt) + v_sw # (timestamps, ensemblemembers)

    cme_r_mean = np.mean(cme_r_ensemble, axis=1) # (timestamps,) 
    cme_r_std = np.std(cme_r_ensemble, axis=1) # (timestamps,)
    cme_v_mean = np.mean(cme_v_ensemble, axis=1) # (timestamps,)
    cme_v_std = np.std(cme_v_ensemble, axis=1) # (timestamps,)

    cme_r[:, 0] = cme_r_mean / 1.496e8  # convert km to AU
    cme_r[:, 1] = (cme_r_mean - 2 * cme_r_std) / 1.496e8  # convert km to AU
    cme_r[:, 2] = (cme_r_mean + 2 * cme_r_std) / 1.496e8  # convert km to AU

    cme_v[:, 0] = cme_v_mean
    cme_v[:, 1] = cme_v_mean - 2 * cme_v_std
    cme_v[:, 2] = cme_v_mean + 2 * cme_v_std

    # Ellipse parameters

    # angle of the ellipse in radians
    theta = np.arctan(
        f**2 * np.tan(half_width_rad)
    ) # scalar

    # r = b/omega, where b is the semi-minor axis of the ellipse
    omega = np.sqrt(
        (f**2 - 1) * np.cos(theta)**2 + 1
    ) # scalar
    
    # notice that cme_r is R(t) in the paper, which is the radial distance of the apex at time t, while r is thedistance of a point on the ellipse from the center of the ellipse
    cme_b = cme_r * omega * np.sin(half_width_rad) / (np.cos(half_width_rad-theta) + omega * np.sin(half_width_rad)) # (timestamps, 3)

    # a = b/f, where a is the semi-major axis of the ellipse and b is the semi-minor axis of the ellipse
    cme_a = cme_b / f # (timestamps, 3)

    # c = R(t) - b, where c is the distance from the Sun to the center of the ellipse at time t
    cme_c = cme_r - cme_b # (timestamps, 3)

    # create a list of timestamps
    timestamps = [
        t_0 + datetime.timedelta(minutes=i * res_in_minutes)
        for i in range(kinminutes)
    ]


    if cme_hit == 1:
        sin_delta_squared = (
            np.sin(abs_delta_lon)**2
        )
        cos_delta_squared = (
            np.cos(abs_delta_lon)**2
        )

        root = np.sqrt(
            (cme_b**2 - cme_c**2) * f**2 * sin_delta_squared + cme_b**2 * cos_delta_squared
        )

        distance_earth = (cme_c * np.cos(abs_delta_lon) + root) / (f**2 * sin_delta_squared + cos_delta_squared)  # front crossing distance


        # find the indices where the distance to Earth is closest to the radial distance of the Earth
        index_window_begin = np.argmin(
            np.abs(distance_earth[:,2] - earth_r)
        )
        index_window_middle = np.argmin(
            np.abs(distance_earth[:,0] - earth_r)
        )
        index_window_end = np.argmin(
            np.abs(distance_earth[:,1] - earth_r)
        )

        window_begin_time = timestamps[index_window_begin]
        window_middle_time = timestamps[index_window_middle]
        window_end_time = timestamps[index_window_end]

        window_begin_speed = cme_v[index_window_begin, 2]
        window_middle_speed = cme_v[index_window_middle, 0]
        window_end_speed = cme_v[index_window_end, 1]
    
    else:
        window_begin_time = None
        window_middle_time = None
        window_end_time = None

        window_begin_speed = None
        window_middle_speed = None
        window_end_speed = None

    # change cme_lon to be the right dimension
    cme_lon = np.full(kinminutes, lon)  # create an array of the same length as timestamps filled with the CME longitude
    cme_lat = np.full(kinminutes, lat)  # create an array of the same length as timestamps filled with the CME latitude

    return {
        "launch_time": t_0,
        "event_id": event.event_id,
        "cme_hit": cme_hit,
        "cme_r": cme_r,
        "cme_v": cme_v,
        "cme_lon": cme_lon,
        "cme_lat": cme_lat,
        "cme_v0": v_0,
        "cme_half_width": half_width,
        "window_begin_time": window_begin_time,
        "window_middle_time": window_middle_time,
        "window_end_time": window_end_time,
        "window_begin_speed": window_begin_speed,
        "window_middle_speed": window_middle_speed,
        "window_end_speed": window_end_speed,
        "cme_a": cme_a,
        "cme_b": cme_b,
        "cme_c": cme_c,
        "cme_time": timestamps,
    }


