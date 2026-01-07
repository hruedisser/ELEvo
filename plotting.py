import numpy as np
import matplotlib.pyplot as plt

def visualize_elevo(
        earth_r, 
        earth_lon,
        earth_lat, 
        plot_date, 
        cme_times1, 
        cme_rs1, 
        cme_lons1, 
        cme_lats1, 
        cme_as1, 
        cme_bs1, 
        cme_cs1, 
        name = "elevo_example", 
        output_dir = None,
        output_format = "pdf",
        figsize=(14,10),
        earth_color = "#75CC41",
        sun_color = '#F9F200',
        backcolor = '#052E37',
        symsize_planet = 110,
        fsize = 16,
        colors = ["dodgerblue", "gold", "firebrick"],
        sun_rot = 26.24, # days
        v = 450, # km/s    
        r0 = 695510, # km
        legend = False,
        ):

    cme_indices = np.where(
        cme_times1 == plot_date
    )

    fig = plt.figure(1, figsize=figsize)

    ax = fig.add_subplot(projection="polar")

    # Plot Earth
    ax.scatter(
        earth_lon, 
        earth_r * np.cos(earth_lat),
        s=symsize_planet,
        color=earth_color,
        label="Earth",
        zorder=3
    )

    # Plot Sun
    ax.scatter(
        0, 
        0,
        s=symsize_planet,
        color=sun_color,
        label="Sun",
        alpha = 1,
        edgecolors="black",
        linewidths=0.3,
    )

    # Plot CME trajectories

    t1 = (np.arange(201) - 10) * np.pi / 180 
    cos_t1 = np.cos(t1)
    sin_t1 = np.sin(t1)

    for p in cme_indices[0]:

        lon_rad = np.deg2rad(cme_lons1[p])
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        t = t1 - lon_rad
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        longcirc1 = np.zeros((3, len(t1)))
        rcirc1 = np.zeros((3, len(t1)))

        for i in range(3):

            a = cme_as1[p, i]
            b = cme_bs1[p, i]
            c = cme_cs1[p, i]

            denom = np.sqrt(
                (b * cos_t1) ** 2 + (a * sin_t1) ** 2
            )
            radius = (a * b) / denom

            xc = c * cos_lon + radius * sin_t
            yc = c * sin_lon + radius * cos_t

            longcirc1[i] = np.arctan2(yc, xc)
            rcirc1[i] = np.sqrt(xc ** 2 + yc ** 2)

        alpha_val = 1 - abs(cme_lats1[p] / 100)
        ax.plot(
            longcirc1[0],
            rcirc1[0],
            color=colors[0],
            alpha=alpha_val,
            linewidth=2,
        )
        ax.fill_between(
            longcirc1[2],
            rcirc1[2],
            rcirc1[1],
            color=colors[0],
            alpha=0.05,
        )

    # set axes and grid
    ax.set_theta_zero_location("E")
    plt.thetagrids(range(0,360,45),(u'0\u00b0',u'45\u00b0',u'90\u00b0',u'135\u00b0',u'+/- 180\u00b0',u'- 135\u00b0',u'- 90\u00b0',u'- 45\u00b0'), ha='center', fmt='%d',fontsize=fsize-1,color=backcolor, alpha=0.9,zorder=4)

    plt.rgrids((0.1,0.3,0.5,0.7,1.0),('0.10','0.3','0.5','0.7','1.0 AU'),angle=125, fontsize=fsize-2,alpha=0.5, color=backcolor)

    ax.set_ylim(0, 1.2)


    # Plot Parker Spiral

    theta=np.arange(0,np.deg2rad(180),0.01)


    v = v / 1.496e8 # AU/s

    r0 = r0 / 1.496e8 # AU

    omega  = 2 * np.pi / (sun_rot * 60 * 60 * 24) # rad/s
    r = v/omega * theta + r0 * 7

    if legend:
        ax.legend(loc='upper right', fontsize=fsize)

    fig.tight_layout()

    if output_dir is not None:
        plot_path = output_dir / f"{name}.{output_format}"
        fig.savefig(plot_path, dpi=300)
        return
    else:
        print("No output directory specified, not saving the figure to file.")
        fig.show()
        return