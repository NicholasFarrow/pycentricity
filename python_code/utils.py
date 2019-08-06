"""

A file containing utility functions and related objects for the pySEOBNRe package.

"""
import numpy as np
import matplotlib.pyplot as plt
import corner


def get_data(data_list):
    return np.transpose(np.vstack(data_list))


def plot_reweighted_posteriors(posterior, weights, label):
    parameters = [posterior[parameter] for parameter in parameter_keys.keys()]
    labels = [parameter_keys[parameter] for parameter in parameter_keys.keys()]
    # First compare unweighted to weighted
    figure = corner.corner(get_data(parameters), labels=labels, bins=20, smooth=0.9, label_kwargs=dict(fontsize=12),
                           titles=True,
                           title_kwargs=dict(fontsize=16), color='#0072C1', quantiles=[0.16, 0.84],
                           levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                           hist_kwargs=dict(density=True),
                           plot_density=False, plot_datapoints=True, fill_contours=True, label='unweighted',
                           max_n_ticks=3)
    corner.corner(get_data(parameters), weights=weights, bins=20, labels=labels, fig=figure, smooth=0.9,
                  label_kwargs=dict(fontsize=12), titles=True,
                  title_kwargs=dict(fontsize=16), color='#FF8C00', quantiles=[0.16, 0.84],
                  hist_kwargs=dict(density=True),
                  levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                  plot_density=False, plot_datapoints=True, fill_contours=True, label='weighted',
                  max_n_ticks=3)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=10)
    plt.legend()
    plt.savefig(label + "_corner.png")
    plt.clf()


psd_files_base_path = "/home/isobel.romero-shaw/lvc_pe_sample_release/"

parameter_keys = dict(
    chirp_mass='$\mathcal{M}$',
    mass_ratio='$q$',
    luminosity_distance='$d_L$',
    ra='RA',
    dec='DEC',
    chi_1='$\chi_1$',
    chi_2='$\chi_2$',
    theta_jn='$\\theta_{jn}$',
    phase='$\phi$',
    psi='$\psi$',
    geocent_time='$t_{geo}$'
)

trigger_time = dict(
    GW150914=1126259462.391,
    GW151012=1128678900.4,
    GW151226=1135136350.6,
    GW170104=1167559936.6,
    GW170608=1180922494.5,
    GW170729=1185389807.3,
    GW170809=1186302519.8,
    GW170814=1186741861.5,
    GW170818=1187058327.1,
    GW170823=1187529256.5,
)

event_channels = dict(
    GW150914=dict(H1="DCS-CALIB_STRAIN_C02", L1="DCS-CALIB_STRAIN_C02"),
    GW151012=dict(H1="DCS-CALIB_STRAIN_C02", L1="DCS-CALIB_STRAIN_C02"),
    GW151226=dict(H1="DCS-CALIB_STRAIN_C02", L1="DCS-CALIB_STRAIN_C02"),
    GW170104=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170608=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170729=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170809=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
    GW170814=dict(
        H1="DCH-CLEAN_STRAIN_C02",
        L1="DCH-CLEAN_STRAIN_C02",
        V1="Hrec_hoft_V1O2Repro2A_16384Hz",
    ),
    GW170818=dict(
        H1="DCH-CLEAN_STRAIN_C02",
        L1="DCH-CLEAN_STRAIN_C02",
        V1="Hrec_hoft_V1O2Repro2A_16384Hz",
    ),
    GW170823=dict(H1="DCH-CLEAN_STRAIN_C02", L1="DCH-CLEAN_STRAIN_C02"),
)

event_psd_file_path = dict(
    GW150914=dict(
        H1=psd_files_base_path + "O1/PE/GW150914/rerun_O2_catalog/h1_psd.dat",
        L1=psd_files_base_path + "O1/PE/GW150914/rerun_O2_catalog/l1_psd.dat",
    ),
    GW151012=dict(
        H1=psd_files_base_path
        + "O1/PE/LVT151012/rerun_O2_catalog/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path
        + "O1/PE/LVT151012/rerun_O2_catalog/BayesWave_median_PSD_L1.dat",
    ),
    GW151226=dict(
        H1=psd_files_base_path
        + "O1/PE/GW151226/rerun_O2_catalog/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path
        + "O1/PE/GW151226/rerun_O2_catalog/BayesWave_median_PSD_L1.dat",
    ),
    GW170104=dict(
        H1=psd_files_base_path
        + "O2/PE/GW170104/rerun_O2_catalog/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path
        + "O2/PE/GW170104/rerun_O2_catalog/BayesWave_median_PSD_L1.dat",
    ),
    GW170608=dict(
        H1=psd_files_base_path + "/O2/PE/GW170608/GW170608_C02_reruns/h1_psd_C02.dat",
        L1=psd_files_base_path + "O2/PE/GW170608/GW170608_C02_reruns/l1_psd_C02.dat",
    ),
    GW170729=dict(
        H1=psd_files_base_path + "O2/PE/GW170729/Median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170729/Median_PSD_L1.dat",
    ),
    GW170809=dict(
        H1="/home/isobel.romero-shaw/public_html/known_events/GW170809/dynesty_isobel_defaults/GW170809_LIGO_Hanford_PSD1Hz_psd.txt",
        L1="/home/isobel.romero-shaw/public_html/known_events/GW170809/dynesty_isobel_defaults/GW170809_LIGO_Livingston_PSD1Hz_psd.txt",
    ),
    GW170814=dict(
        H1=psd_files_base_path + "O2/PE/GW170814/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170814/BayesWave_median_PSD_L1.dat",
        V1=psd_files_base_path + "O2/PE/GW170814/BayesWave_median_PSD_V1.dat",
    ),
    GW170818=dict(
        H1=psd_files_base_path + "O2/PE/GW170818/psd/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170818/psd/BayesWave_median_PSD_L1.dat",
        V1=psd_files_base_path + "O2/PE/GW170818/psd/BayesWave_median_PSD_V1.dat",
    ),
    GW170823=dict(
        H1=psd_files_base_path + "O2/PE/GW170823/BayesWave_median_PSD_H1.dat",
        L1=psd_files_base_path + "O2/PE/GW170823/BayesWave_median_PSD_L1.dat",
    ),
)

event_detectors = {
    key: list(event_channels[key].keys()) for key in event_channels.keys()
}

event_duration = dict(
    GW150914=8,
    GW151012=8,
    GW151226=8,
    GW170104=4,
    GW170608=16,
    GW170729=4,
    GW170809=4,
    GW170814=4,
    GW170818=4,
    GW170823=4,
)

minimum_frequency = dict(
    GW150914=20,
    GW151012=20,
    GW151226=20,
    GW170104=20,
    GW170608=30,
    GW170729=20,
    GW170809=20,
    GW170814=20,
    GW170818=20,
    GW170823=20,
)

sampling_frequency = dict(
    GW150914=4096,
    GW151012=4096,
    GW151226=4096,
    GW170104=4096,
    GW170608=4096,
    GW170729=4096,
    GW170809=4096,
    GW170814=4096,
    GW170818=4096,
    GW170823=4096,
)

search_keys = [
    "mass_ratio",
    "chirp_mass",
    "chi_1",
    "chi_2",
    "dec",
    "ra",
    "theta_jn",
    "psi",
    "luminosity_distance",
    "geocent_time",
    "phase",
]
