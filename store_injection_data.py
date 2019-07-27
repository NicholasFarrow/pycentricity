#!/home/isobel.romero-shaw/anaconda3/bin/python3.6
import bilby as bb
import python_code.waveform as wf
import python_code.overlap as ovlp
import python_code.reweight as rwt
import numpy as np
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# Set up the argument parser
parser = argparse.ArgumentParser("")
parser.add_argument("-e", "--eccentricity", help="The eccentricity of the injected signal")
args = parser.parse_args()

np.random.seed(54321)
eccentricity = float(args.eccentricity)

# injection parameters
injection_parameters = dict(
    mass_1=35.0,
    mass_2=30.0,
    eccentricity=eccentricity,
    luminosity_distance=440.0,
    theta_jn=0.4,
    psi=0.1,
    phase=1.2,
    geocent_time=0.0,
    ra=45,
    dec=5.73,
    chi_1=0.0,
    chi_2=0.0,
)

# Frequency settings
minimum_frequency = 20
maximum_frequency = 2046
sampling_frequency = 4096
duration = 8
deltaT = 0.2

# Comparison waveform
comparison_waveform_generator = wf.get_IMRPhenomD_comparison_waveform_generator(
    minimum_frequency, sampling_frequency, duration
)
comparison_waveform_frequency_domain = comparison_waveform_generator.frequency_domain_strain(
    injection_parameters
)
# Interferometers
interferometers = bb.gw.detector.InterferometerList(["H1", "L1"])
interferometers.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - duration + 2,
)
for ifo in interferometers:
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency

# SEOBNRe waveform to inject
t, seobnre_waveform_time_domain = wf.seobnre_bbh_with_spin_and_eccentricity(
    parameters=injection_parameters,
    sampling_frequency=sampling_frequency,
    minimum_frequency=minimum_frequency - 10,
    maximum_frequency=maximum_frequency + 1000,
)
seobnre_wf_td, seobnre_wf_fd, max_overlap, index_shift, phase_shift = ovlp.maximise_overlap(
    seobnre_waveform_time_domain,
    comparison_waveform_frequency_domain,
    sampling_frequency,
    interferometers[0].frequency_array,
    interferometers[0].power_spectral_density,
)
print('maximum overlap: ' + str(max_overlap))

seobnre_wf_fd = ovlp.zero_pad_frequency_domain_signal(seobnre_wf_fd, interferometers)
# Inject the signal
interferometers.inject_signal(
    parameters=injection_parameters, injection_polarizations=seobnre_wf_fd
)

log_likelihood_circular = rwt.log_likelihood_ratio(comparison_waveform_frequency_domain, interferometers, injection_parameters, duration)
log_likelihood_eccentric = rwt.log_likelihood_ratio(seobnre_wf_fd, interferometers, injection_parameters, duration)

print('circular log likelihood: ' + str(log_likelihood_circular))
print('eccentric log likelihood: ' + str(log_likelihood_eccentric))
print('ratio: ' + str(log_likelihood_eccentric - log_likelihood_circular))

# Test
fig = plt.figure()
bin_number = 20
max_ecc = 0.2
for minimum_log_eccentricity in [-4, -3, -2, -1]:
    e, average_log_likelihood, log_weight, log_likelihood_grid = rwt.new_weight(
    log_likelihood_circular,
    injection_parameters,
    comparison_waveform_frequency_domain,
    interferometers,
    duration,
    sampling_frequency,
    maximum_frequency,
    str(minimum_log_eccentricity),
    minimum_log_eccentricity=minimum_log_eccentricity,
    number_of_eccentricity_bins=bin_number
)

    print('eccentricity-maginalised log likelihood: ' + str(average_log_likelihood))
    print('eccentricity: ' + str(e))
    print('log weight: ' + str(log_weight))

    plt.semilogx(np.logspace(minimum_log_eccentricity, max_ecc, bin_number), log_likelihood_grid,
                 label='minimum_log_ecc='+str(minimum_log_eccentricity))

plt.xlabel('eccentricity')
plt.ylabel('log likelihood')
plt.legend()
plt.savefig('likelihood_granularity_problem.png', bbox_inches='tight')
plt.show()
'''
# Save the interferometer data
label = 'circular'

fig = plt.figure()
if eccentricity > 0.0:
    label = 'eccentric'
for ifo in interferometers:
    with open(label + '_' + ifo.name + '_frequency_domain_strain_data.csv', 'w') as data_file:
        for i, f in enumerate(ifo.frequency_array):
            data_file.write(str(f) + ',' + str(ifo.frequency_domain_strain[i]) + '\n')
        plt.loglog(ifo.frequency_array, abs(ifo.frequency_domain_strain), label=ifo.name)
plt.legend()
plt.xlabel('frequency')
plt.ylabel('strain')
plt.show()
'''