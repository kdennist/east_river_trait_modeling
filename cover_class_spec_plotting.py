
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({'font.size': 20})
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.labelpad'] = 6


def main():
    #cover_extraction_20201021.csv
    file_path = 'metals_v12/data/extraction_chem.csv'
    metal_hist(file_path)
    # wavelengths_file = 'data/neon_wavelengths.txt'
    # spectra, cover_types, wv, genus_types = spec_cleaning(file_path, wavelengths_file, 'refl_B_')
    # plot_avg_spectra(spectra, cover_types, genus_types, wv)

def metal_hist(file_path):
    extract = pd.read_csv(file_path)
    longdf = extract.melt(id_vars=['SampleSiteID', 'SampleSiteCode', 'Site_Veg', 'Genus', 'Needles', 'LMA_gm2', 'LWA_gm2', 'LWC_%', 'd15N', 'd13C', 'N_weight_percent', 'C_weight_percent', 'CN'])

    longdf.rename(columns={'variable': 'Element'}, inplace=True)
    longdf.rename(columns={'value': "Concentration"}, inplace=True)
    longdf.drop_duplicates(inplace=True)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(longdf.head())
    eles = longdf['Element'].unique()
    type = longdf['Genus'].unique()

    for e in range(len(eles)):
        print("Plotting:" + eles[e])
        temp_df = longdf[longdf['Element'] == eles[e]]
        dataset = []
        labels = []
        colorindex = []
        fig, ax = plt.subplots(figsize=(12, 7))

        for t in range(len(type)):
            print(type[t])
            temp_type_df = temp_df[temp_df['Genus'] == type[t]]
            veg = temp_type_df['Site_Veg'].unique()

            for v in range(len(veg)):
                print(veg[v])
                species = temp_type_df[temp_type_df['Site_Veg'] == veg[v]]['Concentration']
                dataset.append(species)
                labels.append(veg[v])
                colorindex.append(type[t])

        # Add Title
        ax.set_title('Distribution of ' + eles[e] + ' as a function of species or class')

        # Add axis labels
        ax.set_xlabel('Species')
        ax.set_ylabel('Concentration (ppm)')

        # Add major gridlines in the y-axis
        ax.grid(color='grey', axis='y', linestyle='-', linewidth=0.25, alpha=0.5)
        # ax.set_xticklabels(labels="Distribution", rotation=60, fontsize=8)
        kwargs = dict(histtype='stepfilled', alpha=0.3, bins=32)

        # plot species in respective colors
        colors = ['lightslategrey', 'bisque', 'orangered', 'forestgreen', 'dodgerblue', 'gold', 'darkcyan', 'firebrick']
        plt.hist(dataset, **kwargs, color=colors, label=labels)
        plt.xlabel("Concentration")
        plt.ylabel("Frequency")
        plt.legend()

        # colors = ['mediumblue', 'lightgreen', 'orange', 'yellow', 'purple', 'firebrick']

        # for patch, c in zip(bplot['boxes'], colorindex):
        #     patch.set_facecolor(colors[c])
        plt.savefig('../east_river_trait_modeling/metals_v12/figures/quantile_plots/' + eles[e] + '.png')

    # header = list(extract)
    # print(header)
    # elements = ['1C', '8P', '8U', 'Al', 'As', 'Co', 'Cr', 'Cu',
    #             'Mo', 'Na', 'Ni', 'Se', 'Si', 'Sr', 'Zn']
    #
    # cover_types = extract['Site_Veg']
    # genus_types = extract['Genus']
    # element = extract[elements]
    #
    # g_types = genus_types.unique()
    # c_types = cover_types.unique()
    #
    # # set plot parameters for this day's plot
    # fig = plt.figure(figsize=(27, 30))  # , constrained_layout=True)
    # grid = gridspec.GridSpec(2, 4, wspace=.3, hspace=.5)
    #
    # metals = []
    #
    # for c in range(len(c_types)): # d = # dates
    #     print('Plotting Standard Data for ' + c_types[c])
    #     # subset standards down to only those for specified day
    #     temp_smpl = extract.loc[extract['Site_Veg'] == c_types[c]]
    #     metals.append(temp_smpl)
    #
    #
    # plt.hist(temp_smpl['1C'])
    # plt.hist(temp_smpl['As'])
    #
    # plt.show()




def find_nearest(array_like, v):
    index = np.argmin(np.abs(np.array(array_like) - v))
    return index

def spec_cleaning(file_path, wavelengths_file, band_preface):
    extract = pd.read_csv(file_path)
    headerSpec = list(extract)  # Reflectance bands start at 18 (17 in zero base)

    # defining wavelengths
    wv = np.genfromtxt(wavelengths_file)
    bad_bands = []
    good_band_ranges = []

    bad_band_ranges = [[0, 425], [1345, 1410], [1805, 2020], [2470, 2700]]
    for _bbr in range(len(bad_band_ranges)):
        bad_band_ranges[_bbr] = [find_nearest(wv, x) for x in bad_band_ranges[_bbr]]
        if (_bbr > 0):
            good_band_ranges.append([bad_band_ranges[_bbr-1][1], bad_band_ranges[_bbr][0]])

        for n in range(bad_band_ranges[_bbr][0], bad_band_ranges[_bbr][1]):
            bad_bands.append(n)
    bad_bands.append(len(wv)-1)

    good_bands = np.array([x for x in range(0, 426) if x not in bad_bands])

    # first column of reflectance data
    rfdat = list(extract).index(band_preface + '1')

    all_band_indices = (np.array(good_bands)+rfdat).tolist()
    all_band_indices.extend((np.array(bad_bands)+rfdat).tolist())
    all_band_indices = np.sort(all_band_indices)

    spectra = np.array(extract[np.array(headerSpec)[all_band_indices]])
    spectra[:, bad_bands] = np.nan

    cover_types = extract['Site_Veg']
    genus_types = extract['Genus']


    return spectra, cover_types, wv, genus_types


def plot_avg_spectra(spectra, cover_types, genus_types, wv, brightness_normalize=True):
    g_types = genus_types.unique()
    c_type = cover_types.unique()
    c_types = c_type[21:]

    # Quantile calculation
    color_sets = ['navy', 'purple', 'forestgreen', 'royalblue', 'gray', 'darkorange', 'black', 'brown', 'tan']

    # Plot the difference between needles and noneedles in reflectance data
    figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    plt.subplots(2, 1)
    for _c, cover in enumerate(c_types):
        # loop through calc
        c_spectra = spectra[cover_types == cover] # arg for threshold
        print(color_sets[_c])
        if brightness_normalize:
            scale_factor = 1.
        else:
            scale_factor = 100.
        plt.plot(wv, np.nanmean(c_spectra, axis=0) / scale_factor, c=color_sets[_c], linewidth=2)
        plt.fill_between(wv, np.nanmean(c_spectra, axis=0) / scale_factor - np.nanstd(c_spectra, axis=0) / scale_factor,
                         np.nanmean(
                             c_spectra, axis=0) / scale_factor + np.nanstd(c_spectra, axis=0) / scale_factor, alpha=.15,
                         facecolor=color_sets[_c])

    plt.legend(c_types, prop={'size':5})
    plt.ylabel('Reflectance (%)')
    if brightness_normalize:
        plt.ylabel('Brightness Norm. Reflectance')
    else:
        plt.ylabel('Reflectance (%)')
    plt.xlabel('Wavelength (nm)')

    plt.savefig(os.path.join('metals_v13', 'figures', 'quantile_plots','class_spectra.png'), **figure_export_settings)
    del fig


if __name__ == "__main__":
    main()