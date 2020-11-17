
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
    longdf = metal_hist(file_path)
    wavelengths_file = 'data/neon_wavelengths.txt'
    spectra, cover_types, wv, genus_types, site_codes = spec_cleaning(file_path, wavelengths_file, 'refl_B_')
    # plot_avg_spectra(spectra, cover_types, genus_types, wv)
    plot_avg_quantile(spectra, longdf, cover_types, genus_types, site_codes, wv, brightness_normalize=True)

def metal_hist(file_path):
    extract = pd.read_csv(file_path)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    longdf = extract.melt(id_vars=['SampleSiteID', 'SampleSiteCode', 'Site_Veg', 'Genus', 'Needles', 'LMA_gm2', 'LWA_gm2', 'LWC_%', 'd15N', 'd13C', 'N_weight_percent', 'C_weight_percent', 'CN', 'CalVal', 'ID','X_UTM','Y_UTM','h_me_B_1','wtrl_B_1','wtrv_B_1','_obs_B_1','_obs_B_2','_obs_B_3','_obs_B_4','_obs_B_5','_obs_B_6','_obs_B_7','_obs_B_8','_obs_B_9','_obs_B_10','ered_B_1','_tch_B_1','refl_B_1','refl_B_2','refl_B_3','refl_B_4','refl_B_5','refl_B_6','refl_B_7','refl_B_8','refl_B_9','refl_B_10','refl_B_11','refl_B_12','refl_B_13','refl_B_14','refl_B_15','refl_B_16','refl_B_17','refl_B_18','refl_B_19','refl_B_20','refl_B_21','refl_B_22','refl_B_23','refl_B_24','refl_B_25','refl_B_26','refl_B_27','refl_B_28','refl_B_29','refl_B_30','refl_B_31','refl_B_32','refl_B_33','refl_B_34','refl_B_35','refl_B_36','refl_B_37','refl_B_38','refl_B_39','refl_B_40','refl_B_41','refl_B_42','refl_B_43','refl_B_44','refl_B_45','refl_B_46','refl_B_47','refl_B_48','refl_B_49','refl_B_50','refl_B_51',
                                   'refl_B_52','refl_B_53','refl_B_54','refl_B_55','refl_B_56','refl_B_57','refl_B_58','refl_B_59','refl_B_60','refl_B_61','refl_B_62','refl_B_63','refl_B_64','refl_B_65','refl_B_66','refl_B_67','refl_B_68','refl_B_69','refl_B_70','refl_B_71','refl_B_72','refl_B_73','refl_B_74','refl_B_75','refl_B_76','refl_B_77','refl_B_78','refl_B_79','refl_B_80','refl_B_81','refl_B_82','refl_B_83','refl_B_84','refl_B_85','refl_B_86','refl_B_87','refl_B_88','refl_B_89','refl_B_90','refl_B_91','refl_B_92','refl_B_93','refl_B_94','refl_B_95','refl_B_96','refl_B_97','refl_B_98','refl_B_99','refl_B_100','refl_B_101','refl_B_102','refl_B_103','refl_B_104','refl_B_105','refl_B_106','refl_B_107','refl_B_108','refl_B_109','refl_B_110','refl_B_111','refl_B_112','refl_B_113','refl_B_114','refl_B_115','refl_B_116','refl_B_117','refl_B_118','refl_B_119','refl_B_120','refl_B_121','refl_B_122','refl_B_123','refl_B_124','refl_B_125','refl_B_126','refl_B_127','refl_B_128','refl_B_129',
                                   'refl_B_130','refl_B_131','refl_B_132','refl_B_133','refl_B_134','refl_B_135','refl_B_136','refl_B_137','refl_B_138','refl_B_139','refl_B_140','refl_B_141','refl_B_142','refl_B_143','refl_B_144','refl_B_145','refl_B_146','refl_B_147','refl_B_148','refl_B_149','refl_B_150','refl_B_151','refl_B_152','refl_B_153','refl_B_154','refl_B_155','refl_B_156','refl_B_157','refl_B_158','refl_B_159','refl_B_160','refl_B_161','refl_B_162','refl_B_163','refl_B_164','refl_B_165','refl_B_166','refl_B_167','refl_B_168','refl_B_169','refl_B_170','refl_B_171','refl_B_172','refl_B_173','refl_B_174','refl_B_175','refl_B_176','refl_B_177','refl_B_178','refl_B_179','refl_B_180','refl_B_181','refl_B_182','refl_B_183','refl_B_184','refl_B_185','refl_B_186','refl_B_187','refl_B_188','refl_B_189','refl_B_190','refl_B_191','refl_B_192','refl_B_193','refl_B_194','refl_B_195','refl_B_196','refl_B_197','refl_B_198','refl_B_199','refl_B_200','refl_B_201','refl_B_202','refl_B_203','refl_B_204',
                                   'refl_B_205','refl_B_206','refl_B_207','refl_B_208','refl_B_209','refl_B_210','refl_B_211','refl_B_212','refl_B_213','refl_B_214','refl_B_215','refl_B_216','refl_B_217','refl_B_218','refl_B_219','refl_B_220','refl_B_221','refl_B_222','refl_B_223','refl_B_224','refl_B_225','refl_B_226','refl_B_227','refl_B_228','refl_B_229','refl_B_230','refl_B_231','refl_B_232','refl_B_233','refl_B_234','refl_B_235','refl_B_236','refl_B_237','refl_B_238','refl_B_239','refl_B_240','refl_B_241','refl_B_242','refl_B_243','refl_B_244','refl_B_245','refl_B_246','refl_B_247','refl_B_248','refl_B_249','refl_B_250','refl_B_251','refl_B_252','refl_B_253','refl_B_254','refl_B_255','refl_B_256','refl_B_257','refl_B_258','refl_B_259','refl_B_260','refl_B_261','refl_B_262','refl_B_263','refl_B_264','refl_B_265','refl_B_266','refl_B_267','refl_B_268','refl_B_269','refl_B_270','refl_B_271','refl_B_272','refl_B_273','refl_B_274','refl_B_275','refl_B_276','refl_B_277','refl_B_278','refl_B_279',
                                   'refl_B_280','refl_B_281','refl_B_282','refl_B_283','refl_B_284','refl_B_285','refl_B_286','refl_B_287','refl_B_288','refl_B_289','refl_B_290','refl_B_291','refl_B_292','refl_B_293','refl_B_294','refl_B_295','refl_B_296','refl_B_297','refl_B_298','refl_B_299','refl_B_300','refl_B_301','refl_B_302','refl_B_303','refl_B_304','refl_B_305','refl_B_306','refl_B_307','refl_B_308','refl_B_309','refl_B_310','refl_B_311','refl_B_312','refl_B_313','refl_B_314','refl_B_315','refl_B_316','refl_B_317','refl_B_318','refl_B_319','refl_B_320','refl_B_321','refl_B_322','refl_B_323','refl_B_324','refl_B_325','refl_B_326','refl_B_327','refl_B_328','refl_B_329','refl_B_330','refl_B_331','refl_B_332','refl_B_333','refl_B_334','refl_B_335','refl_B_336','refl_B_337','refl_B_338','refl_B_339','refl_B_340','refl_B_341','refl_B_342','refl_B_343','refl_B_344','refl_B_345','refl_B_346','refl_B_347','refl_B_348','refl_B_349','refl_B_350','refl_B_351','refl_B_352','refl_B_353','refl_B_354',
                                   'refl_B_355','refl_B_356','refl_B_357','refl_B_358','refl_B_359','refl_B_360','refl_B_361','refl_B_362','refl_B_363','refl_B_364','refl_B_365','refl_B_366','refl_B_367','refl_B_368','refl_B_369','refl_B_370','refl_B_371','refl_B_372','refl_B_373','refl_B_374','refl_B_375','refl_B_376','refl_B_377','refl_B_378','refl_B_379','refl_B_380','refl_B_381','refl_B_382','refl_B_383','refl_B_384','refl_B_385','refl_B_386','refl_B_387','refl_B_388','refl_B_389','refl_B_390','refl_B_391','refl_B_392','refl_B_393','refl_B_394','refl_B_395','refl_B_396','refl_B_397','refl_B_398','refl_B_399','refl_B_400','refl_B_401','refl_B_402','refl_B_403','refl_B_404','refl_B_405','refl_B_406','refl_B_407','refl_B_408','refl_B_409','refl_B_410','refl_B_411','refl_B_412','refl_B_413','refl_B_414','refl_B_415','refl_B_416','refl_B_417','refl_B_418','refl_B_419','refl_B_420','refl_B_421','refl_B_422','refl_B_423','refl_B_424','refl_B_425','refl_B_426'])

    longdf.rename(columns={'variable': 'Element'}, inplace=True)
    longdf.rename(columns={'value': "Concentration"}, inplace=True)
    longdf.drop_duplicates(inplace=True)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):

    eles = longdf['Element'].unique()
    type = longdf['Genus'].unique()


    for e in range(len(eles)):
        print("Plotting:" + eles[e])
        temp_df = longdf[longdf['Element'] == eles[e]]
        dataset = []
        labels = []
        colorindex = []
        minimum = []
        maximum = []

        fig, ax = plt.subplots(figsize=(12, 7))

        for t in range(len(type)):
            print(type[t])
            temp_type_df = temp_df[temp_df['Genus'] == type[t]]
            veg = temp_type_df['Site_Veg'].unique()


            for v in range(len(veg)):
                print(veg[v])
                species = temp_type_df[temp_type_df['Site_Veg'] == veg[v]]['Concentration']
                minimum.append(np.min(species))
                maximum.append(np.max(species))
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

        plt.savefig('../east_river_trait_modeling/metals_v12/figures/quantile_plots/' + eles[e] + '.png')

        return longdf

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
    site_codes = np.array(extract['SampleSiteCode'])



    return spectra, cover_types, wv, genus_types, site_codes


def plot_avg_quantile(spectra, longdf, cover_types, genus_types, site_codes, wv, brightness_normalize=True):

    g_types = genus_types.unique()
    c_types = cover_types.unique()

    color_sets = ['navy', 'purple', 'forestgreen', 'royalblue', 'gray', 'darkorange', 'black', 'brown', 'tan']

    # Plot the difference between needles and noneedles in reflectance data

    eles = ['1C', 'As', 'Co', 'Cr', 'Cu', 'Mo', 'Ni', 'Sr', 'Zn']
    intervals = [0, .2, .4, .6, .8, 1]


    for _c, cover in enumerate(c_types):
        for e in range(len(eles)):
            temp_c = longdf[longdf['Site_Veg'] == c_types[_c]]
            temp_e = temp_c[temp_c['Element'] == eles[e]]
            quantile_intervals = []



            # select dataset that we want to calcualte quantiles of - Chems at the sites of cover types selected
            for q in intervals:
                # save quantile values into a vector
                quantile = np.array(np.nanquantile(temp_e['Concentration'], q))
                quantile_intervals.append(quantile)

            # do spectra selection
            # c_spectra = spectra[cover_types == cover]  # arg for threshold
            # print(list(c_spectra))

            #start the plot here
            figure_export_settings = {'dpi': 200, 'bbox_inches': 'tight'}
            fig = plt.figure(figsize=(20, 10), constrained_layout=True)

            for i in range(len(quantile_intervals)-1):
                #identifying only the sites in this metal concentration quantile
                print(i)
                sites_temp = temp_e.loc[(temp_e['Concentration']>quantile_intervals[i]) & (temp_e['Concentration']<quantile_intervals[i+1])]['SampleSiteCode']
                conc = temp_e['Concentration']
                # boolean array length
                valid = np.zeros(spectra.shape[0],dtype=bool)
                for site in sites_temp:
                    valid[site_codes == site] = True

                #select spectra from these sites only
                c_spectra = spectra[valid,:]


                if brightness_normalize == True:
                    scale_factor = 1.
                else:
                    scale_factor = 100.

                plt.plot(wv, np.nanmean(c_spectra, axis=0) / scale_factor, c=color_sets[i], linewidth=2)
                plt.fill_between(wv, np.nanmean(c_spectra, axis=0) / scale_factor - np.nanstd(c_spectra, axis=0) / scale_factor,
                                 np.nanmean(c_spectra, axis=0) / scale_factor + np.nanstd(c_spectra, axis=0) / scale_factor, alpha=.15,
                                 facecolor=color_sets[i])

            plt.legend(quantile_intervals, prop={'size': 20})
            plt.ylabel('Reflectance (%)')


            if brightness_normalize:
                plt.ylabel('Brightness Norm. Reflectance')
            else:
                plt.ylabel('Reflectance (%)')
            plt.xlabel('Wavelength (nm)')

            plt.savefig(os.path.join('metals_v12', 'figures', 'quantile_plots', 'spectra',cover+eles[e]+'.png'), **figure_export_settings)
            del fig

                # #plot lines here


        # for each quantile loop through plotting code - select only c_spectra that are in each quantile range
        #         for quantiles
        #         print(color_sets[_c])



if __name__ == "__main__":
    main()