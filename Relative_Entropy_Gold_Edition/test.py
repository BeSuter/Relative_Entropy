from RelativeEntropyFunctions import MonteCarloENTROPY
from RelativeEntropyFunctions import LoadAndComputeEntropy

Auto_Sim = "/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/4probes_auto_samples_sim.npy"
Cross_Sim = "/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/4probes_cross_samples_sim.npy"

Auto_G = "/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/4probes_auto_samples_Gaussian.npy"
Cross_G = "/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/4probes_cross_samples_Gaussian.npy"

Auto_Sim_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/tt_dd_gg_kk_samples_sim_nonuisance.npy'
Cross_Sim_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/tt_td_dd_dg_gg_gk_kk_samples_sim_nonuisance.npy'

Auto_G_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/tt_dd_gg_kk_samples_Gaussian_nonuisance.npy'
Cross_G_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/tt_td_dd_dg_gg_gk_kk_samples_Gaussian_nonuisance.npy'

dd_gg_kk_sim_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/dd_gg_kk_samples_sim_nonuisance.npy'
dd_dg_gg_gk_kk_sim_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/dd_dg_gg_gk_kk_samples_sim_nonuisance.npy'

dd_gg_kk_gauss_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/dd_gg_kk_samples_Gaussian_nonuisance.npy'
dd_dg_gg_gk_kk_gauss_noN = '/Users/BenjaminSuter/Documents/My Files/Relative Entropy/Samples/dd_dg_gg_gk_kk_samples_Gaussian_nonuisance.npy'

if __name__ == "__main__":
    """mc1 = MonteCarloENTROPY(Auto_G, Cross_G, 5000)
    mc2 = MonteCarloENTROPY(Auto_Sim, Cross_Sim, 5000)
    mc3 = MonteCarloENTROPY(Auto_G_noN, Cross_G_noN, 5000)
    mc4 = MonteCarloENTROPY(Auto_Sim_noN, Cross_Sim_noN, 5000)
    print(mc1)
    print(mc2)
    print(mc3)
    print(mc4)"""
    ent1 = LoadAndComputeEntropy(Auto_G, Cross_G, steps=100)
    ent2 = LoadAndComputeEntropy(Auto_Sim, Cross_Sim, steps=100)
    ent3 = LoadAndComputeEntropy(Auto_G_noN, Cross_G_noN, steps=100)
    ent4 = LoadAndComputeEntropy(Auto_Sim_noN, Cross_Sim_noN, steps=100)
    print(ent1["rel_ent"])
    print(ent2["rel_ent"])
    print(ent3["rel_ent"])
    print(ent4["rel_ent"])
    print(LoadAndComputeEntropy(dd_gg_kk_sim_noN, dd_dg_gg_gk_kk_sim_noN, steps=100)["rel_ent"])