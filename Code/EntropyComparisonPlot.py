import sys
from Plotter import YAMLplot

'''if __name__ == "__main__":
    args = sys.argv[1:]
    data_path = args[0]
    target_path = args[1]
    YAMLplot(data_path, target_path)'''

"""file_names = ["../YAML/ThreeProbesData_burn=0.6_avrg=15_MCsteps=120000_noN.yml",
              "../YAML/ThreeProbesData_burn=0.6_avrg=15_MCsteps=120000.yml",
              "../YAML/ThreeProbesData_burn=0.7_avrg=15_MCsteps=120000_noN.yml",
              "../YAML/ThreeProbesData_burn=0.7_avrg=15_MCsteps=120000.yml",
              "../YAML/ThreeProbesData_burn=0.8_avrg=15_MCsteps=120000_noN.yml",
              "../YAML/ThreeProbesData_burn=0.8_avrg=15_MCsteps=120000.yml",
              "../YAML/TwoProbesData_burn=0.6_avrg=15_MCsteps=120000_noN.yml",
              "../YAML/TwoProbesData_burn=0.6_avrg=15_MCsteps=120000.yml",
              "../YAML/TwoProbesData_burn=0.7_avrg=15_MCsteps=120000_noN.yml",
              "../YAML/TwoProbesData_burn=0.7_avrg=15_MCsteps=120000.yml",
              "../YAML/TwoProbesData_burn=0.8_avrg=15_MCsteps=120000_noN.yml",
              "../YAML/TwoProbesData_burn=0.8_avrg=15_MCsteps=120000.yml"
              ]"""
file_names = ["../YAML/FourProbesData_avrg5_MCsteps800000.yml"]

for path in file_names:
    target = f"../Plots/{path[8:-4]}_testTEST.pdf"
    YAMLplot(path, target)