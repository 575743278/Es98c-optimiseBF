import matlab.engine
# run blast furnace simulation
def simulation_generate_results(output_filename):
  
    eng = matlab.engine.start_matlab()

    eng.parfor_simulate_map1(nargout=0)
  
    eng.quit()
    

if __name__ == "__main__":
    output_filename ='blast_furnace_results1.csv'

    simulation_generate_results(output_filename)
