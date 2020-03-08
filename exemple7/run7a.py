from skopt import Optimizer
import numpy as np

def dbtime(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10):
    return np.sin(p1)*10+np.sin(p2)*5+np.sin(p3)+np.sin(p4)+np.sin(p5)+np.sin(p6)+np.sin(p7)+np.sin(p8)+np.sin(p9)+np.sin(p10)+3*8+10+5



search_space = {
  'param_1': (0.0, 6.0),
  'param_2': (0.0, 6.0),
  'param_3': (0.0, 6.0),
  'param_4': (0.0, 6.0),
  'param_5': (0.0, 6.0),
  'param_6': (0.0, 6.0),
  'param_7': (0.0, 6.0),
  'param_8': (0.0, 6.0),
  'param_9': (0.0, 6.0),
  'param_10': (0.0, 6.0)    
}

best_runtime=1000
best_config=[]

opt = Optimizer([search_space['param_1'], search_space['param_2'], search_space['param_3'], search_space['param_4'], search_space['param_5'], search_space['param_6'], search_space['param_7'], search_space['param_8'], search_space['param_9'], search_space['param_10']],  "GP", n_initial_points=3)
for iteration  in range(100):
    next_config = opt.ask() 
    print(next_config)
    runtime=dbtime(next_config[0],next_config[1],next_config[2],next_config[3],next_config[4],next_config[5],next_config[6],next_config[7],next_config[8],next_config[9])
    if runtime < best_runtime:
        best_runtime=runtime
        best_config=next_config
    opt.tell(next_config, runtime)

print(best_runtime)
print(best_config)
