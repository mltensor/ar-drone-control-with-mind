from final_model import final_model
import pandas as pd
import torch
import numpy as np

df = pd.read_csv('C:\\Users\\Suryansh\\VS_Codes\\drone_practicum\\final_dataset_for_two_directions.csv')

# defining the variables

x_train = df.drop(['Values'], axis=1)
x_train = torch.from_numpy(np.array(x_train).astype(np.float32))
print(final_model(x_train[40]))

print("Please enter the value between 0 and 53 (both included)")
variable = int(input())

while(variable!='q' or variable!='Q'):
    print(final_model(x_train[variable]))
    with open('data.txt', 'w') as f:
        f.write(str(final_model(x_train[variable])))
    print("Press 'Q' or 'q' to exit")
    variable = input()
    if(variable=='q' or variable=='Q'):
        break
    else:
        variable=int(variable)