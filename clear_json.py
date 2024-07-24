from os import listdir                                               
from os.path import isfile, join                                     
import json                                                          
import pdb                                                           
                                                                     
mypath = 'output_gunshoot_june_txx/'                                
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]  
counter = 0                                                          
for indv_file in all_files:                                          
    error = False                                                    
    file_path = mypath+indv_file                                     
    f = open(file_path)                                              
    try:                                                             
        data=json.load(f)                                            
    except:                                                          
        error = True                                                 
        #print(file_path)                                            
    print(file_path)                                                 
    f.close()                                                        
    counter += 1                                                     
                                                                     
    if error==True:                                                  
        #pdb.set_trace()                                             
        my_file = open(file_path)                                    
        data = my_file.read()                                        
        my_file.close()                                              
        if data[-2]==',':                                            
            out_data = data[:-2]+']'                                 
            my_file = open(file_path,'w')                            
            my_file.write(out_data)                                  
            my_file.close()                                          
        print(indv_file)                                             
        print(data)                                                  
        print(out_data)                                              
        print(counter)  
