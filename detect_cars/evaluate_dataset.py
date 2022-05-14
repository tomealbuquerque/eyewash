import os
from detect_cars import detect_objects,filter_cars_detected,calculate_area_bbox
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



if __name__ == "__main__":
    dataset_path = 'dirty_cars/'

    files = os.listdir(dataset_path)
    
    areas = []
    for f in files:
      print(f)
      #Detect objects
      res = detect_objects(os.getcwd() + "/" + dataset_path + f)
      #Filter by cars
      boxes_cars = filter_cars_detected(res)
      for b in boxes_cars:
        a = calculate_area_bbox(b)
        #print("Area ", a)
        areas.append(a)
    
    areas_df = pd.DataFrame(areas)
    areas_df.describe()
    areas_df.describe().to_csv("Areas_analysis.csv")

    #Plot histogram      
    plt.hist(areas)
    plt.xlabel('Areas of bouding boxes')
    plt.ylabel('Count')
    plt.savefig('histogram_areas_dirty_cars.png')
    plt.show()
