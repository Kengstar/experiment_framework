import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

def main ():
    '''This is a sample template for a first inspection of the dataset, needs to be adjusted individually.
       A jupyter notebook is highly encouraged
    '''


    ## load data
    train_set = pd.read_json("train.json")
    print ("Number of Training Samples: " + str(train_set.shape[0]))
    print ("Names of available features: " + str(list(train_set.columns.values)))

    ## information about our dataset

    is_iceberg = train_set["is_iceberg"] == 1
    iceberg_images = train_set[is_iceberg]
    ship_images = train_set[np.invert(is_iceberg)]
    print ("Iceberg Images: " + str( len(iceberg_images)))
    print ("Ship Images: " + str(len(ship_images)))


    labels =  'Icebergs' ,  'Ships'
    sizes = [753, 851 ]
    colors = ['green', 'red']
        # Plot
    plt.pie(sizes,  labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()


    labels =  train_set["is_iceberg"]
    inc_angle = train_set["inc_angle"]



    ##analyze missing values
    idx_of_missing_values = (train_set["inc_angle"] =='na')
    nr_of_missing_values = idx_of_missing_values.sum()
    print ("Missing Angles: " + str((nr_of_missing_values)))
    angles_of_icebergs = iceberg_images["inc_angle"]


    ## give statistics
    print (angles_of_icebergs.describe())
    print (angles_of_icebergs.max())
    print (angles_of_icebergs.min())
    print (angles_of_icebergs.mean())
    print (angles_of_icebergs.std())



    angles_of_ships = ship_images["inc_angle"]
    print ("Ships")
    print (angles_of_ships.describe())
    angles_of_ships = angles_of_ships[angles_of_ships != 'na']
    print ("after replacing na")
    print (angles_of_ships.describe())
    print (angles_of_ships.max())
    print (angles_of_ships.min())
    print (angles_of_ships.mean())
    print (angles_of_ships.std())



    plt.plot(angles_of_icebergs,np.zeros(753), 'o')
    plt.show()


    ##### data preparation
    band_1 = np.concatenate([im for im in train_set['band_1']]).reshape(-1, 75,75)
    band_2 = np.concatenate([im for im in train_set['band_2']]).reshape(-1, 75, 75)

    min1 = np.amin(band_1)
    max1 = np.amax(band_1)

    min2 = np.amin(band_2)
    max2 = np.amax(band_2)

    print (min1,max1,"\n",min2,max2)



    ### plot random datapoints
    #np.random.seed(777)
    band1_ice = band_1[is_iceberg]
    rand_idx = np.random.permutation(band1_ice.shape[0])
    rand_idx = rand_idx[:3]
    band2_ice = band_2[is_iceberg]
    band1_ice = band1_ice[rand_idx,:]
    band2_ice = band2_ice[rand_idx,:]

    farbe = 'inferno'
    fig, axs = plt.subplots(3, 2, figsize=(15,15))

    axs[0,0].imshow(band1_ice[0], cmap=farbe)
    axs[0,1].imshow(band2_ice[0], cmap=farbe)
    axs[1,0].imshow(band1_ice[1], cmap=farbe)
    axs[1,1].imshow(band2_ice[1], cmap=farbe)
    axs[2,0].imshow(band1_ice[2], cmap=farbe)
    axs[2,1].imshow(band2_ice[2], cmap=farbe)
    plt.show()









    band1_ships = band_1[np.invert(is_iceberg)]
    band2_ships = band_2[np.invert(is_iceberg)]


    band1_ships = band1_ships[rand_idx, :]
    band2_ships = band2_ships[rand_idx, :]

    fig, axs = plt.subplots(3, 2, figsize=(15,15))
    axs[0,0].imshow(band1_ships[0], cmap=farbe)
    axs[0,1].imshow(band2_ships[0], cmap=farbe)
    axs[1,0].imshow(band1_ships[1], cmap=farbe)
    axs[1,1].imshow(band2_ships[1], cmap=farbe)
    axs[2,0].imshow(band1_ships[2], cmap=farbe)
    axs[2,1].imshow(band2_ships[2], cmap=farbe)
    plt.show()



    ### also plot different feature transformations




if __name__ == "__main__":
    main()