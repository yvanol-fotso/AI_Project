 
# les premier model


 # def train_CNN_Primary(self):

 #   # Chemin vers le dossier contenant les images des étudiants
 #   data_dir = "data2/"

 #   # Prétraitement des images
 #   image_size = (224, 224)
 #   batch_size = 16

 #   # Génération de données d'entraînement
 #   datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

 #   train_generator = datagen.flow_from_directory(
 #    data_dir,
 #    target_size=image_size,
 #    batch_size=batch_size,
 #    class_mode='binary',
 #    subset='training'
 #   )

 #   validation_generator = datagen.flow_from_directory(
 #    data_dir,
 #    target_size=image_size,
 #    batch_size=batch_size,
 #    class_mode='binary',
 #    subset='validation'
 #   )

 #   # Définition du modèle CNN
 #   model = tf.keras.Sequential([
 #    tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
 #    tf.keras.layers.GlobalAveragePooling2D(),
 #    tf.keras.layers.Dense(1, activation='sigmoid')
 #   ])

 #   # Compilation du modèle
 #   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 #   # Entraînement du modèle
 #   epochs = 10

 #   model.fit(
 #    train_generator,
 #    steps_per_epoch=train_generator.samples // batch_size,
 #    validation_data=validation_generator,
 #    validation_steps=validation_generator.samples // batch_size,
 #    epochs=epochs
 #   )

 #   # Sauvegarde du modèle entraîné
 #   model.save('modelCNN/model_1.h5')
 #   speak_va("Training datasets completed successfully!")
 #   messagebox.showinfo("Result","Training datasets completed successfully!",parent=self.root)



 # def train_CNN_Secondary(self):

 #  # Définition des chemins des données d'entraînement
 #  train_data_dir = 'data2/'
 #  validation_data_dir = 'data2/'

 #  # Paramètres d'entraînement
 #  batch_size = 32
 #  epochs = 10
 #  # num_classes = len(os.listdir(train_data_dir))


 #  # Prétraitement des données d'entraînement
 #  train_datagen = ImageDataGenerator(
 #    rescale=1./255,
 #    rotation_range=20,
 #    width_shift_range=0.2,
 #    height_shift_range=0.2,
 #    shear_range=0.2,
 #    zoom_range=0.2,
 #    horizontal_flip=True,
 #    fill_mode='nearest'
 #  )

 #  train_generator = train_datagen.flow_from_directory(
 #    train_data_dir,
 #    target_size=(224, 224),
 #    batch_size=batch_size,
 #    class_mode='categorical',
 #    shuffle=True
 #  )

 #  num_classes = len(train_generator.class_indices) #fouille dans le dossier "data" , tous les ous dossier se nommant class_i [i = indice]


 #  # Prétraitement des données de validation
 #  validation_datagen = ImageDataGenerator(rescale=1./255)

 #  validation_generator = validation_datagen.flow_from_directory(
 #    validation_data_dir,
 #    target_size=(224, 224),
 #    batch_size=batch_size,
 #    class_mode='categorical',
 #    shuffle=False
 #  )

 #  # Chargement du modèle pré-entraîné MobileNetV2
 #  base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

 #  # Ajout de nouvelles couches de classification
 #  x = base_model.output
 #  x = GlobalAveragePooling2D()(x)
 #  x = Dropout(0.5)(x)
 #  x = Dense(512, activation='relu')(x)
 #  predictions = Dense(num_classes, activation='softmax')(x)

 #  # Construction du modèle final
 #  model = Model(inputs=base_model.input, outputs=predictions)

 #  # Geler les couches du modèle pré-entraîné
 #  for layer in base_model.layers:
 #      layer.trainable = False

 #  # Compilation du modèle
 #  model.compile(optimizer=Adam(lr=0.001),
 #              loss='categorical_crossentropy',
 #              metrics=['accuracy'])

 #  # Entraînement du modèle
 #  model.fit(train_generator,
 #          steps_per_epoch=train_generator.n // batch_size,
 #          epochs=epochs,
 #          validation_data=validation_generator,
 #          validation_steps=validation_generator.n // batch_size)

 #  # Sauvegarde du modèle entraîné
 #  model.save('modelCNN/model_2.h5')


 





 #Definition de l'architecture du reseau ou encore le model ce model est configuer pour predire 6 classe donc c'est a dire que si on ajoute une ettique=.POUR UN NOUVEAU ETUDIANT NON EXISTANT 
    # il faut augmenter le nombre de classe a predire a la sortie : pour cela il suffit de modifier le nombre de neurone dans la couche de sortie qui ici est la couche Dense

    # model=keras.Sequential([
    #   keras.Input(shape=(300,300,3)),
    #   layers.Conv2D(16,kernel_size=(3,3),
    #     activation='relu'),
    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Conv2D(32,kernel_size=(3,3),
    #     activation='tanh'),

    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Flatten(),
    #   layers.Dropout(0.5),
    #   layers.Dense(6,activation='softmax'), 
    # ])




 
  #----------------------------------- deuxieme apres augmentation des classes ---------------------##
  #---- trouve 1 sur 5 images non utiliser pour l'entrainement ni test ------#


    #puisque les nouveau etiquette varient de o - 7 alors ma couche dense doit avoir 8 neurones  car les valeurs des etiquette doivent appartenir dans [0, num_classes-1].
    # or actuellement j'ai 8 etudiant different ie 8 matricule diffenrent or puisque les etiqutte begin de 0 alors j'aurais 8 neurone en sortie car chaque etudiant predit par une classe donc il faut 8 classes or de 0--7 donne 8

    # model=keras.Sequential([
    #   keras.Input(shape=(300,300,3)),
    #   layers.Conv2D(16,kernel_size=(3,3), activation='relu'),
    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Conv2D(32,kernel_size=(3,3),activation='tanh'),
    #   layers.MaxPooling2D(pool_size=(2,2)),
    #   layers.Flatten(),
    #   layers.Dropout(0.5),
    #   layers.Dense(8,activation='softmax'), 
    # ])

   
    # #compilon notre model
    # model.compile(loss="sparse_categorical_crossentropy",
    #           optimizer="adam",
    #           metrics=["accuracy"]) 





    #------------------------------------- changeaons de model sa fait planter le pc---------------------------######
    #---- trouve 3 sur 5 images non utiliser pour l'entrainement ni test ------#


    # model = keras.Sequential([
    #    keras.Input(shape=(300, 300, 3)),
    #    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Flatten(),
    #    layers.Dropout(0.5),
    #    layers.Dense(8, activation='softmax')
    # ])

    # model.compile(loss="sparse_categorical_crossentropy",
    #           optimizer="adam",
    #           metrics=["accuracy"])







    #---------------------- verifions ceci en ajoutant les " layers.BatchNormalization()" ---------------#
    # trouve tous tout mais en faux positif sur fokou donc c'est tres mauvais ce model

    # model = keras.Sequential([
    #    keras.Input(shape=(300, 300, 3)),
    #    layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
    #    layers.BatchNormalization(),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #    layers.BatchNormalization(),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #    layers.BatchNormalization(),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Flatten(),
    #    layers.Dropout(0.5),
    #    layers.Dense(8, activation='softmax')
    # ])

    # model.compile(loss="sparse_categorical_crossentropy",
    #           optimizer="adam",
    #           metrics=["accuracy"])





    ### ---------------------------- lent ce n'est pas un bon model met 1h pour predire 3 et faux positif ----#

    # model = keras.Sequential([
    #    keras.Input(shape=(300, 300, 3)),
    #    layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    #    layers.MaxPooling2D(pool_size=(2, 2)),
    #    layers.Flatten(),
    #    layers.Dropout(0.5),
    #    layers.Dense(8, activation='softmax')
    # ])

    # model.compile(loss="sparse_categorical_crossentropy",
    #           optimizer="adam",
    #           metrics=["accuracy"])







    #------------------------- architecture utilisant les reseau de neurone convolutif DCNN --------#  
    # les entreers sont bcp 32 va faire planter la machine je vais reduire tous sa ---- #####3

    # model = keras.Sequential([
    #   keras.Input(shape=(300, 300, 3)),
    #   layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    #   layers.BatchNormalization(),
    #   layers.MaxPooling2D(pool_size=(2, 2)),
    #   layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #   layers.BatchNormalization(),
    #   layers.MaxPooling2D(pool_size=(2, 2)),
    #   layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    #   layers.BatchNormalization(),
    #   layers.MaxPooling2D(pool_size=(2, 2)),
    #   layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    #   layers.BatchNormalization(),
    #   layers.MaxPooling2D(pool_size=(2, 2)),
    #   layers.Flatten(),
    #   layers.Dropout(0.5),
    #   layers.Dense(256, activation='relu'),
    #   layers.Dropout(0.5),
    #   layers.Dense(8, activation='softmax')
    # ])

    # model.compile(loss="sparse_categorical_crossentropy",
    #           optimizer="adam",
    #           metrics=["accuracy"])


