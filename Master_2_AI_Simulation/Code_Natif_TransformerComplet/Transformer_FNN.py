import warnings
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from transformer import Encoder, padding_mask
from load_domain import loadDomain
from transformers import TFBertModel, BertTokenizer

# Ignorer les avertissements
warnings.filterwarnings("ignore")

# Configuration d'entraînement
class TrainingConfig(object):
    epochs = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

# Configuration du modèle
class ModelConfig(object):
    embeddingSize = 128
    filters = 128
    numHeads = 8
    numBlocks = 1
    epsilon = 1e-8
    keepProp = 0.9
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

# Configuration générale
class Config(object):
    sequenceLength = 40
    batchSize = 32
    max_features = 95
    num_features = 37
    numClasses = 1

    training = TrainingConfig()
    model = ModelConfig()

if __name__ == "__main__":
    # Charger les données
    X_train = pd.read_csv('data/train.csv', header=None)
    Y_train = pd.read_csv('data/Y_train.csv', header=None)
    X_test = pd.read_csv('data/test.csv', header=None)
    Y_test = pd.read_csv('data/Y_test.csv', header=None)
    domain_train = pd.read_csv('data/domain_train.csv', header=0)
    domain_test = pd.read_csv('data/domain_test.csv', header=0)

    domain_train = loadDomain(domain_train)
    domain_test = loadDomain(domain_test)

    # Configuration
    config = Config()

    # Prétraitement des données
    X_train = X_train.values.reshape(len(X_train), config.num_features, 1)
    X_test = X_test.values.reshape(len(X_test), config.num_features, 1)
    domain_train = sequence.pad_sequences(domain_train, maxlen=config.sequenceLength)
    domain_test = sequence.pad_sequences(domain_test, maxlen=config.sequenceLength)

    # Construire le modèle Transformer
    inputs_A = Input(shape=(config.sequenceLength,), dtype='int32')
    embeddings = Embedding(config.max_features, config.model.embeddingSize)(inputs_A)
    mask_inputs = padding_mask(inputs_A)
    out_seq = Encoder(
        n_layers=4, d_model=128, num_heads=4,
        middle_units=256, max_seq_len=config.sequenceLength)([embeddings, mask_inputs])
    out_seq = GlobalAveragePooling1D()(out_seq)
    out_seq = Dropout(0.3)(out_seq)
    outputs_A = Dense(64, activation='softmax')(out_seq)

    # Construire le modèle CNN
    inputs_B = Input(shape=(config.num_features, 1), dtype='float32')
    c1 = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs_B)
    m1 = MaxPool1D(pool_size=3, strides=3)(c1)
    c2 = Conv1D(filters=32, kernel_size=2, activation='relu')(m1)
    m2 = MaxPool1D(pool_size=3, strides=3)(c2)
    d1 = Dropout(0.3)(m2)
    f1 = Flatten()(d1)
    outputs_B = Dense(64, activation='softmax')(f1)

    # Concaténer les sorties des deux branches
    outputs = concatenate([outputs_A, outputs_B], 1)
    outputs = Dense(4, activation='softmax')(outputs)

    # Construire le modèle final
    model = Model(inputs=[inputs_A, inputs_B], outputs=outputs)
    print(model.summary())

    # Compiler le modèle
    opt = Adam(lr=0.0001, decay=0.00001)
    loss = 'sparse_categorical_crossentropy'
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    # Entraîner le modèle
    print('Entraînement...')
    model.fit([domain_train, X_train], Y_train, batch_size=config.batchSize, epochs=100, validation_split=0.2)

    # Prédictions sur les données de test
    predict_test = model.predict([domain_test, X_test])
    predict = np.argmax(predict_test, axis=1)
    
    # Afficher le rapport de classification
    print(classification_report(Y_test, predict, digits=5))
