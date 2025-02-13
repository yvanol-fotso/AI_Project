import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class Encodeur(tf.keras.layers.Layer):
    def __init__(self, n_couches, d_model, num_heads, middle_units,
                 max_seq_len, epsilon=1e-6, taux_dropout=0.1, entrainement=False, **kwargs):
        super(Encodeur, self).__init__(**kwargs)
        self.n_couches = n_couches

        self.embedding_position = EncodagePosition(sequence_len=max_seq_len, embedding_dim=d_model)
        self.couche_encodeur = [CoucheEncodeur(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len,
                                               middle_units=middle_units,
                                               epsilon=epsilon, taux_dropout=taux_dropout,
                                               entrainement=entrainement)
                                for _ in range(n_couches)]

    def call(self, entrees, **kwargs):
        emb, masque = entrees
        emb = self.embedding_position(emb)
        for i in range(self.n_couches):
            emb = self.couche_encodeur[i](emb, masque)

        return emb


# Couche d'encodeur
class CoucheEncodeur(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_seq_len, middle_units, epsilon=1e-6, taux_dropout=0.1, entrainement=False, **kwargs):
        super(CoucheEncodeur, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(num_heads)
        # self.ffn = reseau_neuronal_couche_avant(d_model + max_seq_len, middle_units)

        self.ffn = reseau_neuronal_couche_avant(d_model, middle_units)  # Utilisez d_model au lieu de (d_model + max_seq_len)

        # self.ffn = reseau_neuronal_couche_avant(d_model, middle_units)
        self.normalisation_couche1 = NormalisationCouche()
        self.normalisation_couche2 = NormalisationCouche()

        self.dropout1 = tf.keras.layers.Dropout(taux_dropout)
        self.dropout2 = tf.keras.layers.Dropout(taux_dropout)

        self.entrainement = entrainement

    def call(self, entrees, masque, **kwargs):
        # Réseau de neurones à attention multiple
        att_sortie = self.mha([entrees, entrees, entrees, masque])
        att_sortie = self.dropout1(att_sortie, training=self.entrainement)
        sortie1 = self.normalisation_couche1(entrees + att_sortie)  # (taille_lot, long_seq_entree, d_model)
        # Réseau neuronal avant
        ffn_sortie = self.ffn(sortie1)
        ffn_sortie = self.dropout2(ffn_sortie, training=self.entrainement)
        sortie2 = self.normalisation_couche2(sortie1 + ffn_sortie)  # (taille_lot, long_seq_entree, d_model)

        return sortie2


# Normalisation de couche
class NormalisationCouche(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(NormalisationCouche, self).__init__(**kwargs)

    def build(self, forme_entree):
        self.gamma = self.add_weight(name='gamma', shape=forme_entree[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=forme_entree[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(NormalisationCouche, self).build(forme_entree)

    def call(self, x):
        moyenne = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        ecart_type = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - moyenne) / (ecart_type + self.eps) + self.beta

    def compute_output_shape(self, forme_entree):
        return forme_entree


# Réseau neuronal avant
def reseau_neuronal_couche_avant(nb_unites, unites_intermediaires):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(unites_intermediaires, activation='relu'),
        tf.keras.layers.Dense(nb_unites, activation='relu')])


# Attention par produit scalaire pondéré
def scaled_dot_product_attention(q, k, v, masque):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)
    if masque is not None:
        scaled_attention_logits += (masque * -1e9)

    poids_attention = tf.nn.softmax(scaled_attention_logits, axis=-1)
    sortie = tf.matmul(poids_attention, v)
    return sortie


# Construction de la couche d'attention à plusieurs têtes
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads

        self.dot_attention = scaled_dot_product_attention

    def split_heads(self, x, taille_lot, profondeur):
        # Diviser en têtes, placer la dimension du nombre de têtes avant la longueur de seq
        x = tf.reshape(x, (taille_lot, -1, self.num_heads, profondeur))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, entrees, **kwargs):
        q, k, v, masque = entrees
        taille_lot = tf.shape(q)[0]
        numUnites = q.get_shape().as_list()[-1]
        # Profondeur après la division
        profondeur = numUnites // self.num_heads
        # Avant la division, réseau neuronal pour obtenir la sémantique de q, k, v
        wq = tf.keras.layers.Dense(numUnites)  # (taille_lot, long_seq, d_model)
        wk = tf.keras.layers.Dense(numUnites)
        wv = tf.keras.layers.Dense(numUnites)
        q = wq(q)
        k = wk(k)
        v = wv(v)

        # Division
        q = self.split_heads(q, taille_lot, profondeur)  # (taille_lot, num_têtes, long_seq_q, profondeur)
        k = self.split_heads(k, taille_lot, profondeur)  # (taille_lot, num_têtes, long_seq_k, profondeur)
        v = self.split_heads(v, taille_lot, profondeur)  # (taille_lot, num_têtes, long_seq_v, profondeur)
        # À travers la couche d'attention par produit scalaire pondéré
        attention_redimensionnee = self.dot_attention(q, k, v, masque)  # (taille_lot, num_têtes, long_seq_q, profondeur)

        # "Dimension de tête multiple" déplacée ensuite
        attention_redimensionnee = tf.transpose(attention_redimensionnee, [0, 2, 1, 3])  # (taille_lot, long_seq_q, num_têtes, profondeur)

        # Fusion de la "dimension de tête multiple"
        attention_concatenee = tf.reshape(attention_redimensionnee, (taille_lot, -1, numUnites))

        # Couche entièrement connectée
        dense = tf.keras.layers.Dense(numUnites)
        sortie = dense(attention_concatenee)

        return sortie

# Fonction de masquage
def masque_remplissage(seq):
    # Obtenir les éléments de remplissage pour lesquels la séquence est égale à 0
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # Ajouter une dimension pour la matrice d'attention
    return seq[:, np.newaxis, np.newaxis, :]  # (taille_lot, 1, 1, long_seq)

# Encodage de position
class EncodagePosition(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(EncodagePosition, self).__init__(**kwargs)

    def call(self, entrees):
        if self.embedding_dim is None:
            self.embedding_dim = int(entrees.shape[-1])

        # Matrice d'encodage de position
        matrice_encodage_position = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        # Application des fonctions sinus et cosinus à des positions alternées
        matrice_encodage_position[:, 0::2] = np.sin(matrice_encodage_position[:, 0::2])  # dim 2i
        matrice_encodage_position[:, 1::2] = np.cos(matrice_encodage_position[:, 1::2])  # dim 2i+1

        matrice_encodage_position = tf.cast(matrice_encodage_position, dtype=tf.float32)

        return matrice_encodage_position + entrees

if __name__ == "__main__":
    n_couches = 2
    d_model = 128
    num_heads = 4
    middle_units = 256
    max_seq_len = 40

    echantillons = 10
    entrainement = False

    masque_remplissage_liste = masque_remplissage(np.random.randint(0, 108, size=(echantillons, max_seq_len)))
    donnees_entree = tf.random.uniform((echantillons, max_seq_len, d_model))

    encodeur_exemple = Encodeur(n_couches, d_model, num_heads, middle_units, max_seq_len, entrainement)
    sortie_encodeur_exemple = encodeur_exemple([donnees_entree, masque_remplissage_liste])

    print(type(sortie_encodeur_exemple))
    print(type(tf.constant([1, 2, 3, 4, 5, 6])))
    sess = tf.compat.v1.InteractiveSession()  # Créer un nouveau graphique de calcul
    sess.run(tf.compat.v1.global_variables_initializer())  # Initialiser tous les paramètres
    print(sess.run(sortie_encodeur_exemple))
