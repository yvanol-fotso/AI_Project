# import numpy as np
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()

# class Encodeur(tf.keras.layers.Layer):
#     def __init__(self, n_layers, d_model, num_heads, middle_units,
#                  max_seq_len, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
#         super(Encodeur, self).__init__(**kwargs)
#         self.n_layers = n_layers

#         self.embedding_position = EncodagePosition(sequence_len=max_seq_len, embedding_dim=d_model)
#         self.couche_encode = [CoucheEncodeur(d_model=d_model, num_heads=num_heads,max_seq_len=max_seq_len,
#                                             middle_units=middle_units,
#                                             epsilon=epsilon, dropout_rate=dropout_rate,
#                                             training=training)
#                              for _ in range(n_layers)]

#     def call(self, inputs, **kwargs):
#         emb, masque = inputs
#         emb = self.embedding_position(emb)
#         for i in range(self.n_layers):
#             emb = self.couche_encode[i](emb, masque)

#         return emb


# # Couche d'encodage
# class CoucheEncodeur(tf.keras.layers.Layer):
#     def __init__(self, d_model, num_heads, max_seq_len, middle_units, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
#         super(CoucheEncodeur, self).__init__(**kwargs)

#         self.mha = AttentionMultiTete(num_heads)
#         # self.ffn = reseau_transformation_point_a_point(d_model + max_seq_len, middle_units) ## Erreur de Dimension
#         self.ffn = reseau_transformation_point_a_point(d_model, middle_units)

#         self.layernorm1 = NormalisationCouche()
#         self.layernorm2 = NormalisationCouche()

#         self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
#         self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

#         self.training = training

#     def call(self, inputs, masque, **kwargs):
#         # Réseau d'attention multi-tête
#         sortie_att = self.mha([inputs, inputs, inputs, masque])
#         sortie_att = self.dropout1(sortie_att, training=self.training)
#         out1 = self.layernorm1(inputs + sortie_att)

#         # Réseau de transformation point à point
#         sortie_ffn = self.ffn(out1)
#         sortie_ffn = self.dropout2(sortie_ffn, training=self.training)
#         out2 = self.layernorm2(out1 + sortie_ffn)  # Problème de dimension ici

#         return out2

# # Normalisation de couche
# class NormalisationCouche(tf.keras.layers.Layer):
#     def __init__(self, epsilon=1e-6, **kwargs):
#         self.eps = epsilon
#         super(NormalisationCouche, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
#                                      initializer=tf.ones_initializer(), trainable=True)
#         self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
#                                     initializer=tf.zeros_initializer(), trainable=True)
#         super(NormalisationCouche, self).build(input_shape)

#     def call(self, x):
#         moyenne = tf.keras.backend.mean(x, axis=-1, keepdims=True)
#         ecart_type = tf.keras.backend.std(x, axis=-1, keepdims=True)
#         return self.gamma * (x - moyenne) / (ecart_type + self.eps) + self.beta

#     def compute_output_shape(self, input_shape):
#         return input_shape


# # Réseau de transformation point à point
# def reseau_transformation_point_a_point(numUnits, middle_units):
#     return tf.keras.Sequential([
#         tf.keras.layers.Dense(middle_units, activation='relu'),
#         tf.keras.layers.Dense(numUnits, activation='relu')])


# # Attention à produit scalaire équilibrée
# def attention_produit_scalaire_equilibre(q, k, v, masque):
#     matmul_qk = tf.matmul(q, k, transpose_b=True)
#     dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
#     attention_logits_mis_a_echelle = matmul_qk / tf.math.sqrt(dim_k)
#     if masque is not None:
#         attention_logits_mis_a_echelle += (masque * -1e9)

#     poids_attention = tf.nn.softmax(attention_logits_mis_a_echelle, axis=-1)
#     sortie = tf.matmul(poids_attention, v)
#     return sortie


# # Construction de la couche d'attention multi-tête
# class AttentionMultiTete(tf.keras.layers.Layer):
#     def __init__(self, num_heads, **kwargs):
#         super(AttentionMultiTete, self).__init__(**kwargs)
#         self.num_heads = num_heads
#         self.attention_produit_scalaire = attention_produit_scalaire_equilibre

#     def separation_tetes(self, x, batch_size, profondeur):
#         # Séparation des têtes, déplace la dimension du nombre de têtes avant la séquence
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, profondeur))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def call(self, inputs, **kwargs):
#         q, k, v, masque = inputs
#         batch_size = tf.shape(q)[0]
#         numUnits = q.get_shape().as_list()[-1]
#         profondeur = numUnits // self.num_heads

#         # Avant la séparation des têtes, réseau avant la séparation
#         wq = tf.keras.layers.Dense(numUnits)
#         wk = tf.keras.layers.Dense(numUnits)
#         wv = tf.keras.layers.Dense(numUnits)
#         q = wq(q)
#         k = wk(k)
#         v = wv(v)

#         # Séparation des têtes
#         q = self.separation_tetes(q, batch_size, profondeur)
#         k = self.separation_tetes(k, batch_size, profondeur)
#         v = self.separation_tetes(v, batch_size, profondeur)

#         # À travers la couche d'attention à produit scalaire équilibré
#         attention_mise_a_echelle = self.attention_produit_scalaire(q, k, v, masque)

#         # Déplacement de la dimension "têtes multiples"
#         attention_mise_a_echelle = tf.transpose(attention_mise_a_echelle, [0, 2, 1, 3])

#         # Fusion de la dimension "têtes multiples"
#         attention_concatenee = tf.reshape(attention_mise_a_echelle, (batch_size, -1, numUnits))

#         # Couche entièrement connectée
#         dense = tf.keras.layers.Dense(numUnits)
#         sortie = dense(attention_concatenee)

#         return sortie

# # Fonction de masquage
# def masque_remplissage(seq):
#     # Obtenir les éléments de remplissage (paddings) égaux à 0
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

#     # Élargir les dimensions pour la matrice d'attention
#     return seq[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)


# # Encodage des positions
# class EncodagePosition(tf.keras.layers.Layer):
#     def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
#         self.sequence_len = sequence_len
#         self.embedding_dim = embedding_dim
#         super(EncodagePosition, self).__init__(**kwargs)

#     def call(self, inputs):
#         if self.embedding_dim is None:
#             self.embedding_dim = int(inputs.shape[-1])

#         encodage_position = np.array([
#             [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
#             for pos in range(self.sequence_len)])

#         encodage_position[:, 0::2] = np.sin(encodage_position[:, 0::2])  # dim 2i
#         encodage_position[:, 1::2] = np.cos(encodage_position[:, 1::2])  # dim 2i+1

#         encodage_position = tf.cast(encodage_position, dtype=tf.float32)

#         return encodage_position + inputs

#     def compute_output_shape(self, input_shape):
#         return input_shape

# if __name__ == "__main__":
#     n_layers = 2
#     d_model = 128
#     num_heads = 4
#     middle_units = 256
#     max_seq_len = 40

#     samples = 10
#     training = False

#     masque_remplissage_liste = masque_remplissage(np.random.randint(0, 108, size=(samples, max_seq_len)))
#     donnees_entree = tf.random.uniform((samples, max_seq_len, d_model))

#     encodeur_exemple = Encodeur(n_layers, d_model, num_heads, middle_units, max_seq_len, training)
#     sortie_encodeur_exemple = encodeur_exemple([donnees_entree, masque_remplissage_liste])
#     print(type(sortie_encodeur_exemple))
#     print(type(tf.constant([1, 2, 3, 4, 5, 6])))
#     session = tf.compat.v1.InteractiveSession()
#     session.run(tf.compat.v1.global_variables_initializer())
#     print(session.run(sortie_encodeur_exemple))





import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class Encodeur(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                 max_seq_len, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(Encodeur, self).__init__(**kwargs)
        self.n_layers = n_layers

        self.embedding_position = EncodagePosition(sequence_len=max_seq_len, embedding_dim=d_model)
        self.couche_encode = [CoucheEncodeur(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len,
                                            middle_units=middle_units, epsilon=epsilon, 
                                            dropout_rate=dropout_rate, training=training)
                             for _ in range(n_layers)]

    def call(self, inputs, **kwargs):
        emb, masque = inputs
        emb = self.embedding_position(emb)
        for i in range(self.n_layers):
            emb = self.couche_encode[i](emb, masque)
        return emb

class CoucheEncodeur(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, max_seq_len, middle_units, epsilon=1e-6, dropout_rate=0.1, training=False, **kwargs):
        super(CoucheEncodeur, self).__init__(**kwargs)

        self.mha = AttentionMultiTete(num_heads)
        self.ffn = reseau_transformation_point_a_point(d_model, middle_units)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=epsilon)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.training = training

    def call(self, inputs, masque, **kwargs):
        # Réseau d'attention multi-tête
        sortie_att = self.mha([inputs, inputs, inputs, masque])
        sortie_att = self.dropout1(sortie_att, training=self.training)
        out1 = self.layernorm1(inputs + sortie_att)

        # Réseau de transformation point à point
        sortie_ffn = self.ffn(out1)
        sortie_ffn = self.dropout2(sortie_ffn, training=self.training)
        out2 = self.layernorm2(out1 + sortie_ffn)

        return out2

def reseau_transformation_point_a_point(numUnits, middle_units):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(numUnits, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LayerNormalization()
    ])

def attention_produit_scalaire_equilibre(q, k, v, masque):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    attention_logits_mis_a_echelle = matmul_qk / tf.math.sqrt(dim_k)
    if masque is not None:
        attention_logits_mis_a_echelle += (masque * -1e9)
    poids_attention = tf.nn.softmax(attention_logits_mis_a_echelle, axis=-1)
    sortie = tf.matmul(poids_attention, v)
    return sortie

class AttentionMultiTete(tf.keras.layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super(AttentionMultiTete, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_produit_scalaire = attention_produit_scalaire_equilibre

    def separation_tetes(self, x, batch_size, profondeur):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, profondeur))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        q, k, v, masque = inputs
        batch_size = tf.shape(q)[0]
        numUnits = q.get_shape().as_list()[-1]
        profondeur = numUnits // self.num_heads

        wq = tf.keras.layers.Dense(numUnits)
        wk = tf.keras.layers.Dense(numUnits)
        wv = tf.keras.layers.Dense(numUnits)
        q = wq(q)
        k = wk(k)
        v = wv(v)

        q = self.separation_tetes(q, batch_size, profondeur)
        k = self.separation_tetes(k, batch_size, profondeur)
        v = self.separation_tetes(v, batch_size, profondeur)

        attention_mise_a_echelle = self.attention_produit_scalaire(q, k, v, masque)

        attention_mise_a_echelle = tf.transpose(attention_mise_a_echelle, [0, 2, 1, 3])

        attention_concatenee = tf.reshape(attention_mise_a_echelle, (batch_size, -1, numUnits))

        dense = tf.keras.layers.Dense(numUnits)
        sortie = dense(attention_concatenee)

        return sortie

def masque_remplissage(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, np.newaxis, np.newaxis, :]

class EncodagePosition(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        super(EncodagePosition, self).__init__(**kwargs)
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        if self.embedding_dim is None:
            self.embedding_dim = int(inputs.shape[-1])

        position_indices = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        dimension_indices = tf.range(self.embedding_dim, dtype=tf.float32)[tf.newaxis, :]

        angle_rads = position_indices / tf.math.pow(10000.0, (2.0 * (dimension_indices // 2)) / tf.cast(self.embedding_dim, tf.float32))
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        position_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]

        position_encoding = tf.cast(position_encoding, dtype=tf.float32)
        return inputs + position_encoding

    def compute_output_shape(self, input_shape):
        return input_shape
