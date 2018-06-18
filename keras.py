from keras.models import Model
from keras.layers import Input, Dense, CuDNNGRU, Embedding

import constants as const

NUM_UNITS = 512

# A list of event IDs e.g [[1, 23, 100]]
events = Input(shape=(const.SEQ_LEN,))
# One hot of the style
# style = Input(shape=(const.SEQ_LEN, const.NUM_STYLES))

events_distributed = Embedding(output_dim=NUM_UNITS, input_dim=const.NUM_ACTIONS)(events)
# [Batch, SEQ_LEN, 32]
# style_distributed = Dense(32)(style)

x = events_distributed

for _ in range(4):
    x = CuDNNGRU(units=NUM_UNITS)(x)

model = Model(inputs=[events], outputs=[x])
model.compile(loss='categorical_crossentropy', optimizer='adam')

# TODO:
# model.fit_generator()
