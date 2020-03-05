import tensorflow as tf

actList = dir(tf.keras.activations)

actListCalls = {}

for value in actList:
    if "__" not in value:
        actListCalls[value] = eval("tf.keras.activations." + value)

optList = dir(tf.keras.optimizers)

optListCalls = {}

for value in optList[:-4]:
    if "__" not in value:
        optListCalls[value] = eval("tf.keras.optimizers." + value)

lossList = dir(tf.keras.losses)

lossListCalls = {}

for value in lossList:
    if "__" not in value:
        lossListCalls[value] = eval("tf.keras.losses." + value)