#!/usr/bin/python

import os, logging

import tensorflow as tf
from struct import pack, unpack

import numpy as np
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as K
import random, math
from src import config

def bitflip(f, pos):
	
	""" Single bit-flip in 32 bit floats """

	f_ = pack('f', f)
	b = list(unpack('BBBB', f_))
	[q, r] = divmod(pos, 8)
	b[q] ^= 1 << r
	f_ = pack('BBBB', *b)
	f = unpack('f', f_)
	return f[0]

class inject():
	def __init__(
		self, model, confFile, log_level="ERROR", **kwargs
		):

		# Logging setup
		logging.basicConfig()
		logging.getLogger().setLevel(log_level)
		logging.debug("Logging level set to {0}".format(log_level))

		# Retrieve config params
		fiConf = config.config(confFile)
		self.Model = model # No more passing or using a session variable in TF v2

		# Call the corresponding FI function
		fiFunc = getattr(self, fiConf["Mode"])
		fiFunc(model, fiConf, **kwargs)

	def layer_states(self, model, fiConf, **kwargs):
		
		""" FI in layer states """
		
		logging.info("Starting fault injection in a random layer")

		# Retrieve type and amount of fault
		fiFault = fiConf["Type"]
		fiSz = fiConf["Amount"]

		# Choose a random layer for injection
		randnum = random.randint(0, len(model.trainable_variables) - 1)

		# Get layer states info
		v = model.trainable_variables[randnum]
		num = v.shape.num_elements()

		if(fiFault == "zeros"):
			fiSz = (fiSz * num) / 100
			fiSz = math.floor(fiSz)

		# Choose the indices for FI
		ind = random.sample(range(num), fiSz)

		# Unstack elements into a single dimension
		elem_shape = v.shape
		v_ = tf.identity(v)
		v_ = tf.keras.backend.flatten(v_)
		v_ = tf.unstack(v_)

		# Inject the specified fault into the randomly chosen values
		if(fiFault == "zeros"):
			for item in ind:
				v_[item] = 0.
		elif(fiFault == "random"):
			for item in ind:
				v_[item] = np.random.random()
		elif(fiFault == "bitflips"):
			for item in ind:
				val = v_[item]
				pos = random.randint(0, 31)
				val_ = bitflip(val, pos)
				v_[item] = val_

		# Reshape into original dimensions and store the faulty tensor
		v_ = tf.stack(v_)
		v_ = tf.reshape(v_, elem_shape)
		v.assign(v_)

		logging.info("Completed injections... exiting")

	def layer_outputs(self, model, fiConf, **kwargs):

		""" FI in layer computations/outputs """

		logging.info("Starting fault injection in a random layer")

		# Retrieve type and amount of fault
		fiFault = fiConf["Type"]
		fiSz = fiConf["Amount"]

		# Get the input for which dynamic injection is to be done
		x_test = kwargs["x_test"]

		# Choose a random layer for injection
		randnum = random.randint(0, len(model.layers) - 2)

		fiLayer = model.layers[randnum]

		# Get the outputs of the chosen layer
		get_output = K.function([model.layers[0].input], [fiLayer.output])
		fiLayerOutputs = get_output([x_test])

		# Unstack elements into a single dimension
		elem_shape = fiLayerOutputs[0].shape
		fiLayerOutputs[0] = fiLayerOutputs[0].flatten()
		num = fiLayerOutputs[0].shape[0]

		if(fiFault == "zeros"):
			fiSz = (fiSz * num) / 100
			fiSz = math.floor(fiSz)

		# Choose the indices for FI
		ind = random.sample(range(num), fiSz)

		# Inject the specified fault into the randomly chosen values
		if(fiFault == "zeros"):
			for item in ind:
				fiLayerOutputs[0][item] = 0.
		elif(fiFault == "random"):
			for item in ind:
				fiLayerOutputs[0][item] = np.random.random()
		elif(fiFault == "bitflips"):
			for item in ind:
				val = fiLayerOutputs[0][item]
				pos = random.randint(0, 31)
				val_ = bitflip(val, pos)
				fiLayerOutputs[0][item] = val_

		# Reshape into original dimensions and get the final prediction
		fiLayerOutputs[0] = fiLayerOutputs[0].reshape(elem_shape)
		get_pred = K.function([model.layers[randnum + 1].input], [model.layers[-1].output])
		pred = get_pred([fiLayerOutputs])
		# return pred
		labels = np.argmax(pred, axis=-1)
		return labels[0]