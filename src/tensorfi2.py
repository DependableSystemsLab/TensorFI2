#!/usr/bin/python

import logging
import math
import random
from struct import pack, unpack

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from src import config
from src.utility import compute_fault_injected_prediction, get_fault_injection_configs


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
		"""Model graph and super nodes are model related. So, if multiple fault injection on same model is required, 
		then we can compute them priorly and pass them as parameter to this function to make the process faster """
		self.model_graph, self.super_nodes = get_fault_injection_configs(model)

		# Call the corresponding FI function
		fiFunc = getattr(self, fiConf["Target"])
		self.final_label = fiFunc(model, fiConf, **kwargs)


	def layer_states(self, model, fiConf, **kwargs):
		
		""" FI in layer states """
		
		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

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
					
					# If random bit chosen to be flipped
					if(fiConf["Bit"] == "N"):
						pos = random.randint(0, 31)

					# If bit position specified for flip
					else:
						pos = int(fiConf["Bit"])
					val_ = bitflip(val, pos)
					v_[item] = val_

			# Reshape into original dimensions and store the faulty tensor
			v_ = tf.stack(v_)
			v_ = tf.reshape(v_, elem_shape)
			v.assign(v_)

			logging.info("Completed injections... exiting")

		elif(fiConf["Mode"] == "multiple"):

			""" Multiple layer fault injection mode """

			logging.info("Starting fault injection in all layers")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Loop through each available layer in the model
			for n in range(len(model.trainable_variables) - 1):

				# Get layer states info
				v = model.trainable_variables[n]
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

						# If random bit chosen to be flipped
						if(fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)

						# If bit position specified for flip
						else:
							pos = int(fiConf["Bit"])
						val_ = bitflip(val, pos)
						v_[item] = val_

				# Reshape into original dimensions and store the faulty tensor
				v_ = tf.stack(v_)
				v_ = tf.reshape(v_, elem_shape)
				v.assign(v_)

			logging.info("Completed injections... exiting")


	def layer_outputs(self, model, fiConf, **kwargs):

		""" FI in layer computations/outputs """

		if(fiConf["Mode"] == "single"):

			""" Single layer fault injection mode """

			logging.info("Starting fault injection in a random layer")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Get the input for which dynamic injection is to be done
			x_test = kwargs["x_test"]

			# Choose a random layer for injection
			randnum = random.randint(0, len(model.layers) - 3) + 1

			# injection_layer_index = 30

			fiLayer = model.layers[randnum]

			# Get the outputs of the chosen layer
			get_output = K.function([model.layers[0].input], [fiLayer.output])
			fiLayerOutputs = get_output([x_test])

			# Unstack elements into a single dimension
			original_output_list = fiLayerOutputs[0]
			batch_size = len(original_output_list)
			faulty_output_list = None
			for i in range(batch_size):
				target_image = original_output_list[i]
				elem_shape = target_image.shape
				target_image = target_image.flatten()
				num = target_image.shape[0]
				if (fiFault == "zeros"):
					fiSz = (fiSz * num) / 100
					fiSz = math.floor(fiSz)

				# Choose the indices for FI
				ind = random.sample(range(num), fiSz)

				# Inject the specified fault into the randomly chosen values
				if (fiFault == "zeros"):
					for item in ind:
						target_image[item] = 0.
				elif (fiFault == "random"):
					for item in ind:
						target_image[item] = np.random.random()
				elif (fiFault == "bitflips"):
					for item in ind:
						val = target_image[item]
						if (fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)
						else:
							pos = int(fiConf["Bit"])
						val_ = bitflip(val, pos)
						target_image[item] = val_
				elif (fiFault == "none"):
					target_image = target_image

				# Reshape into original dimensions and get the final prediction
				target_image = target_image.reshape(elem_shape)
				target_image = np.expand_dims(target_image, axis=0)
				if faulty_output_list is None:
					faulty_output_list = target_image
				else:
					faulty_output_list = np.concatenate((faulty_output_list, target_image), axis=0)

			fiLayerOutputs[0] = faulty_output_list
			pred = compute_fault_injected_prediction(self.model_graph, self.super_nodes, model.layers, randnum, fiLayerOutputs, x_test)
			labels = np.argmax(pred, axis=-1)
			return labels[0]

		elif(fiConf["Mode"] == "multiple"):

			""" Multiple layer fault injection mode """

			logging.info("Starting fault injection in all layers")

			# Retrieve type and amount of fault
			fiFault = fiConf["Type"]
			fiSz = fiConf["Amount"]

			# Get the input for which dynamic injection is to be done
			x_test = kwargs["x_test"]

			# Get the outputs of the first layer
			get_output_0 = K.function([model.layers[0].input], [model.layers[1].output])
			fiLayerOutputs = get_output_0([x_test])

			# Loop through each available layer in the model
			for n in range(1, len(model.layers) - 2):

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
						if(fiConf["Bit"] == "N"):
							pos = random.randint(0, 31)
						else:
							pos = int(fiConf["Bit"])
						val_ = bitflip(val, pos)
						fiLayerOutputs[0][item] = val_

				# Reshape into original dimensions
				fiLayerOutputs[0] = fiLayerOutputs[0].reshape(elem_shape)

				"""
				Check if last but one layer reached;
				if not, replace fiLayerOutputs with the next prediction to continue
				"""
				if(n != (len(model.layers) - 3)):
					get_output = K.function([model.layers[n+1].input], [model.layers[n+2].output])
					fiLayerOutputs = get_output([fiLayerOutputs])

				# Get final prediction
				get_pred = K.function([model.layers[len(model.layers)-1].input], [model.layers[-1].output])
				pred = get_pred([fiLayerOutputs])

				# Uncomment below line and comment next two lines for ImageNet models
				# return pred
				labels = np.argmax(pred, axis=-1)
				return labels[0]
				
				logging.info("Completed injections... exiting")				