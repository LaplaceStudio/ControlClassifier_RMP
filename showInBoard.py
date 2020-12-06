import os
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.framework import ops

path=os.path.join(os.getcwd(),"model","saved_model.pb")\

with tf.compat.v1.Session(graph=ops.Graph()) as sess:
    with gfile.GFile(path, "rb") as f:
        data = compat.as_bytes(f.read())
        sm = saved_model_pb2.SavedModel()
        sm.ParseFromString(data)
        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)
        train_writer = tf.summary.create_file_writer("./log",tf.get_default_graph())
        #train_writer.add_graph(sess.graph)
        train_writer.set_as_default()
        train_writer.flush()
        train_writer.close()