�	��K7��$@��K7��$@!��K7��$@	⤨��?⤨��?!⤨��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��K7��$@�V�/�'�?A�i�q�N$@YM�O��?*	����Lз@2�
RIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map �~�:�@!)�D2IX@).���1�@1��Kj�>X@:Preprocessing2s
<Iterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2�St$�?@!o�,��X@),e�X�?18�Xb��?:Preprocessing2�
JIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip��d�`�@!kd�..nX@)�
F%u�?1h�T��?:Preprocessing2F
Iterator::Model�~j�t��?!"φ2�?);�O��n�?1�Y����?:Preprocessing2|
EIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle����9�@!���g}X@)V-��?1-I:��q�?:Preprocessing2�
ZIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::TensorSlice��ׁsF�?!|P��?)��ׁsF�?1|P��?:Preprocessing2�
_Iterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlicen���?!�[\��?)n���?1�[\��?:Preprocessing2P
Iterator::Model::Prefetch�~j�t�x?!"φ2�?)�~j�t�x?1"φ2�?:Preprocessing2j
3Iterator::Model::Prefetch::Shuffle::MemoryCacheImpl��&SE@!�����X@)Ǻ���v?1��9��?:Preprocessing2Y
"Iterator::Model::Prefetch::Shuffle� �	J@!�0y���X@)F%u�k?1|��z��?:Preprocessing2f
/Iterator::Model::Prefetch::Shuffle::MemoryCache-���F@!���	W�X@)��_�LU?1�M�֕?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9⤨��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�V�/�'�?�V�/�'�?!�V�/�'�?      ��!       "      ��!       *      ��!       2	�i�q�N$@�i�q�N$@!�i�q�N$@:      ��!       B      ��!       J	M�O��?M�O��?!M�O��?R      ��!       Z	M�O��?M�O��?!M�O��?JCPU_ONLYY⤨��?b Y      Y@qB�*:)"@"�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 