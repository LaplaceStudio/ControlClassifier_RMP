�	Z��ڊ=%@Z��ڊ=%@!Z��ڊ=%@	g���?g���?!g���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Z��ڊ=%@;M�O�?A	��g��$@Y�MbX9�?*	�����k�@2�
RIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map�+e�X@!Q�PuW@)Ș���O@1��v�lW@:Preprocessing2s
<Iterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2jM�S�@!6
N�_�X@)!�lV}�?1]����@:Preprocessing2�
JIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip!�lV}@!X����W@)tF��_�?1+[]����?:Preprocessing2F
Iterator::Model������?!jRѿ�&�?)�&S��?1{	�%���?:Preprocessing2|
EIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shufflex$(~�@!0#0�W@)���_vO�?1ůQpZ��?:Preprocessing2�
ZIterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::TensorSlice��0�*�?!@X����?)��0�*�?1@X����?:Preprocessing2�
_Iterator::Model::Prefetch::Shuffle::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::Map::TensorSlice�&S��?!{	�%���?)�&S��?1{	�%���?:Preprocessing2P
Iterator::Model::Prefetch{�G�zt?!�#�g�;�?){�G�zt?1�#�g�;�?:Preprocessing2j
3Iterator::Model::Prefetch::Shuffle::MemoryCacheImplF�����@!vb'vb�X@)/n��r?1� ae��?:Preprocessing2Y
"Iterator::Model::Prefetch::Shufflex��#��@!�.@X��X@)a2U0*�c?1���?:Preprocessing2f
/Iterator::Model::Prefetch::Shuffle::MemoryCacheё\�C�@!��=5��X@)Ǻ���V?1�F�i�k�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9g���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	;M�O�?;M�O�?!;M�O�?      ��!       "      ��!       *      ��!       2		��g��$@	��g��$@!	��g��$@:      ��!       B      ��!       J	�MbX9�?�MbX9�?!�MbX9�?R      ��!       Z	�MbX9�?�MbX9�?!�MbX9�?JCPU_ONLYYg���?b Y      Y@q�ʂl3"@"�
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