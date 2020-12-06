## Tensorflow dataset使用

### Tf.data API使用
* 1、Dataset基础API使用
* 2、Dataset读取csv文件
* 3、Dataset读取tfrecord文件

### API列表
* 1、Dataset基础使用
    * tf.data.Dataset.from_tensor_slices
    * repeat, batch, interleave, map, shuffle, list_files,
* 2、csv
    * tf.data.TextLineDataset, tf.io.decode_csv
* 3、Tfrecord
    * tf.train.FloatList, tf.train.Int64List, tf.train.BytesList
    * tf.train.Feature, tf.train.Features, tf.train.Example
    * example.SerializeToString
    * tf.io.ParseSingleExample
    * tf.io.VarLenFeature, tf.io.FixedLenFeature
    * tf.data.TFRecordDataset, tf.io.TFRecordOptions