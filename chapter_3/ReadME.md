## Tensorflow基础API使用
### tf基础API引入
* 1、基础API
* 2、基础API与keras的集成
  *  自定义损失函数
  *  自定义层次
* 3、@tf.function的使用
  *  图结果
* 4、自定义求导

### @tf.function
* 1、将python函数编译成图
* 2、易于将模型导出成为GraphDef+checkpoint或者SavedModel
* 3、使得eager execution可以默认打开
* 4、1.0的代码可以通过tf.function来继续在2.0里使用
  *  替代session

### API列表
* 1、基础数据类型
   * Tf.constant, tf.string
   * tf.ragged.constant, tf.SparseTensor, Tf.Variable
* 2、自定义损失函数—— Tf.reduce_mean
* 3、自定义层次—— Keras.layers.Lambda和继承法
* 4、Tf.function
   * Tf.function, tf.autograph.to_code, get_concrete_function
* 5、GraphDef
   * get_operations, get_operation_by_name
   * get_tensor_by_name, as_graph_def
* 6、自动求导
   * Tf.GradientTape
   * Optimzier.apply_gradients

