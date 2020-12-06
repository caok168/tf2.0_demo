## Tensorflow模型保存与部署
### 实战
* 1、保存模型
* 2、模型转tflite
* 3、量化处理
* 4、Tensorflow.js部署模型
* 5、Android部署模型

### 模型保存
* 1、文件格式
   *  Checkpoint 与 graphdef（tf1.0）
   *  keras（hdf5），SavedModel（tf2.0）
* 2、保存的是什么？
   * 参数
   *  参数+网络结构
   
### TFLite
* 1、TFLite Converter
   *  模型转化
* 2、TFLite Interpreter
   *  模型加载
   *  支持android与ios设备
   *  支持多种语言
* 3、TFLite - FlatBuffer
   *  Google开源的跨平台数据序列化库
   *  优点
       *  直接读取序列化数据
       *  高效的内存使用和速度，无需占用额外内存
       *  灵活，数据前后向兼容，灵活控制数据结构
       *  使用少量的代码即可完成功能
       *  强数据类型，易于使用
* 4、TFLite - 量化
   *  参数从float变为8-bit整数
       *  模型准确率会有些许损失
       *  模型大小变为原来四分之一
   *  量化方法
       *  真实float值 = （int值 - 归零点）* 因子
       
### 实战
* keras-保存参数与保存模型 + 参数
* keras，签名函数到SavedModel
* keras，SavedModel，签名函数到具体函数
* keras，SavedModel，具体函数到tflite
* Tflite量化
* Tensorflow js部署模型
* Android部署模型

