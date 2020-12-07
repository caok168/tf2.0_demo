## Tensorflow 分布式训练
### 理论部分
* 1、GPU设置
* 2、分布式策略

### 实战部分
* 1、GPU设置实战
* 2、分布式策略实战

### GPU设置
* 1、默认用全部GPU并且内存全部占满
* 2、如何不浪费内存和计算资源？
    * 内存自增长
    * 虚拟设备机制
* 3、多GPU使用
    * 虚拟GPU & 实际GPU
    * 手工设置 & 分布式机制

### API列表
* 1、tf.debugging.set_log_device_placement（打印一些信息，某个变量分配在哪个设备上）
* 2、tf.config.experimental.set_visible_devices（本进程所见的设备）
* 3、tf.config.experimental.list_logical_devices（获取逻辑设备）
* 4、tf.config.experimental.list_physical_devices（获取物理设备）
* 5、tf.config.experimental.set_memory_growth（内存自增）
* 6、tf.config.experimental.VirtualDeviceConfiguration（建立逻辑分区）
* 7、tf.config.set_soft.device_placement（自动把计算分配到设备上）

### GPU实战
### 分布式策略
* 为什么需要分布式
    * 1、数据量太大
    * 2、模型太复杂

### 分布式策略
* 1、MirroredStrategy
* 2、CentralStorageStrategy
* 3、MultiworkerMirroredStrategy
* 4、TPUStrategy
* 5、ParameterServerStrategy

#### MirroredStrategy（镜像策略）
* 1、同步式分布式训练
* 2、适用于一机多卡
* 3、每个GPU都有网络结果的所有参数，这些参数会被同步
* 4、数据并行
    * Batch数据切为N份分给各个GPU
    * 梯度聚合然后更新给各个GPU上的参数

#### CentralStorageStrategy（）
* 1、MirroredStrategy的变种
* 2、参数不是在每个GPU上，而是存储在一个设备上
    * CPU或者唯一的GPU上
* 3、计算是在所有GPU上并行的
* 除了更新参数的计算之外

#### MultiworkerMirroredStrategy
* 1、类似于MirroredStrategy
* 2、适用于多机多卡


#### TPUStrategy
* 1、类似于MirroredStrategy
* 2、使用在TPU上的策略


#### ParameterServerStrategy
* 1、异步分布式
* 2、更加适用于大规模分布式系统
* 3、机器分为Parameter Server和worker两类
    * Parameter server负责整合梯度，更新参数
    * Worker负责计算，训练网络

#### 同步与异步的优劣
* 1、多机多卡
    * 异步可以避免短板效应
* 2、一机多卡
    * 同步可以避免过多的通信
* 3、异步的计算会增加模型的泛化能力
    * 异步不是严格正确的，所以模型更容忍错误

#### 实战
* MirroredStrategy
* 在keras模型上的使用
* 在estimator的使用
* 在自定义训练流程上的使用
