### 实现cpp以及cuda扩展pytorch样板代码
---
##### cpp部分
1. 在cpp文件中声明并实现两个函数,一个为正向传播,一个为反向传播.
2. 将两个函数分别绑定到python的前端中.
3. 使用libtorch提供的接口执行两个向量的相加操作
4. 返回相加后的向量

##### cuda部分
1. 在头文件中声明两个函数,这两个函数用于调用cuda kernel函数.
2. 在cpp文件中声明并实现forward函数以及backward函数并绑定python,这两个函数用于调用头文件中声明的两个函数.
3. 在cu文件中实现声明的两个函数,进行数据处理后(得到tensor的数据指针),传入调用的kernel函数.
4. 声明并实现kernel函数
---
##### 注意点
1. cpu不能访问显存
2. gpu上不能使用std命名空间，打印需要使用printf
3. gpu上不能使用tensor
4. tensor的data_ptr()使用方法：(float*)a.data_ptr<float>()，为指向tensor的数据指针
5. 使用cuda计算的步骤，计算出矩阵的全局索引，计算当前线程在数组中的位置，更新，注意判断数组越界
6. 越界判断：矩阵的话可以通过判断矩阵的m和n是否越界，注意cuda线程模型的x是纵坐标，y是横坐标，也可以通过矩阵的大小进行判断。
##### 项目说明
1. cpp_method是c++以及libtorch扩展pytorch的样板代码
2. cuda_method是使用cuda实现矩阵加法使用gpu并行运算进行加速
3. cuda_matmul是使用cuda实现矩阵乘法使用gpu并行运算进行加速(以上均实现grid-stride-loop,可以处理任意大小的矩阵,而不受cuda core以及SM流多处理器限制)
4. cuda矩阵乘法算法在cuda_matmul/gpu/matmul_cuda.cu文件
