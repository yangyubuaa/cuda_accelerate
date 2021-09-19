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
##### 数据结构以及表示说明
1. 以上算法实现均使用了libtorch提供的at::Tensor类作为张量的载体,优点是可以直接与pytorch前端交互,作为c++与python传递的接口数据结构(torch已经为我们定义好了),我们可以拿到pytorch前端传递过来的tensor引用,通过tensor返回的数据指针((float*)input_tensor.data_ptr<float>())进行tensor数据的修改(cuda核函数的返回类型必须为void,所以只能通过传递指针进行修改).
2. 当然我们可以不使用tensor,我们可以使用c++的多维数组,仍然可以使用cuda加速计算,此项目作为pytorch的扩展,所以使用了at::Tensor作为数据载体(与pytorch保持一致).
##### at::Tensor数据结构说明
1. libtorch的at::Tensor是一层抽象,实际存储数据的是Storage类,Tensor只是Storage上的一个视图,我们调用reshape时只是更换了视角,底层的Storage其实并未变化.
2. Storage存储的数据是使用一维数组,高维张量的索引可以转化为一维数组的索引(这也是cuda核函数操作数组的方式),我们可以获取到数据指针,对pytorch前端传来的Tensor进行原地操作(不需要返回).
3. Storage高维张量存储的一维数组,和C++的高维数组存储的一维数组是一致的.Tensor只是高层封装.所以我们使用cuda操作Tensor和操作C++数组本质上其实是一致的.
---
作者:yangyu  
@电子邮箱:yangyu2019@buaa.edu.cn
