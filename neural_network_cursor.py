import numpy as np
from typing import Tuple, List

class SimpleNN:
    """简单的单隐藏层神经网络
    结构：输入层(2) -> 隐藏层(3) -> 输出层(1)
    """
    def __init__(self, learning_rate: float = 0.1) -> None:
        """初始化神经网络
        结构：输入层(2) -> 隐藏层(3) -> 输出层(1)
        """
        # 初始化权重和偏置
        self.W1 = np.random.randn(3, 2)  # 隐藏层(3个神经元) x 输入层(2个特征)
        self.b1 = np.zeros((3, 1))       # 修改：隐藏层偏置 shape(3,1)
        self.W2 = np.random.randn(1, 3)  # 输出层(1个神经元) x 隐藏层(3个神经元)
        self.b2 = np.zeros((1, 1))       # 输出层偏置 shape(1,1)
        self.lr = learning_rate

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """sigmoid激活函数"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """sigmoid函数的导数: f'(x) = f(x)(1-f(x))"""
        sx = self.sigmoid(x)
        return sx * (1 - sx)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """前向传播
        
        Args:
            X: 输入数据，形状(batch_size, 2)
            
        Returns:
            输出值和中间计算结果
        """
        # 隐藏层
        # X: (batch_size, 2), W1: (3, 2) -> z1: (3, batch_size)
        z1 = np.dot(self.W1, X.T) + self.b1  # W1在前，b1: (3,1) 可以直接广播
        a1 = self.sigmoid(z1)                 # shape: (3, batch_size)
        
        # 输出层
        # a1: (3, batch_size), W2: (1, 3) -> z2: (1, batch_size)
        z2 = np.dot(self.W2, a1) + self.b2   # W2在前，b2: (1,1) 可以直接广播
        a2 = self.sigmoid(z2)                 # shape: (1, batch_size)
        
        # 保存中间值用于反向传播
        cache = {
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2
        }
        
        return a2.T, cache  # 最后输出时转置回 (batch_size, 1)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算均方误差损失
        
        E = 1/2 * Σ(y_true - y_pred)²
        """
        return np.mean(0.5 * (y_true - y_pred) ** 2)

    def backward(self, X: np.ndarray, y: np.ndarray, cache: dict) -> dict:
        """反向传播计算梯度
        
        Args:
            X: 输入数据，shape: (batch_size, 2)
            y: 真实标签，shape: (batch_size, 1)
            cache: 前向传播的中间结果
            
        Returns:
            各参数的梯度
        """
        m = X.shape[0]  # 批量大小
        
        # 获取前向传播的中间值
        a1, a2 = cache['a1'], cache['a2']  # a1: (3, batch_size), a2: (1, batch_size)
        z1, z2 = cache['z1'], cache['z2']  # z1: (3, batch_size), z2: (1, batch_size)
        
        # 输出层的误差：dE/da2 = -(y_true - y_pred) # 损失函数对输出层激活值的偏导
        dE_da2 = -(y.T - a2)  # shape: (1, batch_size)
        
        # 输出层权重的梯度
        # dE/dW2 = dE/da2 * da2/dz2 * dz2/dW2
        da2_dz2 = self.sigmoid_derivative(z2)  # shape: (1, batch_size)
        delta2 = dE_da2 * da2_dz2             # shape: (1, batch_size) # 输出层误差项
        dE_dW2 = np.dot(delta2, a1.T)         # shape: (1, 3)         # 输出层权重梯度
        dE_db2 = np.sum(delta2, axis=1, keepdims=True)  # shape: (1, 1)
        
        # 隐藏层的误差传播
        # dE/da1 = dE/da2 * da2/dz2 * dz2/da1
        dE_da1 = np.dot(self.W2.T, delta2)    # shape: (3, batch_size)
        
        # 隐藏层权重的梯度
        # dE/dW1 = dE/da1 * da1/dz1 * dz1/dW1
        da1_dz1 = self.sigmoid_derivative(z1)  # shape: (3, batch_size)
        delta1 = dE_da1 * da1_dz1             # shape: (3, batch_size)# 隐藏层误差项
        dE_dW1 = np.dot(delta1, X)            # shape: (3, 2)       # 隐藏层权重梯度 
        dE_db1 = np.sum(delta1, axis=1, keepdims=True)  # shape: (3, 1)
        
        return {
            'W1': dE_dW1, 'b1': dE_db1,
            'W2': dE_dW2, 'b2': dE_db2
        }

    def update_params(self, grads: dict) -> None:
        """使用梯度更新参数
        
        Args:
            grads: 各参数的梯度
        """
        self.W1 -= self.lr * grads['W1']  # W1: (3, 2)
        self.b1 -= self.lr * grads['b1']  # b1: (3, 1)
        self.W2 -= self.lr * grads['W2']  # W2: (1, 3)
        self.b2 -= self.lr * grads['b2']  # b2: (1, 1)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> List[float]:
        """训练网络
        
        Args:
            X: 训练数据，shape: (batch_size, 2)
            y: 目标值，shape: (batch_size, 1)
            epochs: 训练轮数
            
        Returns:
            损失历史
        """
        losses = []
        for epoch in range(epochs):
            # 前向传播
            y_pred, cache = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # 反向传播
            grads = self.backward(X, y, cache)
            
            # 更新参数
            self.update_params(grads)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return losses

def main() -> None:
    """示例：训练一个XOR分类器"""
    # 准备XOR数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 创建并训练网络
    nn = SimpleNN(learning_rate=0.1)
    losses = nn.train(X, y, epochs=5000)
    
    # 测试结果
    y_pred, _ = nn.forward(X)
    print("\n预测结果:")
    for i in range(len(X)):
        print(f"输入: {X[i]}, 目标: {y[i][0]}, 预测: {y_pred[i][0]:.4f}")

if __name__ == "__main__":
    main()
