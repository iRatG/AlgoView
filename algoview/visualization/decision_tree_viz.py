import numpy as np
import matplotlib.pyplot as plt
from ..base_algorithms.trees.decision_tree import DecisionTree

class DecisionTreeVisualizer:
    def __init__(self):
        # Генерируем простые данные для классификации
        np.random.seed(42)
        n_samples = 100
        
        # Создаем два признака
        X1 = np.random.uniform(0, 10, (n_samples, 2))
        y1 = (X1[:, 0] + X1[:, 1] > 10).astype(int)
        
        X2 = np.random.uniform(0, 10, (n_samples, 2))
        y2 = (X2[:, 0] - X2[:, 1] > 2).astype(int)
        
        self.X = np.vstack([X1, X2])
        self.y = np.hstack([y1, y2])
        
        # Создаем и обучаем модель
        self.tree = DecisionTree(max_depth=3)
        self.tree.fit(self.X, self.y)

    def plot_decision_boundary(self, node, bounds, depth=0):
        if node.is_leaf:
            x_min, x_max = bounds[0]
            y_min, y_max = bounds[1]
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                                np.linspace(y_min, y_max, 10))
            plt.fill([x_min, x_max, x_max, x_min],
                    [y_min, y_min, y_max, y_max],
                    alpha=0.2,
                    color='blue' if node.predicted_class == 0 else 'red')
            return

        if node.feature_index == 0:
            plt.plot([node.threshold, node.threshold], 
                    [bounds[1][0], bounds[1][1]], 
                    'k-', alpha=0.5)
            
            left_bounds = bounds.copy()
            left_bounds[0][1] = node.threshold
            self.plot_decision_boundary(node.left, left_bounds, depth + 1)
            
            right_bounds = bounds.copy()
            right_bounds[0][0] = node.threshold
            self.plot_decision_boundary(node.right, right_bounds, depth + 1)
        else:
            plt.plot([bounds[0][0], bounds[0][1]], 
                    [node.threshold, node.threshold], 
                    'k-', alpha=0.5)
            
            left_bounds = bounds.copy()
            left_bounds[1][1] = node.threshold
            self.plot_decision_boundary(node.left, left_bounds, depth + 1)
            
            right_bounds = bounds.copy()
            right_bounds[1][0] = node.threshold
            self.plot_decision_boundary(node.right, right_bounds, depth + 1)

    def visualize(self):
        plt.figure(figsize=(12, 6))
        
        # График данных и границ принятия решений
        plt.subplot(1, 2, 1)
        bounds = [[self.X[:, 0].min() - 1, self.X[:, 0].max() + 1],
                 [self.X[:, 1].min() - 1, self.X[:, 1].max() + 1]]
        self.plot_decision_boundary(self.tree.root, bounds)
        
        # Отображаем точки данных
        plt.scatter(self.X[self.y == 0][:, 0], self.X[self.y == 0][:, 1], 
                   c='blue', label='Класс 0', alpha=0.5)
        plt.scatter(self.X[self.y == 1][:, 0], self.X[self.y == 1][:, 1], 
                   c='red', label='Класс 1', alpha=0.5)
        
        plt.title('Границы принятия решений')
        plt.xlabel('Признак 1 (лимит по карте)')
        plt.ylabel('Признак 2 (сумма последнего платежа)')
        plt.legend()
        plt.grid(True)
        
        # Визуализация структуры дерева
        def plot_tree(node, x=0.5, y=0.9, dx=0.25, dy=0.2):
            if node is None:
                return
                
            if node.is_leaf:
                plt.text(x, y, f'Класс {node.predicted_class}',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black'))
            else:
                plt.text(x, y, f'X{node.feature_index} <= {node.threshold:.2f}',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black'))
                
                # Соединительные линии
                if node.left:
                    plt.plot([x, x-dx], [y-0.05, y-dy+0.05], 'k-')
                    plot_tree(node.left, x-dx, y-dy, dx/2, dy)
                if node.right:
                    plt.plot([x, x+dx], [y-0.05, y-dy+0.05], 'k-')
                    plot_tree(node.right, x+dx, y-dy, dx/2, dy)
        
        plt.subplot(1, 2, 2)
        plot_tree(self.tree.root)
        plt.title('Структура дерева решений')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualizer = DecisionTreeVisualizer()
    visualizer.visualize() 