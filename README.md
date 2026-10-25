# Instalado o Pytorch
- Para instalar o Pytorch execute os comando:
```python  
pip3 install torch torchvision
```
- Esse comando irá instalar o ***Pytorch*** e o ***torhvision***
	- torch é o framework principal do Pytorch
	- torchvision é framework de visão computacional
  
- Para saber se  código funcionou execute:
```python  
import torch
x = torch.rand(5, 3)
print(x)
```  
- O resultado será um tensor de 3 colunas e 5 linhas com valores entre 0 e 1.
  
  

# Tensores  
- Tensores para a computação podem ser entendidos como arrays multidimensionais de ordem *k*, onde k é o número de coordenadas para alcançar um valor.
	- *GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. Deep Learning. Cambridge: MIT Press, 2016. Capítulo 2 (Linear Algebra), Seção 2.1.*
- Assim:
	- **Escalar (k=0)**: um número propriamente dito.
	- **Vetor (k=1)**: Uma lista linear com os números.
	- **Matriz (k=2)**: Uma lista bidimensional com linha e coluna.
	- **Tensor (k=3+)**: Uma generalização podendo ter k eixos.  
		- Um exemplo é uma imagem que possui altura, largura e canal de cor (H, W, C)  
- Os tensores são a estrutura básica de dados das redes neurais profundas, codificando entrada, saída e seus parâmetros.  

## Iniciando tensores  
- Os tensores do Pytorch são semelhantes aos [ndarrays do numpy](https://numpy.org/doc/stable/reference/arrays.ndarray.html), com a diferença de serem acelerados por placas gráficas e aceleradores de hardware.  

### Inicializando o Tensor  
1. Os tensores podem ser **iniciados por dados diretos**:  
```python
import torch

data = [[1, 2],[3, 4]]
tensor_data = torch.tensor(data)
print (tensor_data)
```  
2. Os tensores podem **usar dados do NumPy**:  
```python
import torch
import numpy as np

data = [[1, 2],[3, 4]]
np_array = np.array(data)
tensor_np = torch.from_numpy(np_array)
print (tensor_np) 
```  
3. Podemos **copiar as propriedade de outro tensor**:  
```python
import torch

data = [[1, 2],[3, 4]]
tensor_data = torch.tensor(data)
# mantém a propriedade de tensor_data, mas preenche com valor 1
tensor_ones = torch.ones_like(tensor_data) 
print(f"Ones Tensor: \n {tensor_ones} \n")

# mantém a propriedade de tensor_data
# altera os valores para uma distribuição normal entre 0 e 1
# Para não acusar erro, altera o dtype para aceitar float
tensor_rand = torch.rand_like(tensor_data, dtype=torch.float)
print(f"Random Tensor: \n {tensor_rand} \n")
```
4. Preenchendo tensores com **valores aleatórios ou constantes no formato especificado**:
```python
import torch

shape = (2,3) # tensor com 2 linhas e 3 colunas
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#rand preenche aleatoriamente de forma uniforme com valor entre 0 e 1
print(f"Tensor aleatório: \n {rand_tensor} \n")
print(f"Tensor com valor 1: \n {ones_tensor} \n")
print(f"Tensor com valor zero: \n {zeros_tensor}")
```  

### Atributos do tensor
- Sãos as características pricipais de um tensor:
	 - Sua forma
	 - Tipo de dados
	 - Onde está armazenado
```python
import torch

tensor = torch.rand(3,4)

print(f"Forma do tensor: {tensor.shape}")
print(f"Tipos do tensor tensor: {tensor.dtype}")
print(f"Local de armazenamento: {tensor.device}")
```
