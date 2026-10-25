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

5. Sãos as características pricipais de um tensor:
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

### Operações com tensores  

1. Unindo tensores  
	- Usado  `python torch.cat()` para concatenar ao longo de uma das dimensões
```pythons
import torch

tensor_1 = torch.ones(3,3)
tensor_2 = torch.zeros(3,3)	

#concatenando a primeira dimensão (linhas com dim=0)
tensor_3 = torch.cat([tensor_1, tensor_2], dim=0)
print (tensor_3)
```  

2. Multiplicando tensores  
```python
import torch

t1 = torch.ones(3, 3)
t2 = torch.ones(3, 3)

# Multiplicação matricial usa @ ou t1.matmul(t2)
t3 = t1@t2
# * faz multiplicação escalar 1 elemento por elemento
t4 = t1*t2
print(f'Tensor t3: {t3}')
print(f'Tensor t4: {t4}')

# Criando t5 com mesmo formato que t1 com valores aleatórios
t5 = torch.rand_like(t1)
print (f'Tensor t5: {t5}')

# multiplicando t1 pela sua transposta de t5 e salvando em t3
torch.matmul(t1, t5.T, out=t3)

print(f'Tensor t5 Transposta: {t5.T}')
print(f'Tensor t3 reescrito: {t3}')
```  

2. Transformando tensor em um tipo nativo  
```python
import torch

# Tensor inicial com vaários valores
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Soma todos os elementos m um valor escalar, porém ainda é um Tensor
soma = tensor.sum()
print(soma)                  # tensor(10.)
print(type(soma))            # <class 'torch.Tensor'>

# Extrai o valor do tensor em um tipo Python nativo
soma_item = soma.item()
print(soma_item)             # 10.0
print(type(soma_item))       # <class 'float'>
```  

3. Operações *inplace* (local)
	- São operações onde há quebra do gradiente no Pythorch
	- usa-se `_` ao lado do comando para executar o inplace
	- Ao quebrar o gradiene não é possível ver o histórico de modific~]oes dificultado o *backward*
	- Há um pequeno ganho em velocidade e memória com esse comando
	- ***Não é recomendado usar esse método***  
```python
import torch

t1 = torch.ones(3, 3)
t2 = torch.ones(3, 3)

t1.add_(t2) # soma t2 em t1 e guarda o resultado em t1

t1.copy_(t2) # copia t2 para dentro de t1

t1.t_() # transpõe t1 no próprio t1

t1.mul_(2) # multiplica  t1 por 2 e salva em t1

t1.zero_() # zera t1 e salva em t1

# Não há inplace para multiplicação matricial então é preciso reatribuir
t1 = t1@t2 
```

4. Convertendo para numpy  
```python
import torch
import numpy as np

#gerando a versão para numpy
t = torch.ones(5)
print(f"t: {type(t)}")
n = t.numpy()
print(f"n: {type(n)}")

# O numpy.array usa a mesma memória do tensor, logo as alterações se refletem
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# O mesmo se aplica a conversão de numpy em tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```  
6. Usando o acelerador  
	- Todos os comandos acima podem ser usados no acelerador o na cpu
	- para usar no acelerador basta mover os tensores com `.to()`
```python
import torch

tensor = torch.ones(5)
# Movendo o tensor para o acelerador se ele existir
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
```
