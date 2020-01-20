'''
Universidade Federal do Ceará - UFC
Departamento de Estatística e Matemática Aplicada - DEMA
Introdução à Mineração de Dados
Prof : Tibérius de Oliveira e Bonates

Trabalho I - Aprendizado por Disjunções de Monômios - Versao K <= N,onde N é o numero de caracteristicas.

Aluno : Thyago Freitas da Silva - 392035

Dicionario dos métodos da classe "funçoes_uteis"

1. split_string()
Método criada pra auxiliar na construçao da matriz com as amostras lidas no arquivo,a mesma recebe um linha do arquivo
e quebra a linha em n items,onde n é o numero de caracteres. Ex : linhax = ['0100'],retorno da funcao seria linhax = ['0','1','0','0']
Observação : Ao ler a linha do arquivo,cada linha vem com "\n" no final,como nao seria util mante-lo na matriz de dados, só percorremos a 
string de entrada associada a linha ate o penultimo caractere,assim descartando o "\n"

2. validar_lista()
O método foi criada para ajudar a resolver o problema de que durante a criaçao dos monomios,monomios descartaveis sao gerados,como por 
exemplo (1,-1,2),a presenca do 1 e seu negado tornam a hipotese descartavel. Logo, ao receber uma lista,ex = [1,-1,2],convertemos todos os
elementos da lista para positivos,ex = [1,1,2],eliminamos as duplicadas,logo ex = {1,2},e para checarmos se a lista é valida,basta comparar
o tamanho do conjunto resultando com o da lista de entrada,retornando TRUE quando a lista for valida,e FALSE caso o contrario

3. ler_amostras()
Método para ler as amostras do arquivo,onde cada linha primeiro passa pela funçao split_string para que a mesma volte parcialmente formatada
do jeito que as outras funçoes que se utilizem das amostras precisam. Vale notar que a funçao converte todos os valores lidos para inteiros
utilizando a funçao map onde a funçao argumento é int(),a real responsavel pela conversao. Essa conversao de tipos foi necessaria pois ao
aplicar funçoes como a any(aplica uma OR nos elementos de uma lista) e all(aplica um AND nos elementos de uma lista),o resultado quando era
utilizado string nao era coerente. A funçao retorna uma matriz formatada,onde cada linha é um valor da amostra presente no arquivo.

4. gerar_literais()
Método responsavel por gerar todos os literais,a mesma recebe a matriz de amostras,calcula a quantidade de caracteristicas (len(amostras[0]) - 1)
e gera o vetor de literais. O -1 existe para que possamos descartar a ultima coluna da amostra,que seria a classe da amostra.

5. gerar_mapa_conceitos()
O método recebe o vetor de literais e o limite maximo do grau dos monomios a serem gerados,tem por default K = 3, por causa da atividade 
padrao. Aqui é utilizado a funçao combinations do modulo itertools como visto em aula. Um vetor H é criado para armazenar todas as possibi
lidade de monomios com grau entre 1 e K,mas antes de armazenar os monomios,como a funçao gera monomios descartaveis,o resultado da funçao
passa primeiro por um filtro que se utilizada da funçao validar_lista() discutida previamente. Dessa forma,apenas os vetores de monomios de
grau 1<=x<=k validos sao adicionados ao mapa/vetor de conceitos H.

6. inserir_no_conceito()
Este método "injeta" os valores reais da amostra no conceito criado,retornando uma lista com os valores da amostra aplicado no conceito. 
Por exemplo, se tivermos o conceito c = (1,-3,4) e a amostra = [1,1,1,0,0,0],temos que o valor 1 do conceito representa a caracteristica na
posiçao 1-1,ou seja, o valor da amostra que esta na posiçao 0,que é 1,ja o valor -3,representa o valor invertido da amostra na posiçao 2
ou seja,0. O trecho mais importante do método é a funcao lambda f responsavel por fazer a inserçao e inversao de valores,quando necessario,
utilizando como suporte o dicionario "troca" que mapeia as inversoes. Se o valor do conceito for negativo,significa que antes de inserir 
é necessario inverter o valor da amostra,fazemos isso utilizando o dicionario "troca".

7. printar_rxd()
Esse método foi criado apenas para mostrar para o usuario o resultado de forma mais amistosa,ou seja, caso o H resultante seja (1,-2),(3,1,2)
o resultado mostrado para o usuario será (1,-2) ∨ (3,1,2). Fazemos isso concatenando e fazendo uso do format do print,dessa forma criando
a string a ser printada.O trecho "resul[:(len(resul)-3)]" foi utilizado para descartar a possibilidade de acontecer algo do tipo 
" (1,-2) ∨ (3,1,2) ∨ "


X. Implementaçao do algoritmo de Aprendizado por disjunçoes de monomios
A função recebe 2 parametros,o vetor de conceitos H e a matriz de amostras lida do arquivo.
O primeiro for é utilizado para percorrer a matriz de amostras e o segundo para percorrer todos os monomios armazenados em H.
Sempre que o valor da classe da amostra é 0,é checado se o resultado daquele monomio utilizando os dados da amostra é 1,se for,o monomio
deve ser descartado,caso seja 0, checamos o proximo monomio e repetimos o procedimento,ate que todo os monomios sejam analisados para a amostra 1.
Esse procedimento é realizado para todas as amostras,de forma que o resultado é um vetor que possui todos os monomios condizentes com o conjunto
de amostras.
Para sabermos o resultado do monomio,basta utilizar a funçao all() que aplica a operaçao AND item a item de uma lista,ou seja, em um caso
onde x = [1,1,0] o valor retornado seria False(0),caso x = [1,1,1] o resultado seria True(1). A variavel K é utilizada na funçao para sabermos
em qual indice esta o monomio no vetor H,de forma que podemos utilizar essa variavel na hora de usar a funçao del() para excluir o monomio
invalido do vetor H.

'''

import itertools as it

class funcoes_uteis:
    def split_string(self,string):
        return [letra for letra in string[:-1]]
    def validar_listas(self,lista):
        return len({abs(a) for a in lista}) == len(lista)
    def ler_amostras(self):
        amostras = []
        with open("dados.txt", "r") as arquivo:
            for linha in arquivo:
                amostras.append(list(map(int,self.split_string(linha))))
        return amostras
    def gerar_literais(self,amostras):
        n_caracteristicas = len(amostras[0]) - 1
        literais = [x for x in range(1,n_caracteristicas+1)] + [-x for x in range(1,n_caracteristicas+1)]
        return literais
    def gerar_mapa_conceitos(self,literais,k=3):
        H = []
        for i in range(1,k+1):
            H += filter(self.validar_listas,it.combinations(literais,i))
        return H
    def inserir_no_conceito(self,conceito,amostra):
        trocar = {1 : 0 ,0 : 1}
        f = lambda indice: amostra[abs(indice)-1] if indice> 0  else trocar[amostra[abs(indice)-1]]
        return [f(x) for x in conceito]
    def printar_rxd(self,H):
        resul = ""
        for h in H:
            resul += "{} ∨ ".format(h)
        return "Resultado Final : {}".format(resul[:(len(resul)-3)])
    
def aprendizado_por_H(H,amostras):
    f = funcoes_uteis()
    for i in amostras:
        k = 0
        for j in H:
            resul = all(f.inserir_no_conceito(j,i))
            if i[-1] == 0 and resul == True:
                del (H[k])
            k+=1
    return H
        
# INICIO
fun = funcoes_uteis()
amostras_resultado = fun.ler_amostras()
literais_resultado = fun.gerar_literais(amostras_resultado)             
H_resultado        = fun.gerar_mapa_conceitos(literais_resultado)   # CASO QUEIRA UM GRAU DIFERENTE DE 3,BASTA PASSAR UM SEGUNDO PARAMETRO 
                                                                    # PARA A FUNCAO gerar_mapa_conceitos,ONDE ESSE PARAMETRO É O GRAU LIMITE 
                                                                    # DESEJADO. POR PADRAO ESSE GRAU LIMITE ESTA COMO 3.        
H_resultado_final  = aprendizado_por_H(H_resultado,amostras_resultado)
print(fun.printar_rxd(H_resultado_final))



# DESCOMENTE ESSE TRECHO DE CODIGO CASO QUEIRA VER O "CONCEITO RESULTADO" EM AÇAO SENDO APLICADO A CADA AMOSTRA
# for i in amostras_resultado:
#    resul = []
#    for h in H_resultado_final:
#        resul.append(all(fun.inserir_no_conceito(h,i[:-1]))) 
#    print("Real = {}, \t Previsto = {}".format(bool(i[-1]),any(resul)))