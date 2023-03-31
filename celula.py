from numpy import random
import numpy as np

class McCullochPitts():

    def _init_(self, compuerta = "AND", epoch = 50, n_bits = 2):
        self.n_bits = n_bits
        self.compuerta = compuerta
        self.epoch = epoch
        if compuerta == "NOT":
            print("Para la compuerta NOT el numero de bits es, es por defecto, 1")
            self.n_bits = 1
        self.tt = self._tabla_de_verdad(self.n_bits, self.compuerta) #dict
        self.pesos_sinapticos = []
        self.umbral = np.random.uniform(10) #

    def _tabla_de_verdad(self, n_bits, compuerta = "AND"):
        matrix = []
        aux = {}
        tt = {}
        for i in range(n_bits):
            aux[i] = 2**(n_bits-(i+1))
        for k,v in aux.items():
            matrix.insert(k,[])
            bit_actual = 1
            for _ in range(2**n_bits):
                if matrix[k][-v:].count(bit_actual) == v:
                    matrix[k].append(1^bit_actual)
                    bit_actual = 1^bit_actual
                else:
                    matrix[k].append(bit_actual)
        for j in range(len(matrix[0])):
            expression = []
            for i in range(len(matrix)):
                expression.append(matrix[i][j])
            if compuerta == "AND":
                tt[str(expression)] = True if expression.count(1) == n_bits else False
            if compuerta == "OR":
                tt[str(expression)] = True if expression.count(1) >= 1 else False
            if compuerta == "NOT":
                tt[str(expression)] = not expression[0]
        return tt
    
    def _fit(self):
        import json 
        correct = False
        pesos_sinapticos = []
        actual_epoch = 0
        self.pesos_sinapticos = np.zeros(self.n_bits)
        
        while actual_epoch < self.epoch and not correct:
            aux_correcto = 0
            for k,v in self.tt.items():
                input_=np.asarray(json.loads(k))
                suma_ponderada = np.dot(self.pesos_sinapticos,input_)
                predict = True if suma_ponderada > self.umbral else False
                error = v - predict 
                self.pesos_sinapticos = self.pesos_sinapticos + (error*input_)
                self.umbral = self.umbral + error
                if error == 0:
                    aux_correcto += 1
                    
            if aux_correcto == len(self.tt):
                correct = True          
            actual_epoch += 1
        return correct, actual_epoch
    
    def entrenar(self):
        result, actual_epoch = self._fit()
        if result:
            print('APRENDIZAJE EXITOSO EN LA  EPOCA: {}'.format(actual_epoch))
            print('Valor del umbral = {}'.format(self.umbral))
            print('Valor de los pesos = {}'.format(self.pesos_sinapticos))
        else:
            print('APRENDIZAJE NO EXITOSO')
            print('Valor del umbral = {}'.format(self.umbral))
            print('Valor de los pesos = {}'.format(self.pesos_sinapticos))

    def evaluar(self, bits = []):
        if len(bits) != self.n_bits:
            print('La cantidad de bits dada no es igual a el numero de bits entrenados')
        else:
            try:
                suma = 0
                for i, bit in enumerate(bits):
                    suma += bit* self.pesos_sinapticos[i]
                result = True if suma > self.umbral else False
            except IndexError:
                raise IndexError("La neurona no ha sido entrenada")
            return result